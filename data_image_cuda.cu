/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// C standard libraries
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

// Unix library
#include <unistd.h>

// Imported Libraries
#include <sw/redis++/redis++.h>
#include <curl/curl.h>
#include "vips/vips.h"
#include <pqxx/pqxx>

// C++ standard libraries
#include <iostream>
#include <string>
#include <memory>
#include <utility>
#include <iomanip>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;
using namespace pqxx;
using namespace sw::redis;

#define MAX_POSTGRES_QUERY_SIZE 1024*10
#define REDIS_QUEUE_NAME "data_image_qda:queue"
#define REDIS_STATUS_DIR_NAME "data_image_qda:status"

enum QualityCalcMethod { QualityLogExpSum = 0, QualityFirst = 1 };

/**
* Point-In-Polygons for a block using CUDA
*/
__global__ void PointsInPolygonsCUDA(long start_lat, long start_lng, long image_size,
	long num_polys, double quality_scale, enum QualityCalcMethod quality_calc_method, double quality_calc_value,
	unsigned char *image_mem, char *found_mem, char *all_blank, long *vectors, double *poly_values, long *vector_lengths) {

	long x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= image_size || y >= image_size) return;
	long lat, lng, image_pos;
	lat = start_lat + y;
	lng = start_lng + x;
	image_pos = y * image_size + x;

	double value = 0;
	for (long i = 0, vector_ind = 0, j, intersections; i < num_polys; i++) {
		// Point in Polygon calculation
		for (intersections = 0, j = 0; j < vector_lengths[i]; j++, vector_ind += 4) {
			if (((vectors[vector_ind + 1]>lat) != (vectors[vector_ind + 3]>lat)) &&
				(lng < ((long long)vectors[vector_ind + 2] - vectors[vector_ind]) * ((long long)lat - vectors[vector_ind + 1]) / ((long long)vectors[vector_ind + 3] - vectors[vector_ind + 1]) + vectors[vector_ind])) {
				intersections++;
			}
		}
		if ((intersections & 1) == 1) {
			// Add power of value to sum
			if (quality_calc_method == QualityLogExpSum) {
				value += pow(quality_calc_value, poly_values[i]);
			}
			// Halt on first value
			else if (quality_calc_method == QualityFirst)
			{
				value = poly_values[i];
				break;
			}
		}
	}
	//printf("value: %f\n", value);
	
	if (value) {
		//*all_blank = 0;
		switch (quality_calc_method) {
		case QualityLogExpSum:
			value = log(value) / log(quality_calc_value);
			break;
		case QualityFirst:
			found_mem[image_pos] = 1;
			break;
		default:
			break;
		}
	}
	value *= quality_scale;
	unsigned long fixed_value;
	if (value > UINT32_MAX)
		fixed_value = UINT32_MAX;
	else if (value < 0)
		fixed_value = 0;
	else
		fixed_value = value;

	image_pos = image_pos * 4;
	image_mem[image_pos] = (fixed_value >> 24) & 0xFF; // red
	image_mem[image_pos + 1] = (fixed_value >> 16) & 0xFF; // green
	image_mem[image_pos + 2] = (fixed_value >> 8) & 0xFF; // blue
	image_mem[image_pos + 3] = fixed_value & 0xFF; // alpha
}
//
/**
* Return an array of char poly_values that is equal to whether this point is in polygon[i]
*/

int PointInPolygonsImage(void **image, size_t *png_size, long start_lat,
	long start_lng, long image_size, double quality_scale, enum QualityCalcMethod quality_calc_method,
	double quality_calc_value, long *vectors, long total_length, double *poly_values, long num_polys, long *vector_lengths) {
	cudaStream_t stream;

	// Allocate device memory
	long vectors_mem_size = total_length * 4 * sizeof(*vectors);
	long vector_lengths_mem_size = num_polys * sizeof(*vector_lengths);
	long values_mem_size = num_polys * sizeof(*poly_values);
	long found_mem_size = image_size * image_size * sizeof(unsigned char);
	long image_mem_size = found_mem_size * 4;
	long *d_vectors;
	long *d_vector_lengths;
	unsigned char *d_image_mem, *h_image_mem;
	char *d_found_mem, *h_found_mem, *d_all_blank, h_all_blank[1] = { 1 };
	double *d_poly_values;

	long max_vector_lengths = 0;
	for (long i = 0; i < num_polys; i++)
		if (vector_lengths[i] > max_vector_lengths)
			max_vector_lengths = vector_lengths[i];


	switch (quality_calc_method) {
	case QualityFirst:
		if (!(h_found_mem = reinterpret_cast<char *>(malloc(found_mem_size)))) {
			fprintf(stderr, "Failed to allocate host found mem!\n");
			exit(EXIT_FAILURE);
		}
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_found_mem), found_mem_size));
		break;
	default:
		break;
	}


	if (!(h_image_mem = reinterpret_cast<unsigned char *>(malloc(image_mem_size)))) {
		fprintf(stderr, "Failed to allocate host image mem!\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vectors), vectors_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vector_lengths), vector_lengths_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_poly_values), values_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_image_mem), image_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_all_blank), sizeof(char)));
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// copy host memory to device
	checkCudaErrors(cudaMemcpyAsync(d_vectors, vectors, vectors_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_poly_values, poly_values, values_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_vector_lengths, vector_lengths, vector_lengths_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_all_blank, h_all_blank, sizeof(char), cudaMemcpyHostToDevice, stream));

	// Setup execution parameters
	dim3 threads(28, 28);
	dim3 grid((image_size + threads.x - 1) / threads.x, (image_size + threads.y - 1) / threads.y);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, stream));
	printf("Sending grid: [%d,%d]\n", grid.x, grid.y);
	// Performs warmup operation using matrixMul CUDA kernel
	PointsInPolygonsCUDA << < grid, threads, 0, stream >> > (start_lat, start_lng, image_size,
		num_polys, quality_scale, quality_calc_method, quality_calc_value,
		d_image_mem, d_found_mem, d_all_blank, d_vectors, d_poly_values, d_vector_lengths);
	cudaError_t error = (cudaGetLastError());

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code '%s')!\n", cudaGetErrorString(error));
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	printf(
		"Time= %.3f msec",
		msecTotal);
	std::cin.ignore();

	// Copy result from device to host
	checkCudaErrors(cudaMemcpyAsync(h_all_blank, d_all_blank, sizeof(char), cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(h_image_mem, d_image_mem, image_mem_size, cudaMemcpyDeviceToHost, stream));
	switch (quality_calc_method) {
	case QualityFirst:
		checkCudaErrors(cudaMemcpyAsync(h_found_mem, d_found_mem, found_mem_size, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaFree(d_found_mem));
		break;
	default:
		break;
	}
	checkCudaErrors(cudaStreamSynchronize(stream));

	// Clean up memory
	checkCudaErrors(cudaFree(d_all_blank));
	checkCudaErrors(cudaFree(d_vectors));
	checkCudaErrors(cudaFree(d_image_mem));
	checkCudaErrors(cudaFree(d_poly_values));
	checkCudaErrors(cudaFree(d_vector_lengths));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	long *value_at;
	for (long y = 0, x; y < image_size; y++) {
		printf("y: %ld\n", y);
		for (x = 0; x < image_size; x++) {
			value_at = (long *)(h_image_mem + (y*image_size + x) * 4);
			printf("x: %ld, = %lu\n", x, *value_at);
		}
	}

	free(h_image_mem);
	switch (quality_calc_method) {
	case QualityFirst:
		free(h_found_mem);
		break;
	default:
		break;
	}
	//if (correct) {
	return EXIT_SUCCESS;
	//} else {
	//    return EXIT_FAILURE;
	//}
}

void JsonGeometryToVectors(string json_array, double multiply_const, long **vectors, long *total_length, long *vector_length) {
	size_t pos = 0, coord_ind, end_pos, current_pos;
	*vector_length = 0;
	float lat_flt, lng_flt;
	long lat_1, lng_1, lat_2, lng_2;
	while(json_array.find_first_of('[', pos) != string::npos) { // Polygon
		pos++;
		while(json_array[pos] == '[') { // Polygon or Hole
			pos++;
			coord_ind = 0;
			while(json_array[pos] == '[') { // Coord
				pos++;
				end_pos = json_array.find_first_of(']', pos);
				sscanf(json_array.substr(pos, end_pos-pos+1).c_str(), "%f,%f]", &lng_flt, &lat_flt);
				lat_2 = multiply_const * lat_flt;
				lng_2 = multiply_const * lng_flt;
				if(coord_ind != 0) {
					(*total_length)++;
					(*vector_length)++;
					if(*total_length == 1) {
						*vectors = reinterpret_cast<long *>(malloc(sizeof(**vectors) * 4));
					}
					else {
						*vectors = reinterpret_cast<long *>(realloc(*vectors, sizeof(**vectors)*(*total_length)*4));
					}
					current_pos=(*total_length-1)*4;
					(*vectors)[current_pos] = lng_1;
					(*vectors)[current_pos+1] = lat_1;
					(*vectors)[current_pos+2] = lng_2;
					(*vectors)[current_pos+3] = lat_2;
				}
				lat_1 = lat_2;
				lng_1 = lng_2;
				pos = end_pos+2;
				coord_ind++;
			}
			pos++;
		}
		//[[[[],[]],[[]]],[[[],[]]]]
		pos++;
	}
}

int RetrieveValuesFromPG(string connection_details, string select_request, double multiply_const,
		long **vectors, long *total_length, double **poly_values, long *num_polys, long **vector_lengths) {
	try {
		connection C(connection_details);
		if (C.is_open()) {
			cout << "Opened database successfully: " << C.dbname() << endl;
		} else {
			cout << "Can't open database" << endl;
			return 1;
		}
		/* Create a non-transactional object. */
		nontransaction N(C);
		
		/* Execute SQL query */
		result R( N.exec( select_request ));
		*num_polys = R.size();
		(*poly_values) = reinterpret_cast<double *>(malloc(sizeof(**poly_values)*(*num_polys)));
		(*vector_lengths) = reinterpret_cast<long *>(malloc(sizeof(**vector_lengths)*(*num_polys)));
		*total_length = 0;
		long i = 0;
		for (result::const_iterator c = R.begin(); c != R.end(); ++c, i++) {
			(*poly_values)[i] = c[1].as<double>();
			JsonGeometryToVectors(c[0].as<string>(), multiply_const, vectors, total_length, (*vector_lengths)+i);
		}
		C.disconnect();
	} catch (const std::exception &e) {
		cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}

int SendDataToURL(char *url, void *data, size_t data_size) {
	CURL *curl;
	CURLcode res;
	auto image_curl_read_callback = [&](void *dest_ptr, size_t size, size_t nmemb, void *src_ptr) {
		size_t amount_to_read = size*nmemb;
		if(data_size < amount_to_read)
			amount_to_read = data_size;
	
		memcpy(dest_ptr, src_ptr, amount_to_read);
		return amount_to_read;
	};

	/* In windows, this will init the winsock stuff */ 
	curl_global_init(CURL_GLOBAL_ALL);

	/* get a curl handle */ 
	curl = curl_easy_init();
	if(curl) {
		/* First set the URL that is about to receive our POST. This URL can
			just as well be a https:// URL if that is what should receive the
			data. */ 
		curl_easy_setopt(curl, CURLOPT_URL, url);
		/* Now specify the POST data */ 
		curl_easy_setopt(curl, CURLOPT_PUT, 1L);
		curl_easy_setopt(curl, CURLOPT_READFUNCTION, image_curl_read_callback);

		curl_easy_setopt(curl, CURLOPT_INFILESIZE, data_size);
		
		curl_easy_setopt(curl, CURLOPT_READDATA, data);

		/* Perform the request, res will get the return code */ 
		res = curl_easy_perform(curl);
		/* Check for errors */ 
		if(res != CURLE_OK)
			fprintf(stderr, "curl_easy_perform() failed: %s\n",
				curl_easy_strerror(res));

		/* always cleanup */ 
		curl_easy_cleanup(curl);
	}
	curl_global_cleanup();
	cout << endl;
	return 0;
}

int CheckForQueue(Redis &redis, char *queue_name, string &queue_data) {
	try {
    auto val = redis.lpop(queue_name);
		if(val) {
			queue_data = *val;
			return 1;
		}
		else
			return 0;
	} catch (const Error &e) {
		fprintf(stderr, "%s\n", e.what());
		return -1;
	}
}

void ParseQueueData(string queue_data, long *queue_id, long *start_lat,
		long *start_lng, double *multiply_const, long *image_size,
		double *quality_scale, enum QualityCalcMethod *quality_calc_method, 
		double *quality_calc_value, char *polygons_db_request, char *aws_s3_url) {
	int temp_for_enum;
	sscanf(queue_data.c_str(), "%ld %d %d %lf %d %lf %d %lf %[^\n] %[^\n]",
		queue_id,
		start_lat,
		start_lng,
		multiply_const,
		image_size,
		quality_scale,
		&temp_for_enum,
		quality_calc_value,
		aws_s3_url,
		polygons_db_request		
	);
	*quality_calc_method = (enum QualityCalcMethod) temp_for_enum;
}

/**
* Program main
*/
int main(int argc, char **argv) {
	const char *REDIS_URL, *PG_URL;
	if(!(REDIS_URL = getenv("REDIS_URL"))) {
		fprintf(stderr, "Missing REDIS_URL\n");
		return 1;
	}
	if(!(PG_URL = getenv("PG_URL"))) {
		fprintf(stderr, "Missing PG_URL\n");
		return 1;
	}

	long *vectors;
	double *poly_values;
	long num_polys, start_lat, start_lng, image_size, total_length;
	enum QualityCalcMethod quality_calc_method;
	double quality_scale, quality_calc_value, multiply_const;
	long *vector_lengths;
	long queue_id;
	char polygons_db_request[MAX_POSTGRES_QUERY_SIZE];
	char aws_s3_url[1024];
	char status_key[128];

	Redis redis = Redis(REDIS_URL);
	string queue_data;
	int queue_status;
	while(1)
	{
		queue_status = CheckForQueue(redis, (char *) REDIS_QUEUE_NAME, queue_data);
		if(queue_status == 0) {
			sleep(2);
			continue;
		}
		else if(queue_status == -1) {
			return 1;
		}
		sprintf(status_key,"%s:%d", REDIS_STATUS_DIR_NAME, queue_id);
		redis.set(status_key, "started");
		ParseQueueData(
			queue_data,
			&queue_id,
			&start_lat,
			&start_lng,
			&multiply_const,
			&image_size,
			&quality_scale,
			&quality_calc_method,
			&quality_calc_value,
			polygons_db_request,
			aws_s3_url
		);
		cout << "Retrieving values" << endl;
		RetrieveValuesFromPG(
			PG_URL,
			polygons_db_request,
			multiply_const,
			&vectors,
			&total_length,
			&poly_values,
			&num_polys,
			&vector_lengths
		);
		cout << "Values Retrieved" << endl;

		void *png_pointer = NULL;
		size_t png_size;

		cout << "Calculating..." << endl;

		if(PointInPolygonsImage(&png_pointer, &png_size, start_lat, start_lng, image_size, quality_scale, quality_calc_method, quality_calc_value,
				vectors, total_length, poly_values, num_polys, vector_lengths))
			exit(1);

		cout << "Sending Request" << endl;
		cout << "Png size: " << png_size << endl;
		if(SendDataToURL(aws_s3_url, png_pointer, png_size))
			exit(1);
		redis.set(status_key, "complete");

		free(png_pointer);
		free(vectors);
		free(poly_values);
		free(vector_lengths);
	}

	std::cin.ignore();
	exit(0);
}