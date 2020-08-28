// C standard libraries
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>

// Unix library
#include <unistd.h>
#include <pthread.h>

// Imported Libraries
#include "vips/vips.h"
#include <curl/curl.h>
#include <sw/redis++/redis++.h>
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
#define REDIS_WORKING_NAME "data_image_qda:working"
#define REDIS_COMPLETE_BASE_NAME "data_image_qda:complete"
#define REDIS_QUEUE_DETAILS_BASE_NAME "data_image_qda:queue_details"
#define REDIS_DETAILS_BASE_NAME "data_image_qda:details"
#define PNG_POOL_SIZE 32
#define PNG_UPLOAD_WAIT_TIME 30000

enum QualityCalcMethod { QualityLogExpSum = 0, QualityFirst = 1 };

/**
* Point-In-Polygons for a block using CUDA
*/
__global__ void PointsInPolygonsCUDA(int32_t start_lat, int32_t start_lng, int32_t image_size,
	int32_t num_polys, double quality_scale, enum QualityCalcMethod quality_calc_method, double quality_calc_value,
	uint8_t *image_mem, uint8_t *found_mem, uint8_t *all_blank, int32_t *vectors, double *poly_values, int32_t *vector_lengths) {

	int32_t x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= image_size || y >= image_size) return;
	int32_t lat, lng, image_pos;
	lat = start_lat + (image_size-y-1);
	lng = start_lng + x;
	image_pos = y * image_size + x;

	double value = 0;
	for (int32_t i = 0, vector_ind = 0, j, intersections; i < num_polys; i++) {
		// Point in Polygon calculation
		for (intersections = 0, j = 0; j < vector_lengths[i]; j++, vector_ind += 4) {
			if (((vectors[vector_ind + 1]>lat) != (vectors[vector_ind + 3]>lat)) &&
				(lng < ((int64_t)vectors[vector_ind + 2] - vectors[vector_ind]) * ((int64_t)lat - vectors[vector_ind + 1]) / ((int64_t)vectors[vector_ind + 3] - vectors[vector_ind + 1]) + vectors[vector_ind])) {
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
		*all_blank = 0;
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
	uint32_t fixed_value;
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

static int MemArrayToPngPointerWithFilter(VipsObject *scope, uint8_t *image_mem, uint8_t* found_mem, int32_t size, void **pngPointer, size_t *image_size) {
	VipsImage **ims = (VipsImage **) vips_object_local_array( scope, 7 );
  if(!(ims[0] = vips_image_new_from_memory( image_mem, 4 * size * size, size, size, 4, VIPS_FORMAT_UCHAR)))
    return -1;
  if(vips_copy(ims[0], ims+1, "bands", 1, "format", VIPS_FORMAT_UINT, NULL))
    return -1;
  if(found_mem) {
		// Apply median rank filtering on holes
    if(!(ims[2] = vips_image_new_from_memory( found_mem, size * size, size, size, 1, VIPS_FORMAT_UCHAR )) ||
      vips_median( ims[1], ims+3, 3, NULL ) ||
      vips_equal_const1( ims[2], ims+4, 1, NULL ) ||
      vips_ifthenelse( ims[4], ims[1], ims[3], ims+4, NULL ) )
        return -1;
  }
  else {
    ims[4] = ims[1];
    ims[1] = NULL;
  }
	if(vips_copy(ims[4], ims+5, "bands", 4, "format", VIPS_FORMAT_UCHAR, NULL))
    return -1;

  if( vips_pngsave_buffer(ims[5], pngPointer, image_size, "compression", 9, NULL) )
    return -1;

  return 0;
}

int PointInPolygonsImage(void **png_pointer, size_t *png_size, int32_t start_lat,
	int32_t start_lng, int32_t image_size, double quality_scale, enum QualityCalcMethod quality_calc_method,
	double quality_calc_value, int32_t *vectors, int32_t total_length, double *poly_values, int32_t num_polys, int32_t *vector_lengths) {
	cudaStream_t stream;

	// Allocate device memory
	int32_t vectors_mem_size = total_length * 4 * sizeof(*vectors);
	int32_t vector_lengths_mem_size = num_polys * sizeof(*vector_lengths);
	int32_t values_mem_size = num_polys * sizeof(*poly_values);
	int32_t found_mem_size = image_size * image_size * sizeof(uint8_t);
	int32_t image_mem_size = found_mem_size * 4;
	int32_t *d_vectors;
	int32_t *d_vector_lengths;
	uint8_t *d_image_mem, *h_image_mem, *d_found_mem,
			*h_found_mem = NULL, *d_all_blank, h_all_blank[1] = { 1 };
	double *d_poly_values;
	const char *CUDA_DEVICE_ENV = getenv("CUDA_DEVICE_ENV");

	int32_t max_vector_lengths = 0;
	for (int32_t i = 0; i < num_polys; i++)
		if (vector_lengths[i] > max_vector_lengths)
			max_vector_lengths = vector_lengths[i];


	switch (quality_calc_method) {
	case QualityFirst:
		if (!(h_found_mem = reinterpret_cast<uint8_t *>(malloc(found_mem_size)))) {
			cerr << "Failed to allocate host found mem!" << endl;
			exit(EXIT_FAILURE);
		}
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_found_mem), found_mem_size));
		break;
	default:
		break;
	}


	if (!(h_image_mem = reinterpret_cast<uint8_t *>(malloc(image_mem_size)))) {
		cerr << "Failed to allocate host image mem!" << endl;
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vectors), vectors_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_vector_lengths), vector_lengths_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_poly_values), values_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_image_mem), image_mem_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_all_blank), sizeof(uint8_t)));
	// Allocate CUDA events that we'll use for timing
	// cudaEvent_t start, stop;
	// checkCudaErrors(cudaEventCreate(&start));
	// checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// copy host memory to device
	checkCudaErrors(cudaMemcpyAsync(d_vectors, vectors, vectors_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_poly_values, poly_values, values_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_vector_lengths, vector_lengths, vector_lengths_mem_size, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_all_blank, h_all_blank, sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

	// Setup execution parameters
	dim3 threads(28, 28);
	if(CUDA_DEVICE_ENV && strcmp(CUDA_DEVICE_ENV, "production")) {
		// 2560 cores O_o
		threads.x = 64;
		threads.y = 40;
	}
	dim3 grid((image_size + threads.x - 1) / threads.x, (image_size + threads.y - 1) / threads.y);

	// Create and start timer
	// printf("Computing result using CUDA Kernel...\n");

	// Record the start event
	// checkCudaErrors(cudaEventRecord(start, stream));
	// printf("Sending grid: [%d,%d]\n", grid.x, grid.y);
	// printf("Processing %d vectors %d times\n", total_length, image_size*image_size);
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
	// checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	// checkCudaErrors(cudaEventSynchronize(stop));

	// float msecTotal = 0.0f;
	// checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	// printf(
	// 	"Time= %.3f msec, GInt64OPS=%.3f\n",
	// 	msecTotal, (float)total_length*image_size*image_size/(msecTotal/1000)/1024/1024/1024);

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
	// checkCudaErrors(cudaEventDestroy(start));
	// checkCudaErrors(cudaEventDestroy(stop));

	if(*h_all_blank) {
		*png_pointer = NULL;
		free(h_image_mem);
		switch (quality_calc_method) {
		case QualityFirst:
			free(h_found_mem);
			break;
		default:
			break;
		}
		return EXIT_SUCCESS;
	}

	VipsObject *scope;
	scope = VIPS_OBJECT( vips_image_new() );
	if(MemArrayToPngPointerWithFilter(scope, h_image_mem, h_found_mem, image_size, png_pointer, png_size))
		vips_error_exit( NULL );
  g_object_unref( scope );

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

void JsonGeometryToVectors(string json_array, double multiply_const, int32_t **vectors, int32_t *total_length, int32_t *vector_length) {
	size_t pos = 1, coord_ind, end_pos, current_pos;
	*vector_length = 0;
	float lat_flt, lng_flt;
	int32_t lat_1, lng_1, lat_2, lng_2;
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
						*vectors = reinterpret_cast<int32_t *>(malloc(sizeof(**vectors) * 4));
					}
					else {
						*vectors = reinterpret_cast<int32_t *>(realloc(*vectors, sizeof(**vectors)*(*total_length)*4));
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
		//Example (Pretend no whitespace)
		//[  Entire Geometry
		//	[ A Polygon
		//		[ A polygon or hole
		//			[-104.862892,39.745213],[-104.86039,39.742725],[-104.859138,39.742374],[-104.857269,39.740959],[-104.856567,39.739964],[-104.857697,39.73896],[-104.857925,39.737965],[-104.86097,39.735962],[-104.863129,39.733803],[-104.864136,39.734241],[-104.865135,39.73394],[-104.867683,39.735962],[-104.86821,39.736885],[-104.870064,39.737965],[-104.870186,39.738903],[-104.871124,39.739964],[-104.866142,39.744408],[-104.863136,39.745396],[-104.862892,39.745213]
		//		]
		//	]
		//]
		pos++;
	}
}

int RetrieveValuesFromPG(connection *C, string select_request, double multiply_const,
		int32_t **vectors, int32_t *total_length, double **poly_values, int32_t *num_polys, int32_t **vector_lengths) {
	try {
		/* Create a non-transactional object. */
		nontransaction N(*C);
		
		/* Execute SQL query */
		result R( N.exec( select_request ));
		*num_polys = R.size();
		(*poly_values) = reinterpret_cast<double *>(malloc(sizeof(**poly_values)*(*num_polys)));
		(*vector_lengths) = reinterpret_cast<int32_t *>(malloc(sizeof(**vector_lengths)*(*num_polys)));
		*total_length = 0;
		int32_t i = 0;
		for (result::const_iterator c = R.begin(); c != R.end(); ++c, i++) {
			(*poly_values)[i] = c[1].as<double>();
			JsonGeometryToVectors(c[0].as<string>(), multiply_const, vectors, total_length, (*vector_lengths)+i);
		}
	} catch (const exception &e) {
		cerr << e.what() << endl;
		return 1;
	}
	return 0;
}

struct WriteThis {
  char *readptr;
  size_t sizeleft;
};
 
static size_t image_curl_read_callback(void *dest, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *wt = (struct WriteThis *)userp;
  size_t buffer_size = size*nmemb;
 
  if(wt->sizeleft) {
    /* copy as much as possible from the source to the destination */ 
    size_t copy_this_much = wt->sizeleft;
    if(copy_this_much > buffer_size)
      copy_this_much = buffer_size;
    memcpy(dest, wt->readptr, copy_this_much);
 
    wt->readptr += copy_this_much;
    wt->sizeleft -= copy_this_much;
    return copy_this_much; /* we copied this many bytes */ 
  }
 
  return 0; /* no more data left to deliver */ 
}

struct CurlThreadInfo {
	char url[1024];
	void *data;
	pthread_t thread_id;
	size_t thread_num;
	size_t data_size;
};

void * SendDataToURL(void *args) {
	struct CurlThreadInfo *curl_info = (struct CurlThreadInfo *) args;
 	/* get a curl handle */ 
	CURL *curl = curl_easy_init();
  CURLcode res;
	if(curl) {
		/* First set the URL that is about to receive our POST. This URL can
			just as well be a https:// URL if that is what should receive the
			data. */ 
 
    /* set our custom set of headers */ 
		struct curl_slist *curl_header = NULL;
		curl_header = curl_slist_append(curl_header, "Content-Type: image/png");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_header);
		
		curl_easy_setopt(curl, CURLOPT_URL, curl_info->url);
		/* Now specify the PUT data */ 
		curl_easy_setopt(curl, CURLOPT_PUT, 1L);
		
		curl_easy_setopt(curl, CURLOPT_READFUNCTION, image_curl_read_callback);

		curl_easy_setopt(curl, CURLOPT_INFILESIZE, curl_info->data_size);
	
		struct WriteThis wt;
 
		wt.readptr = (char *)curl_info->data;
		wt.sizeleft = curl_info->data_size;
		curl_easy_setopt(curl, CURLOPT_READDATA, &wt);

		/* Perform the request, res will get the return code */ 
		res = curl_easy_perform(curl);
		if(res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
							curl_easy_strerror(res));
			return (void *)1;
		}
 
		curl_slist_free_all(curl_header);
    /* always cleanup */ 
    curl_easy_cleanup(curl);
	}
	else {
		cerr << "Curl Failed to Initialize" << endl;
		return (void *)1;
	}
	return NULL;
}

int CheckForQueue(Redis &redis, char *queue_name, char *working_name,
		char *queue_details_key, int32_t *queue_id, int32_t *start_lat,
		int32_t *start_lng, double *multiply_const, int32_t *image_size,
		double *quality_scale, enum QualityCalcMethod *quality_calc_method, 
		double *quality_calc_value, char *polygons_db_request, char *aws_s3_url) {
	try {
    auto id = redis.brpoplpush(queue_name, working_name, 30);
		if(id) {
			unordered_map<string, string> m;
			sscanf((*id).c_str(), "%d", queue_id);
			sprintf(queue_details_key, "%s:%d", REDIS_QUEUE_DETAILS_BASE_NAME, *queue_id);
			auto queue_details = redis.get(queue_details_key);
			int temp_for_enum;
			if(!queue_details) {
				redis.lrem(working_name, 1, (*id).c_str());
				return 0;
			}
			sscanf((*queue_details).c_str(), "%d %d %lf %d %lf %d %lf %[^\n] %[^\n]",
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
			return 1;
		}
		else
			return 0;
	} catch (const Error &e) {
		cerr << e.what() << endl;
		return -1;
	}
}

int has_sig_inted = 0;

static void hdl (int sig, siginfo_t *siginfo, void *context)
{
	has_sig_inted = 1;
}

/**
* Program main
*/
int main(int argc, char **argv) {
	struct sigaction act;
 
	memset (&act, '\0', sizeof(act));
 
	/* Use the sa_sigaction field because the handles has two additional parameters */
	act.sa_sigaction = &hdl;
 
	/* The SA_SIGINFO flag tells sigaction() to use the sa_sigaction field, not sa_handler. */
	act.sa_flags = SA_SIGINFO;

	if (sigaction(SIGINT, &act, NULL) < 0) {
		perror ("sigaction");
		return 1;
	}

	if (sigaction(SIGTERM, &act, NULL) < 0) {
		perror ("sigaction");
		return 1;
	}
 
	const char *REDIS_URL, *PG_URL;
	// Check ENV Vars
	if(!(REDIS_URL = getenv("REDIS_URL")) || !REDIS_URL[0]) {
		cerr << "Missing REDIS_URL" << endl;
		fprintf(stderr, "Missing REDIS_URL\n");
		exit(1);
	}
	if(!(PG_URL = getenv("PG_URL")) || !PG_URL[0]) {
		cerr << "Missing PG_URL" << endl;
		exit(1);
	}

	// Parse Redis URL
	char REDIS_PASSWORD[128] = "", REDIS_HOST[128], REDIS_PORT[8] = "";
	const char *loc_of_at, *loc_of_colon;
	if(loc_of_at = strchr(REDIS_URL,'@')) {
		loc_of_colon = strchr(REDIS_URL+8,':'); // : after "redis://"
		if(loc_of_colon && loc_of_colon < loc_of_at) {
			sscanf(REDIS_URL,"redis://%*[^:]:%[^@]@%[^:]:%s", REDIS_PASSWORD, REDIS_HOST, REDIS_PORT);
		}
		else {
			sscanf(REDIS_URL,"redis://%[^@]@%[^:]:%s", REDIS_PASSWORD, REDIS_HOST, REDIS_PORT);
		}
	}
	else {
		sscanf(REDIS_URL,"redis://%[^:]:%s", REDIS_HOST, REDIS_PORT);
	}
	int32_t *vectors = NULL;
	double *poly_values = NULL;
	int32_t num_polys, start_lat, start_lng, image_size, total_length;
	enum QualityCalcMethod quality_calc_method;
	double quality_scale, quality_calc_value, multiply_const;
	int32_t *vector_lengths = NULL;
	int32_t queue_id;
	char polygons_db_request[MAX_POSTGRES_QUERY_SIZE];
	char complete_key[64];
	char queue_details_key[64];
	char id_str[32];
	int thread_status;
	size_t i;

	/* In windows, this will init the winsock stuff */ 
	curl_global_init(CURL_GLOBAL_ALL);
	
	// DB connections setup
	connection *postgres_connection;
	ConnectionOptions connection_options; // Redis connection
	connection_options.host = REDIS_HOST;
	if(REDIS_PORT[0])
		connection_options.port = atoi(REDIS_PORT);
	if(REDIS_PASSWORD[0])
		connection_options.password = REDIS_PASSWORD;
	connection_options.keep_alive = true;
	Redis *redis;

	try {
		redis = new Redis(connection_options);
		postgres_connection = new connection(PG_URL);
		if (postgres_connection->is_open()) {
			cout << "Opened PG database successfully: " << postgres_connection->dbname() << endl;
		} else {
			cerr << "Can't open database" << endl;
			exit(1);
		}
	} catch (const Error &e) {
		cerr << e.what() << endl;
		exit(1);
	}
	auto pipe = redis->pipeline();
	string queue_data;
	int queue_status;
	struct CurlThreadInfo curl_threads[PNG_POOL_SIZE] = {0};
	for(i = 1; i <= PNG_POOL_SIZE; i++)
		curl_threads[i].thread_num = i;

	void *thread_res;
	int current_png = 0;
	while(1)
	{
		if(has_sig_inted) {
			cout << "Received SIGINT or TERM and gracefully quiting" << endl;
			break;
		}
		// cout << "Waiting for Queue\n";
		queue_status = CheckForQueue(
			*redis, 
			(char *) REDIS_QUEUE_NAME,
			(char *) REDIS_WORKING_NAME,
			queue_details_key,
			&queue_id,
			&start_lat,
			&start_lng,
			&multiply_const,
			&image_size,
			&quality_scale,
			&quality_calc_method,
			&quality_calc_value,
			polygons_db_request,
			curl_threads[current_png].url
		);
		if(queue_status == 0) { // Redis Blocking Timeout
			cout << "Clearing CURL Queue" << endl;

			for (i = 0; i < current_png; i++) {
				if(thread_status = pthread_join(curl_threads[i].thread_id, &thread_res)) {
					cerr << "Error joining thread: " << thread_status << endl;
					exit(1);
				}
				free(curl_threads[i].data);
				curl_threads[i].data = NULL;
			}
			current_png = 0;
			continue;
		}
		else if(queue_status == -1) {
			exit(1);
		}

		RetrieveValuesFromPG(
			postgres_connection,
			polygons_db_request,
			multiply_const,
			&vectors,
			&total_length,
			&poly_values,
			&num_polys,
			&vector_lengths
		);

		cout << "Calculating...\n";

		if(PointInPolygonsImage(&curl_threads[current_png].data, &curl_threads[current_png].data_size, start_lat, start_lng, image_size, quality_scale, quality_calc_method, quality_calc_value,
				vectors, total_length, poly_values, num_polys, vector_lengths))
			exit(1);
		if(vectors) {
			free(vectors);
			vectors = NULL;
		}
		if(poly_values) {
			free(poly_values);
			poly_values = NULL;
		}
		if(vector_lengths) {
			free(vector_lengths);
			vector_lengths = NULL;
		}
		if(curl_threads[current_png].data) {
			printf("Sending png image of size: %.2f KB\n", curl_threads[current_png].data_size/1024.0);
			if (thread_status = pthread_create(&curl_threads[current_png].thread_id, NULL,
				&SendDataToURL, &curl_threads[current_png])) {
				cerr << "Error creating thread: " << thread_status << endl;
				exit(1);
			}
			if(++current_png >= PNG_POOL_SIZE) {
				current_png = 0;
				cout << "Clearing CURL Queue" << endl;

				for (i = 0; i < PNG_POOL_SIZE; i++) {
					if(thread_status = pthread_join(curl_threads[i].thread_id, &thread_res)) {
						cerr << "Error joining thread: " << thread_status << endl;
						exit(1);
					}
					free(curl_threads[i].data);
					curl_threads[i].data = NULL;
				}
			}
		}
		else {
			cout << "No image to send\n";
		}
		sprintf(complete_key, "%s:%d", REDIS_COMPLETE_BASE_NAME, queue_id);
		sprintf(id_str, "%d", queue_id);
		try {
			pipe.lpush(complete_key, "success").
			expire(complete_key, 300).
			lrem(REDIS_WORKING_NAME, 1, id_str).
			del(queue_details_key).
			exec();
		} catch (const Error &e) {
			cerr << e.what() << endl;
			exit(1);
		}
		
	}
	cout << "Clearing CURL Queue" << endl;
	for (i = 0; i < current_png; i++) {
		if(thread_status = pthread_join(curl_threads[i].thread_id, &thread_res)) {
			cerr << "Error joining thread: " << thread_status << endl;
			exit(1);
		}
		free(curl_threads[i].data);
	}
	try {
		postgres_connection->disconnect();
	} catch (const Error &e) {
		cerr << e.what() << endl;
		exit(1);
	}
	delete postgres_connection;
	delete redis;
	curl_global_cleanup();
	exit(0);
}