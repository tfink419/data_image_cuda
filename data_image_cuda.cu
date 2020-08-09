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

/**
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication which makes use of shared memory
* to ensure data reuse, the matrix multiplication is done using tiling approach.
* It has been written for clarity of exposition to illustrate various CUDA programming
* principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// System includes
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <fstream>
#include <string>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;
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

int PointInPolygonsImage(char **image, size_t *png_size, long start_lat,
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

int retrieveValuesFromStream(istream &s, long *start_lat, long *start_lng, long *image_size, double *quality_scale, enum QualityCalcMethod *quality_calc_method,
	double *quality_calc_value, long **vectors, long *total_length, double **poly_values, long *num_polys, long **vector_lengths) {
	double multiply_const;
	s.read((char *)start_lat, sizeof(*start_lat));
	s.read((char *)start_lng, sizeof(*start_lng));
	s.read((char *)&multiply_const, sizeof(multiply_const));
	s.read((char *)image_size, sizeof(*image_size));

	long temp_for_enum;
	s.read((char *)quality_scale, sizeof(*quality_scale));
	s.read((char *)&temp_for_enum, sizeof(temp_for_enum));
	s.read((char *)quality_calc_value, sizeof(*quality_calc_value));
	s.read((char *)num_polys, sizeof(*num_polys));
	s.read((char *)total_length, sizeof(*total_length));

	(*quality_calc_method) = (enum QualityCalcMethod) temp_for_enum;
	(*vectors) = reinterpret_cast<long *>(malloc(sizeof(**vectors)*(*total_length) * 4));
	(*poly_values) = reinterpret_cast<double *>(malloc(sizeof(**poly_values)*(*num_polys)));
	(*vector_lengths) = reinterpret_cast<long *>(malloc(sizeof(**vector_lengths)*(*num_polys)));
	if (!(*vectors || *poly_values || *vector_lengths)) {
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}
	long current_pos = 0;
	float coord;
	for (long i = 0, j, k; i < *num_polys; i++) {
		s.read((char *)((*poly_values) + i), sizeof(**poly_values));
		s.read((char *)((*vector_lengths) + i), sizeof(**vector_lengths));
		for (j = 0; j < (*vector_lengths)[i]; j++, current_pos += 4) {
			for (k = 0; k < 4; k++) {
				s.read((char *)&coord, sizeof(coord));
				(*vectors)[current_pos+k] = (long)(coord * multiply_const);
			}
		}
	}
	return EXIT_SUCCESS;
}

/**
* Program main
*/
int main(int argc, char **argv) {
	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
		checkCmdLineFlag(argc, (const char **)argv, "?")) {
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		exit(EXIT_SUCCESS);
	}

	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line


	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("Max grid size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		printf("maxSharedMemory: %d\n\n", prop.sharedMemPerBlock);
	}


	int dev = findCudaDevice(argc, (const char **)argv);
	ifstream request("test_request.bin", ios::out | ios::binary);
	if (!request || !request.is_open())
	{
		exit(1);
	}

	long *vectors;
	double *poly_values;
	long num_polys, start_lat, start_lng, image_size, total_length;
	enum QualityCalcMethod quality_calc_method;
	double quality_scale, quality_calc_value;
	long *vector_lengths;
	cout << "Retrieving values" << endl;
	retrieveValuesFromStream(
		request,
		&start_lat,
		&start_lng,
		&image_size,
		&quality_scale,
		&quality_calc_method,
		&quality_calc_value,
		&vectors,
		&total_length,
		&poly_values,
		&num_polys,
		&vector_lengths
	);

	//for (long poly = 0, ind = 0; poly < num_polys; poly++) {
	//	cout << "Polygon: " << poly << endl;
	//	for (long i = 0; i < vector_lengths[poly]; i++, ind += 4) {
	//		printf("Vector: [%d, %d, %d, %d]\n", vectors[ind], vectors[ind + 1], vectors[ind + 2], vectors[ind + 3]);
	//	}
	//}
	request.close();
	cout << "Values Retrieved" << endl;

	char *image = NULL;
	size_t png_size;

	cout << "Calculating..." << endl;

	PointInPolygonsImage(&image, &png_size, start_lat, start_lng, image_size, quality_scale, quality_calc_method, quality_calc_value,
		vectors, total_length, poly_values, num_polys, vector_lengths);

	if (image) {
		free(image);
	}
	free(vectors);
	free(poly_values);
	free(vector_lengths);

	std::cin.ignore();
	exit(0);
}