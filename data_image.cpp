// System includes
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <fstream>
#include <string>
#include <stdint.h>

using namespace std;
enum QualityCalcMethod { QualityLogExpSum = 0, QualityFirst = 1 };

void PointsInPolygons(int32_t x, int32_t y, int32_t start_lat, int32_t start_lng, int32_t image_size,
	int32_t num_polys, double quality_scale, enum QualityCalcMethod quality_calc_method, double quality_calc_value,
	unsigned char *image_mem, char *found_mem, char *all_blank, int32_t *vectors, double *poly_values, int32_t *vector_lengths) {
	int32_t lat, lng, image_pos;
	lat = start_lat + y;
	lng = start_lng + x;
	image_pos = y * image_size + x;

	double value = 0;
	for (int32_t i = 0, vector_ind = 0, j, intersections; i < num_polys; i++) {
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
		// printf("i: %d, vector_ind: %d\n", i, vector_ind);
	}

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
	image_mem[image_pos] = (fixed_value >> 24) & 0xFF; // red
	image_mem[image_pos + 1] = (fixed_value >> 16) & 0xFF; // green
	image_mem[image_pos + 2] = (fixed_value >> 8) & 0xFF; // blue
	image_mem[image_pos + 3] = fixed_value & 0xFF; // alpha
}
//
/**
* Return an array of char poly_values that is equal to whether this point is in polygon[i]
*/

int PointInPolygonsImage(char **image, size_t *png_size, int32_t start_lat,
	int32_t start_lng, int32_t image_size, double quality_scale, enum QualityCalcMethod quality_calc_method,
	double quality_calc_value, int32_t *vectors, int32_t total_length, double *poly_values, int32_t num_polys, int32_t *vector_lengths) {

	// Allocate device memory
	char *h_found_mem, h_all_blank[1] = { 1 };

	unsigned char *h_image_mem;

	switch (quality_calc_method) {
	case QualityFirst:
		if (!(h_found_mem = reinterpret_cast<char *>(malloc(image_size * image_size)))) {
			fprintf(stderr, "Failed to allocate host found mem!\n");
			exit(EXIT_FAILURE);
		}
		break;
	default:
		break;
	}


	if (!(h_image_mem = reinterpret_cast<unsigned char *>(malloc(image_size * image_size * 4)))) {
		fprintf(stderr, "Failed to allocate host image mem!\n");
		exit(EXIT_FAILURE);
	}

	printf("Starting Calc...\n");

	for(int32_t x = 0, y; x < image_size; x++) {
		printf("x: %ld\n", x);
		fflush(stdout);
		for(y = 0; y < image_size; y++) {
			PointsInPolygons(x, y, start_lat, start_lng, image_size,
				num_polys, quality_scale, quality_calc_method, quality_calc_value,
				h_image_mem, h_found_mem, h_all_blank, vectors, poly_values, vector_lengths);

		}
	}
	printf("done\n");

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

int retrieveValuesFromStream(istream &s, int32_t *start_lat, int32_t *start_lng, int32_t *image_size, double *quality_scale, enum QualityCalcMethod *quality_calc_method,
	double *quality_calc_value, int32_t **vectors, int32_t *total_length, double **poly_values, int32_t *num_polys, int32_t **vector_lengths) {
	double multiply_const;
	s.read((char *)start_lat, sizeof(*start_lat));
	s.read((char *)start_lng, sizeof(*start_lng));
	s.read((char *)&multiply_const, sizeof(multiply_const));
	s.read((char *)image_size, sizeof(*image_size));

	int32_t temp_for_enum;
	s.read((char *)quality_scale, sizeof(*quality_scale));
	s.read((char *)&temp_for_enum, sizeof(temp_for_enum));
	s.read((char *)quality_calc_value, sizeof(*quality_calc_value));
	s.read((char *)num_polys, sizeof(*num_polys));
	s.read((char *)total_length, sizeof(*total_length));

	(*quality_calc_method) = (enum QualityCalcMethod) temp_for_enum;
	(*vectors) = reinterpret_cast<int32_t *>(malloc(sizeof(**vectors)*(*total_length) * 4));
	(*poly_values) = reinterpret_cast<double *>(malloc(sizeof(**poly_values)*(*num_polys)));
	(*vector_lengths) = reinterpret_cast<int32_t *>(malloc(sizeof(**vector_lengths)*(*num_polys)));
	if (!(*vectors && *poly_values && *vector_lengths)) {
		fprintf(stderr, "Failed to allocate data!\n");
		exit(EXIT_FAILURE);
	}
	int32_t current_pos = 0;
	float coord;
	int32_t added_length = 0;
	for (int32_t i = 0, j, k; i < *num_polys; i++) {
		s.read((char *)((*poly_values) + i), sizeof(**poly_values));
		s.read((char *)((*vector_lengths) + i), sizeof(**vector_lengths));
		for (j = 0; j < (*vector_lengths)[i]; j++, current_pos += 4) {
			for (k = 0; k < 4; k++) {
				s.read((char *)&coord, sizeof(coord));
				(*vectors)[current_pos+k] = (int32_t)(coord * multiply_const);
			}
		}
	}
	return EXIT_SUCCESS;
}

/**
* Program main
*/
int main(int argc, char **argv) {
	int num = 1;

	ifstream request("test_request.bin", ios::out | ios::binary);
	if (!request || !request.is_open())
	{
		exit(1);
	}

	int32_t *vectors;
	double *poly_values;
	int32_t num_polys, start_lat, start_lng, image_size, total_length;
	enum QualityCalcMethod quality_calc_method;
	double quality_scale, quality_calc_value;
	int32_t *vector_lengths;
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

	//for (int32_t poly = 0, ind = 0; poly < num_polys; poly++) {
	//	cout << "Polygon: " << poly << endl;
	//	for (int32_t i = 0; i < vector_lengths[poly]; i++, ind += 4) {
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