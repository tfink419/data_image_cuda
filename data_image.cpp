// System includes
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <utility>
#include <cstdlib>

#include <iomanip>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include <sw/redis++/redis++.h>
#include <pqxx/pqxx> 
#include <curl/curl.h>
#include "vips/vips.h"


using namespace std;
using namespace pqxx;
using namespace sw::redis;

#define MAX_POSTGRES_QUERY_SIZE 1024*10
#define REDIS_QUEUE_NAME "data_image_qda:queue"
#define REDIS_STATUS_DIR_NAME "data_image_qda:status"
// #define TEST_DB_CONNECTION "dbname = qol_indicator_development user = tyler hostaddr = 192.168.0.168 port = 5432"

enum QualityCalcMethod { QualityLogExpSum = 0, QualityFirst = 1 };

void PointsInPolygons(int32_t x, int32_t y, int32_t start_lat, int32_t start_lng, int32_t image_size,
		int32_t num_polys, double quality_scale, enum QualityCalcMethod quality_calc_method, double quality_calc_value,
		unsigned char *image_mem, unsigned char *found_mem, unsigned char *all_blank, int32_t *vectors, double *poly_values, int32_t *vector_lengths) {
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


static int MemArrayToPngPointerWithFilter(VipsObject *scope, unsigned char *image_mem, unsigned char* found_mem, long size, void **pngPointer, size_t *image_size) {
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
//
/**
* Return Data Image representing points in polygons
*/
int PointInPolygonsImage(void **png_pointer, size_t *png_size, int32_t start_lat,
		int32_t start_lng, int32_t image_size, double quality_scale, enum QualityCalcMethod quality_calc_method,
		double quality_calc_value, int32_t *vectors, int32_t total_length, double *poly_values, int32_t num_polys, int32_t *vector_lengths) {

	// Allocate device memory
	unsigned char *h_found_mem, h_all_blank[1] = { 1 }, *h_image_mem;

	switch (quality_calc_method) {
	case QualityFirst:
		if (!(h_found_mem = reinterpret_cast<unsigned char *>(malloc(image_size * image_size)))) {
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

	for(int32_t x = 0, y; x < image_size; x++) {
		printf("x: %ld\n", x);
		fflush(stdout);
		for(y = 0; y < image_size; y++) {
			h_image_mem[((y*image_size)+x)*4] = h_image_mem[((y*image_size)+x)*4+1] = h_image_mem[((y*image_size)+x)*4+2] = h_image_mem[((y*image_size)+x)*4+3] = 0;
			// PointsInPolygons(x, y, start_lat, start_lng, image_size,
			// 	num_polys, quality_scale, quality_calc_method, quality_calc_value,
			// 	h_image_mem, h_found_mem, h_all_blank, vectors, poly_values, vector_lengths);

		}
	}
	printf("done\n");

	long *value_at;
	// for (long y = 0, x; y < image_size; y++) {
	// 	printf("y: %ld\n", y);
	// 	for (x = 0; x < image_size; x++) {
	// 		value_at = (long *)(h_image_mem + (y*image_size + x) * 4);
	// 		printf("x: %ld, = %lu\n", x, *value_at);
	// 	}
	// }

	
	VipsObject *scope;
	scope = VIPS_OBJECT( vips_image_new() );
	if(MemArrayToPngPointerWithFilter(scope, h_image_mem, h_found_mem, image_size, png_pointer, png_size))
		vips_error_exit( NULL );

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

void JsonGeometryToVectors(string json_array, double multiply_const, int32_t **vectors, int32_t *total_length, int32_t *vector_length) {
	size_t pos = 0, coord_ind, end_pos, current_pos;
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
		//[[[[],[]],[[]]],[[[],[]]]]
		pos++;
	}
}

int RetrieveValuesFromPG(string connection_details, string select_request, double multiply_const,
		int32_t **vectors, int32_t *total_length, double **poly_values, int32_t *num_polys, int32_t **vector_lengths) {
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
		(*vector_lengths) = reinterpret_cast<int32_t *>(malloc(sizeof(**vector_lengths)*(*num_polys)));
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

size_t png_size_for_callback;
static size_t image_curl_read_callback(void *dest_ptr, size_t size, size_t nmemb, void *src_ptr)
{
	size_t amount_to_read = size*nmemb;
	if(png_size_for_callback < amount_to_read)
		amount_to_read = png_size_for_callback;

	memcpy(dest_ptr, src_ptr, amount_to_read);
  return amount_to_read;
}

int SendDataToURL(char *url, void *data, size_t data_size) {
	CURL *curl;
	CURLcode res;

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
		png_size_for_callback = data_size;
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

void ParseQueueData(string queue_data, long *queue_id, int32_t *start_lat,
		int32_t *start_lng, double *multiply_const, int32_t *image_size,
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
  Listen to port given and if the special key is given
*/
int main(int argc, char **argv) {
	const char *PORT, *REDIS_URL, *PG_URL;
	if(!(PORT = getenv("PORT"))) {
		fprintf(stderr, "Missing PORT\n");
		return 1;
	}
	if(!(REDIS_URL = getenv("REDIS_URL"))) {
		fprintf(stderr, "Missing REDIS_URL\n");
		return 1;
	}
	if(!(PG_URL = getenv("PG_URL"))) {
		fprintf(stderr, "Missing PG_URL\n");
		return 1;
	}

	int32_t *vectors;
	double *poly_values;
	int32_t num_polys, start_lat, start_lng, image_size, total_length;
	enum QualityCalcMethod quality_calc_method;
	double quality_scale, quality_calc_value, multiply_const;
	int32_t *vector_lengths;
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
		sprintf(status_key,"%s:%d", REDIS_STATUS_DIR_NAME, queue_id)
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
		
		// for (int32_t poly = 0, ind = 0; poly < num_polys; poly++) {
		// 	cout << "Polygon: " << poly << endl;
		// 	cout << "  Vector Length: " << vector_lengths[poly] << endl;
		// 	for (int32_t i = 0; i < vector_lengths[poly]; i++, ind += 4) {
		// 		printf("Vector: [%d, %d, %d, %d]\n", vectors[ind], vectors[ind + 1], vectors[ind + 2], vectors[ind + 3]);
		// 	}
		// }

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