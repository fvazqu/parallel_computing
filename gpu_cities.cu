#include <iostream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel to calculate distance using the Haversine formula
__global__
void haversine_kernel(double* latitudes, double* longitudes, double start_lat, double start_lon, double* distances, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        double r = 3958.8;  // Radius of Earth in miles

        double lat1_r = start_lat * M_PI / 180.0;
        double long1_r = start_lon * M_PI / 180.0;
        double lat2_r = latitudes[tid] * M_PI / 180.0;
        double long2_r = longitudes[tid] * M_PI / 180.0;

        double a = pow(sin((lat2_r - lat1_r) / 2), 2) + cos(lat1_r) * cos(lat2_r) * pow(sin((long2_r - long1_r) / 2), 2);
        double c = 2 * atan2(sqrt(a), sqrt(1 - a));
        distances[tid] = r * c;
    }
}

// Define haversine function for distance calculation
__device__ double haversine(double lat1, double lon1, double lat2, double lon2) {
    double r = 3958.8;  // Radius of Earth in miles

    double lat1_r = lat1 * M_PI / 180.0;
    double lon1_r = lon1 * M_PI / 180.0;
    double lat2_r = lat2 * M_PI / 180.0;
    double lon2_r = lon2 * M_PI / 180.0;

    double dlat = lat2_r - lat1_r;
    double dlon = lon2_r - lon1_r;

    double a = pow(sin(dlat / 2.0), 2) + cos(lat1_r) * cos(lat2_r) * pow(sin(dlon / 2.0), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return r * c;
}

// CUDA kernel to calculate maximum distance from one city to another for each city
__global__
void calculate_max_distances_kernel(double* latitudes, double* longitudes, double* max_distances, int* max_indexes, int num_cities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_cities) {
        double max_distance = 0.0;
        int max_index = -1;
        for (int j = 0; j < num_cities; ++j) {
            if (i != j) {
                double distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j]);
                if (distance > max_distance) {
                    max_distance = distance;
                    max_index = j;
                }
            }
        }
        max_distances[i] = max_distance;
        max_indexes[i] = max_index;
    }
}

// Define the City Struct
struct City {
    std::string name;
    double latitude;
    double longitude;
    int max_index;
};

// Initiazlize the cities vector
std::vector<City> cities;


int main(int argc, char* argv[]) {
    // Set up configurations
    int start_row = 1;
    int end_row = 47868;
    int columns[] = { 0, 2, 3 };
    double start_lat = 26.3017;
    double start_lon = -98.1633;

    // Read CSV file and populate arrays
    std::string filePath = "worldcities.csv";
    std::ifstream file(filePath);
    std::vector<std::string> names;
    std::vector<double> latitudes;
    std::vector<double> longitudes;


    // Part 1
    if (!file.is_open()) {
        std::cerr << "Failed to open file at " << filePath << std::endl;
        return -1;
    }

    if (file.is_open()) {
        std::string line;
        int current_row = 0;
        while (std::getline(file, line)) {
            if (current_row >= start_row && current_row <= end_row) {
                std::istringstream iss(line);
                std::string token;
                City city;
                int col_index = 0;
                while (std::getline(iss, token, ',') && col_index <= 3) {
                    if (col_index == columns[0]) {
                        city.name = token;
                        names.push_back(token);
                    }
                    else if (col_index == columns[1]) {
                        city.latitude = std::stod(token);
                        latitudes.push_back(std::stod(token));
                    }
                    else if (col_index == columns[2]) {
                        city.longitude = std::stod(token);
                        longitudes.push_back(std::stod(token));
                    }
                    col_index++;
                }
                cities.push_back(city);
                if (current_row == end_row) {
                    break;
                }
            }
            current_row++;
        }
        file.close();
    }

    // Allocate memory on the GPU
    double* latitudes_gpu;
    double* longitudes_gpu;
    double* distances_gpu;
    cudaMalloc(&latitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&longitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&distances_gpu, end_row * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(latitudes_gpu, latitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(longitudes_gpu, longitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (end_row + blockSize - 1) / blockSize;

    // Launch the kernel
    haversine_kernel << <numBlocks, blockSize >> > (latitudes_gpu, longitudes_gpu, start_lat, start_lon, distances_gpu, end_row);

    // Copy results from device to host
    std::vector<double> distances(end_row);
    cudaMemcpy(distances.data(), distances_gpu, end_row * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(latitudes_gpu);
    cudaFree(longitudes_gpu);
    cudaFree(distances_gpu);


    // Part 2
    // Count cities within 1000 miles
    int count = 0;
    for (int i = 0; i < end_row; i++) {
        if (distances[i] <= 1000 && distances[i] > 0) {
            count += 1;
        }
    }

    std::cout << "Total count of cities within 1000 miles of Edinburg, TX: " << count << std::endl;


    // Part 3
    // Find Cairo Egypt Coordinates
    double start_lat2 = 30.0444;
    double start_lon2 = 31.2358;

    // Allocate memory on the GPU
    double* distances2_gpu;
    cudaMalloc(&latitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&longitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&distances2_gpu, end_row * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(latitudes_gpu, latitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(longitudes_gpu, longitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    haversine_kernel << <numBlocks, blockSize >> > (latitudes_gpu, longitudes_gpu, start_lat2, start_lon2, distances2_gpu, end_row);

    // Copy results from device to host
    std::vector<double> distances2(end_row);
    cudaMemcpy(distances2.data(), distances2_gpu, end_row * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(distances2_gpu);

    // Find the closest city to Cairo
    double min_distance = 100000;
    int closest_index = 0;
    for (int i = 0; i < end_row; i++) {
        if (distances2[i] < min_distance && distances2[i] > 0) {
            min_distance = distances2[i];
            closest_index = i;
        }
    }

    // Print the closest city to Cairo
    std::cout << "The closest city to Cairo, Egypt is " << cities[closest_index].name << " at " << min_distance << " miles" << std::endl;



    // Part 4
    // Find Folsom Coordinates
    double start_lat3 = 38.6668;
    double start_lon3 = -121.142;

    // Allocate memory on the GPU
    double* distances3_gpu;
    cudaMalloc(&distances3_gpu, end_row * sizeof(double));

    // Launch the kernel
    haversine_kernel << <numBlocks, blockSize >> > (latitudes_gpu, longitudes_gpu, start_lat3, start_lon3, distances3_gpu, end_row);

    // Copy results from device to host
    std::vector<double> distances3(end_row);
    cudaMemcpy(distances3.data(), distances3_gpu, end_row * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    //cudaFree(latitudes_gpu);
    //cudaFree(longitudes_gpu);
    cudaFree(distances3_gpu);

    // Find the farthest city to Folsom
    double far_distance = 0;
    int far_index = 0;
    for (int i = 0; i < end_row; i++) {
        if (distances3[i] > far_distance) {
            far_distance = distances3[i];
            far_index = i;
        }
    }

    // Print the closest city to Cairo
    std::cout << "The farthest city to Folsom, United States is " << cities[far_index].name << " at " << far_distance << " miles" << std::endl;


    // Part 5

    // Allocate memory on the GPU
    double* max_distances_gpu;
    int* max_indexes_gpu;
    cudaMalloc(&max_distances_gpu, end_row * sizeof(double));
    cudaMalloc(&max_indexes_gpu, end_row * sizeof(int));

    // Launch the kernel
    calculate_max_distances_kernel << <numBlocks, blockSize >> > (latitudes_gpu, longitudes_gpu, max_distances_gpu, max_indexes_gpu, end_row);

    // Copy results from device to host
    double* max_distances = new double[end_row];
    cudaMemcpy(max_distances, max_distances_gpu, end_row * sizeof(double), cudaMemcpyDeviceToHost);

    // Copy max_indexes from device to host
    int* max_indexes = new int[end_row];
    cudaMemcpy(max_indexes, max_indexes_gpu, end_row * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(latitudes_gpu);
    cudaFree(longitudes_gpu);
    cudaFree(max_distances_gpu);
    cudaFree(max_indexes_gpu);

    // Find the city with the farthest maximum distance
    int farthest_city_index = std::distance(max_distances, std::max_element(max_distances, max_distances + end_row));

    std::cout << "The city with the farthest maximum distance is " << cities[farthest_city_index].name << " at " << *std::max_element(max_distances, max_distances + end_row) << " miles" << std::endl;


    delete[] max_distances;
    delete[] max_indexes;

    // Measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print execution time
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
