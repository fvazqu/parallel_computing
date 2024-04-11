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
                int col_index = 0;
                while (std::getline(iss, token, ',') && col_index <= 3) {
                    if (col_index == columns[0]) {
                        names.push_back(token);
                    }
                    else if (col_index == columns[1]) {
                        latitudes.push_back(std::stod(token));
                    }
                    else if (col_index == columns[2]) {
                        longitudes.push_back(std::stod(token));
                    }
                    col_index++;
                }
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

    // Measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Free memory on the GPU
    cudaFree(latitudes_gpu);
    cudaFree(longitudes_gpu);
    cudaFree(distances_gpu);

    // Count cities within 1000 miles
    int count = 0;
    for (int i = 0; i < end_row; i++) {
        if (distances[i] <= 1000 && distances[i] > 0) {
            count += 1;
        }
    }

    std::cout << "Total count of cities within 1000 miles of Edinburg, TX: " << count << std::endl;

    // Print execution time
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
