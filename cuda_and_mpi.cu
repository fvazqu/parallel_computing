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
#include <mpi.h>
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
__device__ double haversinecuda(double lat1, double lon1, double lat2, double lon2) {
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
                double distance = haversinecuda(latitudes[i], longitudes[i], latitudes[j], longitudes[j]);
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

struct City {
    std::string name;   // name, latitude, and longitude obtained from csv file
    double latitude;
    double longitude;
    double distance;    // distance calculated from haversine formula
    double close;       // binary to see cities less than 1000 miles
    double max_distance; // max distance from one city to another
    int max_index;      // index of city with max distance
};

// Function to convert degrees to radians
double degrees_to_rads(double degrees) {
    return degrees * M_PI / 180.0;
}


double haversine(double lat1, double lon1, double lat2, double lon2) {
    double r = 3958.8;  // Radius of Earth in miles

    double lat1_r = degrees_to_rads(lat1);
    double long1_r = degrees_to_rads(lon1);
    double lat2_r = degrees_to_rads(lat2);
    double long2_r = degrees_to_rads(lon2);

    //Haversine Formula
    double a = pow(sin((lat2_r - lat1_r) / 2), 2) + cos(lat1_r) * cos(lat2_r) * pow(sin((long2_r - long1_r) / 2), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double distance = r * c;

    return distance;
}

int main(int argc, char* argv[]) {

    // Set up Configurations
    int num_p = 4; // Total number of processors to be used
    int start_row = 1;  // Start at row 1 instead of 0 because 0 is column titles
    int end_row = 47868; // total of 47868 rows in csv file
    int columns[] = { 0, 2, 3 }; // City name, latitude, longitude
    double start_lat = 26.3017;   // Latitude of the Edinburg, TX
    double start_lon = -98.1633;   // Longitude of the Edinburg, TX

    // Processor 0 reads CSV file and populates struct
    std::string filePath = "worldcities.csv";
    std::ifstream file(filePath);
    std::vector<City> cities;

    // To store calculated distances
    std::vector<double> distances(end_row);
    std::vector<double> gathered_distances(end_row); // a buffer to store gathered distances

    // To store closest cities to Edinburg
    std::vector<double> cl;



    // Part 1
    if (!file.is_open()) {
        std::cerr << "Failed to open file at " << filePath << std::endl;
        return -1;  // or handle the error in another appropriate way
    }

    // For GPU Part
    std::vector<std::string> names;
    std::vector<double> latitudes;
    std::vector<double> longitudes;

    if (file.is_open()) {
        //std::cout << "File exists at " << filePath << std::endl;
        std::string line;
        int current_row = 0;
        while (std::getline(file, line)) {
            if (current_row >= start_row && current_row <= end_row) {
                std::istringstream iss(line);
                std::string token;
                int col_index = 0;
                City city;
                while (std::getline(iss, token, ',') && col_index <= 3) {
                    for (int col : columns) {
                        if (col_index == col) {
                            if (col_index == 0) {
                                city.name = token;
                                names.push_back(token);
                            }
                            else if (col_index == 2) {
                                city.latitude = std::stod(token);
                                latitudes.push_back(std::stod(token));
                            }
                            else if (col_index == 3) {
                                city.longitude = std::stod(token);
                                longitudes.push_back(std::stod(token));
                            }
                        }
                    }
                    col_index++;
                    double distance = haversine(start_lat, start_lon, city.latitude, city.longitude);
                    city.distance = distance;
                }
                cities.push_back(city); // Store the city in the vector
                if (current_row == end_row) {
                    break; // Stop reading after the third row
                }
            }
            current_row++;
        }
        file.close();
    }

    // Start of MPI
    MPI_Init(&argc, &argv);

    auto start = std::chrono::high_resolution_clock::now(); // Get current time before execution

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Broadcast city information to all processors
    MPI_Bcast(cities.data(), cities.size() * sizeof(City), MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<double> index_s(num_p);
    std::vector<double> index_e(num_p);
    int cities_per_processor = end_row / num_p; // number of cities each processor should process
    int remainder = end_row % num_p; // Remainder cities to be handled by last processor
    int s = (rank)*cities_per_processor;
    int e = ((rank)*cities_per_processor) + cities_per_processor - 1;

    for (int i = s; i <= e; ++i) {
        double distance = haversine(start_lat, start_lon, cities[i].latitude, cities[i].longitude);
        distances[i] = distance;
        cities[i].close = (distance <= 1000 && distance > 0) ? 1.0 : 0.0;

    }

    // Gather distances from all processors to processor 0
    MPI_Gather(distances.data() + s, cities_per_processor, MPI_DOUBLE,
        gathered_distances.data() + s, cities_per_processor, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // If there is a remainder:
    if (rank == 0 && remainder >= 1) {
        int new_s1 = end_row - remainder;
        for (int i = new_s1; i < end_row; ++i) {
            double distance = haversine(start_lat, start_lon, cities[i].latitude, cities[i].longitude);
            gathered_distances[i] = distance;
            cities[i].close = (distance <= 1000 && distance > 0) ? 1.0 : 0.0;
        }
    }



    //Part 2
    int count = 0;
    for (int i = s; i <= e; i++) {
        if (cities[i].close == 1) {
            count += 1;
        }
    }

    std::vector<int> counts(size); // Vector to store count values from all processors
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Process the gathered counts on processor 0
        int total_count = 0;
        for (int i = 0; i < size; ++i) {
            total_count += counts[i];
        }
        std::cout << "Total count of cities within 1000 miles: " << total_count << std::endl;
    }

    // Wait for all processes to synchronize 
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();




   // GPU Part

    // Part 3
    // Find Cairo Egypt Coordinates
    double start_lat2 = 30.0444;
    double start_lon2 = 31.2358;

    // Allocate memory on the GPU
    double* latitudes_gpu;
    double* longitudes_gpu;
    double* distances2_gpu;
    cudaMalloc(&latitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&longitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&distances2_gpu, end_row * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(latitudes_gpu, latitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(longitudes_gpu, longitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (end_row + blockSize - 1) / blockSize;

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

    // Print the closest city to Folsom
    std::cout << "The farthest city to Folsom, United States is " << cities[far_index].name << " at " << far_distance << " miles" << std::endl;



    // Part 5

    // Allocate memory on the GPU
    /*double* latitudes_gpu;
    double* longitudes_gpu;
    cudaMalloc(&latitudes_gpu, end_row * sizeof(double));
    cudaMalloc(&longitudes_gpu, end_row * sizeof(double));*/

    double* max_distances_gpu;
    int* max_indexes_gpu;
    cudaMalloc(&max_distances_gpu, end_row * sizeof(double));
    cudaMalloc(&max_indexes_gpu, end_row * sizeof(int));

    // Copy data from host to device
    /*cudaMemcpy(latitudes_gpu, latitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(longitudes_gpu, longitudes.data(), end_row * sizeof(double), cudaMemcpyHostToDevice);*/

    // Calculate grid and block sizes
    /*int blockSize = 256;
    int numBlocks = (end_row + blockSize - 1) / blockSize;*/

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

   

    delete[] max_distances;
    delete[] max_indexes;

    std::cout << "The city with the farthest maximum distance is " << cities[farthest_city_index].name << " at " << *std::max_element(max_distances, max_distances + end_row) << " miles" << std::endl;
    auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

    //Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //Print the execution time
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    /*MPI_Finalize*/

    return 0;
}
