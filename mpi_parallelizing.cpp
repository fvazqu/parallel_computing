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

struct City {
    std::string name;   // name, latitude, and longitude obtained from csv file
    double latitude;
    double longitude;
    double distance;    // distance calculated from haversine formula
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

    int num_p = 5; // Total number of processors to be used
    int start_row = 1;  // Start at row 1 instead of 0 because 0 is column titles
    int end_row = 5000; // total of 47869 rows in csv file
    int columns[] = { 0, 2, 3 }; // City name, latitude, longitude
    double start_lat = 26.3017;   // Latitude of the Edinburg, TX
    double start_lon = -98.1633;   // Longitude of the Edinburg, TX

    // Processor 0 reads CSV file and populates struct
    std::string filePath = "worldcities.csv";
    std::ifstream file(filePath);
    std::vector<City> cities;

    // Calculate distances from each city to the starting city
    std::vector<double> distances(end_row);
    std::vector<double> gathered_distances(end_row); // a buffer to store gathered distances

    if (!file.is_open()) {
        std::cerr << "Failed to open file at " << filePath << std::endl;
        return -1;  // or handle the error in another appropriate way
    }

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
                            }
                            else if (col_index == 2) {
                                city.latitude = std::stod(token);
                            }
                            else if (col_index == 3) {
                                city.longitude = std::stod(token);
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

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = std::chrono::high_resolution_clock::now(); // Get current time before execution

    // Broadcast city information to all processors
    MPI_Bcast(cities.data(), cities.size() * sizeof(City), MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<double> index_s(num_p);
    std::vector<double> index_e(num_p);
    int cities_per_processor = end_row / num_p; // number of cities each processor should process
    int remainder = end_row % num_p; // Remainder cities to be handled by last processor
    int s = (rank)*cities_per_processor;
    int e = ((rank)*cities_per_processor) + cities_per_processor - 1;


    // Adjust cities_per_processor for the last processor
    if (rank == num_p - 1) {
        e += remainder;
    }

    for (int i = s; i <= e; ++i) {
        double distance = haversine(start_lat, start_lon, cities[i].latitude, cities[i].longitude);
        distances[i] = distance;
        // std::cout << "City: " << i << ", Distance: " << distance << std::endl;
    }

    // Gather distances from all processors to processor 0
    MPI_Gather(distances.data() + s, cities_per_processor, MPI_DOUBLE,
               gathered_distances.data() + s, cities_per_processor, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_count = duration.count();
    std::cout << "Processor: " << rank << ", Time: " << duration_count << std::endl;

    // Gather execution times from all processors
    std::vector<long long> execution_times(size);
    MPI_Gather(&duration_count, 1, MPI_LONG_LONG, execution_times.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Wait for all processes to synchronize 
    MPI_Barrier(MPI_COMM_WORLD);

    //Print out Process #, cities_per_processor,  start index, and end index
    std::cout << "Process #: " << rank << ", Cities: " << cities_per_processor << ", Start: " << s << ", End : " << e << std::endl;

    if (rank == 0) {
        // Print out Distances
        for (int i = 0; i < end_row; ++i) {
            std::cout << "Distance to city " << i << ": " << gathered_distances[i] << std::endl;
        }

        //Print out total execution time
        long long total_execution_time = 0;
        for (int i = 0; i < size; ++i) {
            total_execution_time += execution_times[i];
        }
        std::cout << "Total execution time across all processors: " << total_execution_time << " milliseconds" << std::endl;

        //auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

        ////Calculate the duration in milliseconds
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        ////Print the execution time
        //std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    }


    MPI_Finalize();


    //auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

    ////Calculate the duration in milliseconds
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ////Print the execution time
    //std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
