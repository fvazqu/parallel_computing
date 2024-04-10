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
    double close;       // binary to see cities less than 1000 miles
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

    int num_p = 10; // Total number of processors to be used
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
        int new_s = e - remainder;
        for (int i = new_s; i < end_row; ++i) {
            double distance = haversine(start_lat, start_lon, cities[i].latitude, cities[i].longitude);
            gathered_distances[i] = distance;
            cities[i].close = (distance <= 1000 && distance > 0) ? 1.0 : 0.0;
        }
    }

    // For Part 2
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

    if (rank == 0) {
        // Code Below for verifying distances calculated
        /*for (int i = 0; i < end_row; ++i) {
            std::cout << "Distance to " << cities[i].name << ": " << gathered_distances[i] << " miles" << std::endl;
        }*/

        //Code Below for Verifying Total count of cities within 1000 miles
        /*int counter = 0;
        for (const auto& city : cities) {
            if (city.close == 1) {
                counter++;
            }
        }
        std::cout << "Total count of cities within 1000 miles: " << counter << std::endl;*/

        auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

        //Calculate the duration in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        //Print the execution time
        std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
