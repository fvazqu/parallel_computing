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

struct City {
    std::string name;   // Store the entire line as a string
    double latitude;
    double longitude;
    double distance;
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

int main() {

    std::string filePath = "C:\\Insert\\Your\\Full\\Path\\here";
    std::ifstream file(filePath);
    std::vector<City> cities;

    if (!file.is_open()) {
        std::cerr << "Failed to open file at " << filePath << std::endl;
        return -1;  // or handle the error in another appropriate way
    }

    int start_row = 1;
    int end_row = 10; // total of 47869 rows in csv file
    int columns[] = {0, 2, 3}; // City name, latitude, longitude
    double start_lat = 26.3017;   // Latitude of the Edinburg, TX
    double start_lon = -98.1633;   // Longitude of the Edinburg, TX

    auto start = std::chrono::high_resolution_clock::now(); // Get current time before execution

    if (file.is_open()) {
        std::cout << "File exists at " << filePath << std::endl;
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
                            // std::cout << token << ",";
                            if (col_index == 0) {
                                city.name = token;
                            } else if (col_index == 2) {
                                city.latitude = std::stod(token);
                            } else if (col_index == 3) {
                                city.longitude = std::stod(token);
                            }
                        }
                    }
                    col_index++;
                    double distance = haversine(start_lat, start_lon, city.latitude, city.longitude);
                    city.distance = distance;
                }
                // std::cout << std::endl; // Print a newline after each row
                cities.push_back(city); // Store the city in the vector
                if (current_row == end_row) {
                    break; // Stop reading after the third row
                }
            }
            current_row++;
        }
        file.close();
    }

    auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;


    // Iterate through the cities and print the names and distances
    for (const auto& city : cities) {
        std::cout << "City: " << city.name << ", Latitude: " << city.latitude << ", Longitude: " << city.longitude << ", Distance: " << city.distance << std::endl;
    }

    return 0;
}
