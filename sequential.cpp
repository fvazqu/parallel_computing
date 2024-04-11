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
    int start_row = 1;  // Start at row 1 instead of 0 because 0 is column titles
    int end_row = 47868; // total of 47868 rows in csv file
    int columns[] = { 0, 2, 3 }; // City name, latitude, longitude
    double start_lat = 26.3017;   // Latitude of the Edinburg, TX
    double start_lon = -98.1633;   // Longitude of the Edinburg, TX

    // To Read CSV file and populate City struct
    std::string filePath = "worldcities.csv";
    std::ifstream file(filePath);
    std::vector<City> cities;

    // To store calculated distances
    std::vector<double> distances(end_row);
    std::vector<double> gathered_distances(end_row); // a buffer to store gathered distances

    // To store closest cities to Edinburg
    std::vector<double> cl; // a buffer to store closest cities to Edinburg

    auto start = std::chrono::high_resolution_clock::now(); // Get current time before execution

    // Part 1
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


    for (int i = 0; i < end_row; ++i) {
        double distance = haversine(start_lat, start_lon, cities[i].latitude, cities[i].longitude);
        distances[i] = distance;
        cities[i].close = (distance <= 1000 && distance > 0) ? 1.0 : 0.0;
    }



    //Part 2
    int count = 0;
    for (int i = 0; i < end_row; i++) {
        if (cities[i].close == 1) {
            count += 1;
        }
    }

    std::cout << "Total count of cities within 1000 miles of Edinburg, TX: " << count << std::endl;




    // Part 3:
    // Find Cairo Egypt Coordinates
    std::string target = "Cairo";
    int index = -1;

    for (int i = 0; i < cities.size(); ++i) {
        if (cities[i].name == target) {
            index = i;
            break;
        }
    }

    // Starting City: Cairo
    double st_lat = cities[index].latitude;
    double st_lon = cities[index].longitude;

    // Create New Distance Array

    // To store calculated distances
    std::vector<double> distances2(end_row);

    double min_distance = 100000;
    int closest_index = 0;
    for (int i = 0; i < end_row; ++i) {
        double distance2 = haversine(st_lat, st_lon, cities[i].latitude, cities[i].longitude);
        distances2[i] = distance2;
        if (distances2[i] < min_distance && distances2[i] > 0) {
            min_distance = distances2[i];
            closest_index = i;
        }
    }

    // Print the closest city to Cairo
    std::cout << "The closest city to Cairo, Egypt is " << cities[closest_index].name << " at " << min_distance << " miles" << std::endl;




    // Part 4:
    // Find Folsom, United States Coordinates
    std::string target2 = "Folsom";
    int index2 = -1;

    for (int i = 0; i < cities.size(); ++i) {
        if (cities[i].name == target2) {
            index2 = i;
            break;
        }
    }

    // Starting City: Folsom
    double st_lat2 = cities[index2].latitude;
    double st_lon2 = cities[index2].longitude;


    // Create New Distance Array

    // To store calculated distances
    std::vector<double> distances3(end_row);

    double max_distance = 1;
    int farthest_index = 0;
    for (int i = 0; i < end_row; ++i) {
        double distance3 = haversine(st_lat2, st_lon2, cities[i].latitude, cities[i].longitude);
        distances3[i] = distance3;
        if (distances3[i] > max_distance) {
            max_distance = distances3[i];
            farthest_index = i;
        }
    }

    // Print the farthest city from Folsom
    std::cout << "The farthest city from Folsom, United States is " << cities[farthest_index].name << " at " << max_distance << " miles" << std::endl;



    // Part 5:
    // Find the 2 cities furthest apart:
    double total_max = 0;
    int total_index = 0;
    int startingcity = 0;
    for (int i = 0; i < end_row; ++i) {
        double max_distance2 = 0;
        int max_index = 0;
        for (int j = 0; j < end_row; j++) {
            double distance = haversine(cities[i].latitude, cities[i].longitude, cities[j].latitude, cities[j].longitude);
            if (distance > max_distance2) {
                max_distance2 = distance;
                max_index = j;
            }
        }
        cities[i].max_distance = max_distance2;
        cities[i].max_index = max_index;
        if (max_distance2 > total_max) {
            total_max = max_distance2;
            total_index = max_index;
            startingcity = i;
        }
    }

    // Output the furthest cities
    std::cout << "Furthest Cities: " << cities[startingcity].name << " and " << cities[total_index].name << ", Distance: " << total_max << " miles" << std::endl;





    // End part

    // Verify Distances Array
    /*for (int i = 0; i < end_row; i++) {
        std::cout << "Distance from Edinburg, TX to " << cities[i].name << " is " << distances[i] << " miles" << std::endl;
    }*/

    auto end = std::chrono::high_resolution_clock::now(); // Get current time after execution

    //Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //Print the execution time
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;


    return 0;
}
