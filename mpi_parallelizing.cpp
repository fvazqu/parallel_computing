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
    int num_p = 20; // Total number of processors to be used
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

    // Broadcast the index value to all other processors
    MPI_Bcast(&index, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Starting City: Cairo
    double st_lat = cities[index].latitude;
    double st_lon = cities[index].longitude;

    // Create New Distance Array
    
    // To store calculated distances
    std::vector<double> distances2(end_row);
    std::vector<double> gathered_distances2(end_row); // a buffer to store gathered distances

    int min_distance = 100000;
    int closest_index = 0;
    for (int i = s; i <= e; ++i) {
        double distance2 = haversine(st_lat, st_lon, cities[i].latitude, cities[i].longitude);
        distances2[i] = distance2;
        if (distances2[i] < min_distance && distances2[i] > 0) {
            min_distance = distances2[i];
            closest_index = i;
        }
    }

    // Gather distances from all processors to processor 0
    MPI_Gather(distances2.data() + s, cities_per_processor, MPI_DOUBLE,
        gathered_distances2.data() + s, cities_per_processor, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // If there is a remainder:
    if (rank == 0 && remainder >= 1) {
        int new_s2 = end_row - remainder;
        for (int i = new_s2; i < end_row; ++i) {
            double distance = haversine(st_lat, st_lon, cities[i].latitude, cities[i].longitude);
            gathered_distances2[i] = distance;
            // cities[i].close = (distance <= 1000 && distance > 0) ? 1.0 : 0.0;
        }
    }

    std::vector<int> min(size); // Vector to store count values from all processors
    MPI_Gather(&closest_index, 1, MPI_INT, min.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Process the gathered closest cities on processor 0
        int closest = 100000;
        int close_i = 0;
        for (int i = 0; i < size; ++i) {
            int index = min[i];
            int current = gathered_distances2[index];
            //std::cout << "Processor: " << i << ", Closest City: " << cities[index].name << ", Distance: " << gathered_distances2[index] << std::endl;
            if (current < closest) {
                closest = current;
                close_i = i;
            }
        }
        // Output the closest city
        std::cout << "Closest City to Cairo, Egypt: " << cities[min[close_i]].name << ", Distance: " << gathered_distances2[min[close_i]] << " miles" << std::endl;
    }
    // Wait for all processes to synchronize 
    MPI_Barrier(MPI_COMM_WORLD);



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

    // Broadcast the index value to all other processors
    MPI_Bcast(&index2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Starting City: Folsom
    double st_lat2 = cities[index2].latitude;
    double st_lon2 = cities[index2].longitude;


    // Create New Distance Array

    // To store calculated distances
    std::vector<double> distances3(end_row);
    std::vector<double> gathered_distances3(end_row); // a buffer to store gathered distances

    int max_distance = 1;
    int farthest_index = 0;
    for (int i = s; i <= e; ++i) {
        double distance3 = haversine(st_lat2, st_lon2, cities[i].latitude, cities[i].longitude);
        distances3[i] = distance3;
        if (distances3[i] > max_distance) {
            max_distance = distances3[i];
            farthest_index = i;
        }
    }

    // Gather distances from all processors to processor 0
    MPI_Gather(distances3.data() + s, cities_per_processor, MPI_DOUBLE,
        gathered_distances3.data() + s, cities_per_processor, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // If there is a remainder:
    if (rank == 0 && remainder >= 1) {
        int new_s3 = end_row - remainder;
        //std::cout << "Remaining Cities start index: " << new_s3 << ", Remainder: " << remainder << std::endl;
        for (int i = new_s3; i < end_row; ++i) {
            double distance = haversine(st_lat, st_lon, cities[i].latitude, cities[i].longitude);
            gathered_distances3[i] = distance;
            //std::cout << "Remaining Cities: " << cities[i].name << ", Distance: " << gathered_distances3[i] << std::endl;
            if (distance > max_distance) {
				max_distance = distance;
				farthest_index = i;
                std::cout << "New Farthest City: " << cities[i].name << ", Distance: " << gathered_distances3[i] << std::endl;
			}
        }
    }

    std::vector<int> max(size); // Vector to store count values from all processors
    MPI_Gather(&farthest_index, 1, MPI_INT, max.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Process the gathered farthest cities on processor 0
        int farthest = 1;
        int far_i = 0;
        for (int i = 0; i < size; ++i) {
            int index = max[i];
            int current = gathered_distances3[index];
            // std::cout << "Processor: " << i << ", Farthest City: " << cities[index].name << ", Distance: " << gathered_distances3[index] << std::endl;
            if (current > farthest) {
                farthest = current;
                far_i = i;
            }
        }
        // Output the closest city
        std::cout << "Farthest City to Folsom, United States: " << cities[max[far_i]].name << ", Distance: " << gathered_distances3[max[far_i]] << " miles" << std::endl;
    }

    // Wait for all processes to synchronize 
    MPI_Barrier(MPI_COMM_WORLD);



    // Part 5:
    // Find the 2 cities furthest apart:

    // Create New Array of Max Distances for each city in index range of processor
    //std::vector<std::pair<double, int>> max_distances(cities.size());
    //std::vector<std::pair<double, int>> gathered_max_distances(cities.size());

    for (int i = s; i <= e; ++i) {
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
    }

    //If there are any remainder cities
    if (rank == 0 && remainder >= 1) {
		int new_s4 = end_row - remainder;
        for (int i = new_s4; i < end_row; ++i) {
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
		}
	}

    //Find max distance in max_distances array
    if (rank == 0) {
		double max_distance3 = 0.0;
		int max_index3 = 0;
        int start = 0;
        for (int i = 0; i < end_row; ++i) {
            double m = cities[i].max_distance;
            if ( m > max_distance3) {
				max_distance3 = m;
				max_index3 = cities[i].max_index;
                start = i;
			}
            //std::cout << "City: " << i << ", to " << max_index3 << ", Max Distance: " << m << std::endl;
		}
		// Output the furthest cities
		std::cout << "Furthest Cities: " << cities[max_index3].name << " and " << cities[start].name << ", Distance: " << max_distance3 << " miles" << std::endl;
	}


    // End part
    if (rank == 0) {

        // Print out the indexes for each processor
       /* for (int i = 0; i < num_p; i++) {
			index_s[i] = i * cities_per_processor;
			index_e[i] = (i * cities_per_processor) + cities_per_processor - 1;
			std::cout << "Processor " << i << " starts at index " << index_s[i] << " and ends at index " << index_e[i] << std::endl;
		}*/
        
        // Code Below for verifying distances calculated
        /*for (int i = 0; i < end_row; ++i) {
            std::cout << "Distance to " << cities[i].name << ": " << gathered_distances[i] << " miles" << std::endl;
        }*/
        /*for (int i = 0; i < end_row; ++i) {
            std::cout << "Distance to " << cities[i].name << ": " << gathered_distances2[i] << " miles" << std::endl;
        }*/
        /*for (int i = 0; i < end_row; ++i) {
            std::cout << "Distance to " << cities[i].name << ": " << gathered_distances3[i] << " miles" << std::endl;
        }*/
        /*for (int i = 0; i < end_row; ++i) {
        	std::cout << "Farthest City Distance to " << cities[i].name << ": " << gathered_max_distances[i].second << " miles" << std::endl;
        }*/
        /*for (int i = 0; i < end_row; ++i) {
			std::cout << "Distance from " << cities[i].name << " to " << cities[cities[i].max_index].name << ": " << cities[i].max_distance << " miles" << std::endl;
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
