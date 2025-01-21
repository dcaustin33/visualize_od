cmake_minimum_required(VERSION 3.10)
project(kalman_filtering)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find Eigen package
find_package(Eigen3 REQUIRED)

# Set output directories
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/debug)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/release)

# Add executable
add_executable(kalman_filter
    test.cpp
    kalman_filter.cpp
)

# Link Eigen
target_link_libraries(kalman_filter Eigen3::Eigen)

# Include directories
target_include_directories(kalman_filter PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)