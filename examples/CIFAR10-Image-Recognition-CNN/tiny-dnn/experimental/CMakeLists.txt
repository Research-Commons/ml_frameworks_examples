cmake_minimum_required(VERSION 3.10)
project(cifar10_cnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add tiny-dnn include directory
include_directories(${CMAKE_SOURCE_DIR}/tiny-dnn)

# Find required packages
find_package(Threads REQUIRED)

# Add executable
add_executable(main main.cpp)

# Link libraries
target_link_libraries(main Threads::Threads)

# Add compiler flags for tiny-dnn
target_compile_definitions(main PRIVATE DNN_USE_IMAGE_API)