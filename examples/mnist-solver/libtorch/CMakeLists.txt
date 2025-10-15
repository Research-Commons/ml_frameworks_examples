cmake_minimum_required(VERSION 3.25)
project(example_libtorch_mnist_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to LibTorch (adjust path to your installation if not system-wide)
set(CMAKE_PREFIX_PATH "./share/cmake")  # Replace with absolute or relative path to libtorch/share/cmake if needed

find_package(Torch REQUIRED)

# Enable debug symbols for profiling
set(CMAKE_BUILD_TYPE Debug)  # Ensures debug symbols are included
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -g -fno-omit-frame-pointer")  # Add debug symbols and preserve frame pointers

# Add executable
add_executable(main main.cpp)

target_link_libraries(main PRIVATE
    "${TORCH_LIBRARIES}"
)

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
