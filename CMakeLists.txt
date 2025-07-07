cmake_minimum_required(VERSION 3.25)
project(examples_libtorch)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to libtorch cmake config
set(CMAKE_PREFIX_PATH "./global/libtorch/linux/2.7.1/libtorch/share/cmake/")

find_package(Torch REQUIRED)

# Add submodule
add_subdirectory(./examples/Tabular-Regression–MLP/libtorch/experimental/)
