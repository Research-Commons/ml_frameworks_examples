cmake_minimum_required(VERSION 3.25)
project(example_libtorch)

set(CMAKE_PREFIX_PATH "./share/cmake")  # Replace this

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 26)
