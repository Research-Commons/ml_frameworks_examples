cmake_minimum_required(VERSION 3.25)
project(example1_libtorch_experimental)

set(CMAKE_PREFIX_PATH "./libtorch/share/cmake")  # Replace this

find_package(Torch REQUIRED)

add_executable(example1-libtorch-experimental main.cpp)
set_property(TARGET example1-libtorch-experimental PROPERTY CXX_STANDARD 26)
target_link_libraries(example1-libtorch-experimental "${TORCH_LIBRARIES}")
target_link_options(example1-libtorch-experimental PRIVATE -static-libstdc++)
