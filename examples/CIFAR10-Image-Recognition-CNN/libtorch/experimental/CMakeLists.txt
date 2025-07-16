cmake_minimum_required(VERSION 3.25)
project(example3_libtorch_experimental)

set(CMAKE_PREFIX_PATH "./libtorch/share/cmake")  # Replace this

find_package(Torch REQUIRED)

add_executable(example3-libtorch-experimental main.cpp)
set_property(TARGET example3-libtorch-experimental PROPERTY CXX_STANDARD 26)
target_link_libraries(example3-libtorch-experimental "${TORCH_LIBRARIES}")
target_link_options(example3-libtorch-experimental PRIVATE -static-libstdc++)
