cmake_minimum_required(VERSION 3.25)
project(example2_libtorch_experimental)

set(CMAKE_PREFIX_PATH "./libtorch/share/cmake/")  # Replace this

find_package(Torch REQUIRED)


add_executable(example2-libtorch-experimental main.cpp)

target_include_directories(example2-libtorch-experimental PRIVATE
    ${CMAKE_SOURCE_DIR}/xgboost/include
)

target_link_directories(example2-libtorch-experimental PRIVATE
    ${CMAKE_SOURCE_DIR}/xgboost/lib
)

set_property(TARGET example2-libtorch-experimental PROPERTY CXX_STANDARD 26)
target_link_libraries(example2-libtorch-experimental "${TORCH_LIBRARIES}" xgboost)
target_link_options(example2-libtorch-experimental PRIVATE -static-libstdc++)
