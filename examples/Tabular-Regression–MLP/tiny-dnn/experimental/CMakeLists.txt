cmake_minimum_required(VERSION 3.10)
project(test_tinydnn)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/tiny-dnn)

add_executable(test_tinydnn main.cpp)