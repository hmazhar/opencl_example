cmake_minimum_required(VERSION 2.8)
project(opencl_example)
include(FindOpenCL.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
FIND_PACKAGE(OpenCL)

find_package("OpenMP" REQUIRED)

# Add the OpenMP-specific compiler and linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")


include_directories(${OPENCL_INCLUDE_DIR})

ADD_EXECUTABLE(example main.cpp)		
TARGET_LINK_LIBRARIES(example	${OPENCL_LIBRARY})

ADD_EXECUTABLE(omp_example openmp_example.cpp)		
TARGET_LINK_LIBRARIES(omp_example)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")