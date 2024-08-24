cmake_minimum_required(VERSION 3.2)
project(GTI LANGUAGES CXX) 

set(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(include)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)
include_directories(${PROJECT_SOURCE_DIR}/extern_libraries/n2/include)
include_directories(/usr/include/spdlog/include) 
link_directories(${PROJECT_SOURCE_DIR}/extern_libraries/n2/build/lib)

add_subdirectory(src)