set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB SRC_LIST *.cpp *.c)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} n2 -ltcmalloc)

add_definitions(-std=c++11 -O3 -lboost -ltcmalloc_minimal -march=native -Wall -DINFO)
