set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB SRC_LIST *.cpp *.c)
add_executable(${PROJECT_NAME} ${SRC_LIST})

# 链接库注意顺序和名称
target_link_libraries(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/extern_libraries/n2/build/lib/libn2.so
    #n2              
    fmt::fmt        # fmt 库
    tcmalloc_minimal
    OpenMP::OpenMP_CXX
)

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -Wall -DINFO)
