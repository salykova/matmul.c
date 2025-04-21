cmake_minimum_required(VERSION 3.10)

project(MATMUL)

set(CMAKE_C_FLAGS "-O3 -march=native -Wall -mno-avx512f -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

include_directories("${PROJECT_SOURCE_DIR}/common")
include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)

add_executable(benchmark benchmark.c)
add_executable(test_matmul test.c)
add_executable(plot_data plot_data.c)

list(APPEND MATMUL_LIBS -lm -lpthread)
target_link_libraries(test_matmul PUBLIC matmul ${MATMUL_LIBS})
target_link_libraries(benchmark PUBLIC matmul ${MATMUL_LIBS})