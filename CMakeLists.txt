cmake_minimum_required(VERSION 3.10)

project(MATMUL)


set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_C_FLAGS "-O2 -march=native -mno-avx512f -fopenmp")

add_executable(benchmark benchmark.c)
add_subdirectory(src)
target_link_libraries(benchmark PUBLIC matmul)
target_include_directories(benchmark PUBLIC "${PROJECT_BINARY_DIR}"
"${PROJECT_SOURCE_DIR}/src")