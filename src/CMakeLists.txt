add_library(matmul matmul.c kernel.c)

target_compile_definitions(matmul PRIVATE)