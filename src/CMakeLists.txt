add_library(matmul matmul.c kernel.c)

if(NOT DEFINED NTHREADS)
  set(NTHREADS 16)
endif()

target_compile_definitions(matmul PRIVATE NTHREADS=${NTHREADS})