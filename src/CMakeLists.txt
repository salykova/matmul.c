add_library(matmul matmul.c kernel.c)

if((NOT DEFINED NTHREADS) OR (NTHREADS STREQUAL ""))
  message(FATAL_ERROR "Error, please specify the number of threads!")
endif()

if(INTEL_PROC)
  add_definitions(-DINTEL_PROC)
endif(INTEL_PROC)

target_compile_definitions(matmul PRIVATE NTHREADS=${NTHREADS})