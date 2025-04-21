add_library(matmul matmul.c kernel.c)

if((NOT DEFINED NTHREADS) OR (NTHREADS STREQUAL ""))
  message(FATAL_ERROR "Error, please specify the number of threads!")
endif()

if((NOT DEFINED OMP_SCHEDULE) OR (OMP_SCHEDULE STREQUAL ""))
  set(OMP_SCHEDULE "auto")
  message(WARNING "OMP_SCHEDULE was not specified, set to \"auto\"")
endif()

target_compile_definitions(matmul PRIVATE OMP_SCHEDULE=${OMP_SCHEDULE} NTHREADS=${NTHREADS})