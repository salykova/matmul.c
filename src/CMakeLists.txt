add_library(matmul avx2_kernels.c matmul.c)

set(kernels "16x6" "8x12" "8x13" "8x14" "8x8")

if(NOT DEFINED MR)
  set(MR 16)
endif()

if(NOT DEFINED NR)
  set(NR 6)
endif()

if(NOT "${MR}x${NR}" IN_LIST kernels)
  message(
    FATAL_ERROR
      "No kernels found for MR=${MR}, NR=${NR}.\nAvailable kernels: ${kernels}")
endif()

if(NOT DEFINED NTHREADS)
  set(NTHREADS 16)
endif()

if(NOT DEFINED MC)
  math(EXPR MC ${MR}*${NTHREADS}*4)
endif()

if(NOT DEFINED NC)
  math(EXPR NC ${NR}*${NTHREADS}*32)
endif()

if(NOT DEFINED KC)
  set(KC 1000)
endif()

target_compile_definitions(
  matmul PRIVATE MR=${MR} NR=${NR} NTHREADS=${NTHREADS} MC=${MC} NC=${NC}
                 KC=${KC})