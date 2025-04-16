#pragma once
#include <immintrin.h>

void kernel_16x6_zero_init_accum(float* blockA_packed,
                                 float* blockB_packed,
                                 float* C,
                                 int mr,
                                 int nr,
                                 int kc,
                                 int M);

void kernel_16x6_load_accum(float* blockA_packed,
                            float* blockB_packed,
                            float* C,
                            int mr,
                            int nr,
                            int kc,
                            int M);