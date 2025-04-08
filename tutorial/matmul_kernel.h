#pragma once
#include <immintrin.h>

void kernel_16x6(float* A_start, float* B_start, float* C_start, int M, int N, int K) {
    __m256 C_accum[6][2] = {};
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;

    for (int p = 0; p < K; p++) {
        a0_packFloat8 = _mm256_loadu_ps(&A_start[p * M]);
        a1_packFloat8 = _mm256_loadu_ps(&A_start[p * M + 8]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[p]);
        C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
        C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[1 * K + p]);
        C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
        C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[2 * K + p]);
        C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
        C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[3 * K + p]);
        C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
        C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[4 * K + p]);
        C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
        C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

        b_packFloat8 = _mm256_broadcast_ss(&B_start[5 * K + p]);
        C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
        C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);
    }

    for (int j = 0; j < 6; j++) {
        _mm256_storeu_ps(&C_start[j * M], C_accum[j][0]);
        _mm256_storeu_ps(&C_start[j * M + 8], C_accum[j][1]);
    }
}

void matmul_kernel(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int i = 0; i < M; i += 16) {
        for (int j = 0; j < N; j += 6) {
            kernel_16x6(&A[i], &B[j * K], &C[j * M + i], M, N, K);
        }
    }
}