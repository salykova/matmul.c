#pragma once

#include <immintrin.h>
#include <stdint.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

#define BLOCK_A_MAXSIZE 500000
#define BLOCK_B_MAXSIZE 200000

static float blockA_buffer[BLOCK_A_MAXSIZE] __attribute__((aligned(64)));
static float blockB_buffer[BLOCK_B_MAXSIZE] __attribute__((aligned(64)));

void copy_pad_blockA(float* A, float* blockA_packed, int m, int M, int K) {
    for (int p = 0; p < K; p++) {
        for (int i = 0; i < m; i++) {
            *blockA_packed = A[p * M + i];
            blockA_packed++;
        }
        for (int i = m; i < 16; i++) {
            *blockA_packed = 0.0;
            blockA_packed++;
        }
    }
}

void copy_pad_blockB(float* B, float* blockB_packed, int n, int N, int K) {
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < K; p++) {
            *blockB_packed = B[j * K + p];
            blockB_packed++;
        }
    }
    for (int j = n; j < 6; j++) {
        for (int p = 0; p < K; p++) {
            *blockB_packed = 0;
            blockB_packed++;
        }
    }
}

void kernel_16x6(float* blockA,
                 float* blockB,
                 float* C_start,
                 int m,
                 int n,
                 int M,
                 int K,
                 int blockA_ld) {

    __m256i masks[2];
    __m256 C_accum[6][2] = {};
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;

    for (int p = 0; p < K; p++) {
        a0_packFloat8 = _mm256_loadu_ps(&blockA[blockA_ld * p]);
        a1_packFloat8 = _mm256_loadu_ps(&blockA[blockA_ld * p + 8]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[p]);
        C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
        C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[1 * K + p]);
        C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
        C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[2 * K + p]);
        C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
        C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[3 * K + p]);
        C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
        C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[4 * K + p]);
        C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
        C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

        b_packFloat8 = _mm256_broadcast_ss(&blockB[5 * K + p]);
        C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
        C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);
    }

    if (m != 16) {
        const uint32_t bit_mask = 65535;
        masks[0] = _mm256_setr_epi32(bit_mask << (m + 15),
                                     bit_mask << (m + 14),
                                     bit_mask << (m + 13),
                                     bit_mask << (m + 12),
                                     bit_mask << (m + 11),
                                     bit_mask << (m + 10),
                                     bit_mask << (m + 9),
                                     bit_mask << (m + 8));
        masks[1] = _mm256_setr_epi32(bit_mask << (m + 7),
                                     bit_mask << (m + 6),
                                     bit_mask << (m + 5),
                                     bit_mask << (m + 4),
                                     bit_mask << (m + 3),
                                     bit_mask << (m + 2),
                                     bit_mask << (m + 1),
                                     bit_mask << m);
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C_start[j * M], masks[0], C_accum[j][0]);
            _mm256_maskstore_ps(&C_start[j * M + 8], masks[1], C_accum[j][1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C_start[j * M], C_accum[j][0]);
            _mm256_storeu_ps(&C_start[j * M + 8], C_accum[j][1]);
        }
    }
}

void matmul_pad(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i += 16) {
        int m = min(16, M - i);
        float* blockA = &A[i];
        int blockA_ld = M;
        if (m != 16) {
            copy_pad_blockA(&A[i], blockA_buffer, m, M, K);
            blockA = blockA_buffer;
            blockA_ld = 16;
        }
        for (int j = 0; j < N; j += 6) {
            int n = min(6, N - j);
            float* blockB = &B[j * K];
            if (n != 6) {
                copy_pad_blockB(&B[j * K], blockB_buffer, n, N, K);
                blockB = blockB_buffer;
            }
            kernel_16x6(blockA, blockB, &C[j * M + i], m, n, M, K, blockA_ld);
        }
    }
}