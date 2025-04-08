#pragma once

#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#define MR 16
#define NR 6

#define MC MR * 64
#define NC NR * 256
#define KC 2000

#define min(x, y) ((x) < (y) ? (x) : (y))

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));
static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

void pack_panelB(float* B, float* blockB_packed, const int nr, const int kc, const int K) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, const int nc, const int kc, const int K) {
    for (int j = 0; j < nc; j += NR) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

void pack_panelA(float* A, float* blockA_packed, const int mr, const int kc, const int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, const int mc, const int kc, const int M) {
    for (int i = 0; i < mc; i += MR) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void kernel_16x6(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 const int mr,
                 const int nr,
                 const int kc,
                 const int m) {

    __m256 C00 = {};
    __m256 C10 = {};
    __m256 C01 = {};
    __m256 C11 = {};
    __m256 C02 = {};
    __m256 C12 = {};
    __m256 C03 = {};
    __m256 C13 = {};
    __m256 C04 = {};
    __m256 C14 = {};
    __m256 C05 = {};
    __m256 C15 = {};

    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;

    __m256i packed_mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
    __m256i packed_mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));

    if (mr != 16) {
        switch (nr) {
        case 1 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            break;
        case 2 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            break;
        case 3 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            break;
        case 4 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            break;
        case 5 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);
            break;
        case 6 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);
            C05 = _mm256_maskload_ps(&C[5 * m], packed_mask0);
            C15 = _mm256_maskload_ps(&C[5 * m + 8], packed_mask1);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            break;
        case 2 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            break;
        case 3 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            break;
        case 4 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            break;
        case 5 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);
            break;
        case 6 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);
            C05 = _mm256_loadu_ps(&C[5 * m]);
            C15 = _mm256_loadu_ps(&C[5 * m + 8]);
            break;
        }
    }
    for (int p = 0; p < kc; p++) {
        a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
        C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
        C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
        C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
        C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
        C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C05 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C05);
        C15 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C15);

        blockA_packed += 16;
        blockB_packed += 6;
    }
    if (mr != 16) {
        switch (nr) {
        case 1 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            break;
        case 2 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            break;
        case 3 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            break;
        case 4 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            break;
        case 5 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            break;
        case 6 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            _mm256_maskstore_ps(&C[5 * m], packed_mask0, C05);
            _mm256_maskstore_ps(&C[5 * m + 8], packed_mask1, C15);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            break;
        case 2 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            break;
        case 3 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            break;
        case 4 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            break;
        case 5 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            break;
        case 6 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            _mm256_storeu_ps(&C[5 * m], C05);
            _mm256_storeu_ps(&C[5 * m + 8], C15);
            break;
        }
    }
}

void matmul_micro(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        kernel_16x6(&blockA_packed[ir * kc],
                                    &blockB_packed[jr * kc],
                                    &C[(j + jr) * M + (i + ir)],
                                    mr,
                                    nr,
                                    kc,
                                    M);
                    }
                }
            }
        }
    }
}