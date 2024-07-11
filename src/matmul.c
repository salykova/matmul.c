#include "avx2_kernels.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

#if MR == 16 && NR == 6
#define kernel kernel_16x6

#elif MR == 8 && NR == 12
#define kernel kernel_8x12

#elif MR == 8 && NR == 13
#define kernel kernel_8x13

#elif MR == 8 && NR == 14
#define kernel kernel_8x14

#elif MR == 8 && NR == 8
#define kernel kernel_8x8

#else
#define kernel kernel_16x6

#endif

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

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
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
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
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for (int i = 0; i < mc; i += MR) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void matmul(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int nr = min(NR, nc - jr);
                        const int mr = min(MR, mc - ir);
                        kernel(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                               &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
            }
        }
    }
}

void matmul_naive(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int p = 0; p < K; p++) {
                C[j * M + i] += A[p * M + i] * B[j * K + p];
            }
        }
    }
}
