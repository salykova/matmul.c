#include "kernel.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
    #define NTHREADS 8
#endif

#define MC (16 * (40 / NTHREADS) * NTHREADS)
#define NC (6 * (800 / NTHREADS) * NTHREADS)
#define KC 500

#ifndef OMP_SCHEDULE
    #define OMP_SCHEDULE auto
#endif

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

#define min(x, y) ((x) < (y) ? (x) : (y))

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

void pack_panelB(float* B, float* blockB_packed, int nr, int kc, int K) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < 6; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, int nc, int kc, int K) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int j = 0; j < nc; j += 6) {
        int nr = min(6, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

void pack_panelA(float* A, float* blockA_packed, int mr, int kc, int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < 16; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, int mc, int kc, int M) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < mc; i += 16) {
        int mr = min(16, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void matmul(float* A, float* B, float* C, int M, int N, int K) {

    // The function computes C[M x N] = A[M x K] @ B[K x N]
    // All operands are stored in column-major format, with lda=M, ldb=K, ldc=M

    for (int j = 0; j < N; j += NC) {
        int nc = min(NC, N - j);
        int kc = min(KC, K);
        pack_blockB(&B[j * K], blockB_packed, nc, kc, K);
        for (int i = 0; i < M; i += MC) {
            int mc = min(MC, M - i);
            pack_blockA(&A[i], blockA_packed, mc, kc, M);
            PRAGMA_OMP_PARALLEL_FOR
            for (int jr = 0; jr < nc; jr += 6) {
                int nr = min(6, nc - jr);
                for (int ir = 0; ir < mc; ir += 16) {
                    int mr = min(16, mc - ir);
                    kernel_16x6_zero_init_accum(&blockA_packed[ir * kc],
                                                &blockB_packed[jr * kc],
                                                &C[(j + jr) * M + (i + ir)],
                                                mr,
                                                nr,
                                                kc,
                                                M);
                }
            }
        }
        for (int p = kc; p < K; p += KC) {
            int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
                PRAGMA_OMP_PARALLEL_FOR
                for (int jr = 0; jr < nc; jr += 6) {
                    int nr = min(6, nc - jr);
                    for (int ir = 0; ir < mc; ir += 16) {
                        int mr = min(16, mc - ir);
                        kernel_16x6_load_accum(&blockA_packed[ir * kc],
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

void matmul_naive(float* A, float* B, float* C, int m, int n, int k) {
#pragma omp parallel for collapse(2) num_threads(NTHREADS)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float accumulator = 0;
            for (int p = 0; p < k; p++) {
                accumulator += A[p * m + i] * B[j * k + p];
            }
            C[j * m + i] = accumulator;
        }
    }
}
