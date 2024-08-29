// clang-17 -O2 -mno-avx512f -march=native -DTEST -DNITER=1000 matmul_kernel.c -o matmul_kernel.out && ./matmul_kernel.out
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

#define MR 16
#define NR 6

#ifndef MDIM
#define MDIM MR * 62
#endif

#ifndef NDIM
#define NDIM NR * 166
#endif

#ifndef KDIM
#define KDIM 1000
#endif

#ifndef NITER
#define NITER 100
#endif

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int M, const int N,
                 const int K) {
    __m256 C_buffer[6][2];
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    for (int j = 0; j < N; j++) {
        C_buffer[j][0] = _mm256_loadu_ps(&C[j * M]);
        C_buffer[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
    }
    for (int p = 0; p < K; p++) {
        a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][0]);
        C_buffer[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[0][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[1][0]);
        C_buffer[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[2][0]);
        C_buffer[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[2][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[3][0]);
        C_buffer[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[3][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[4][0]);
        C_buffer[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[4][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[5][0]);
        C_buffer[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[5][1]);

        blockA_packed += 16;
        blockB_packed += 6;
    }
    for (int j = 0; j < N; j++) {
        _mm256_storeu_ps(&C[j * M], C_buffer[j][0]);
        _mm256_storeu_ps(&C[j * M + 8], C_buffer[j][1]);
    }
}

void matmul_kernel(float* A, float* B, float* C, const int M, const int N, const int K) {
    assert(M % MR == 0);
    assert(N % NR == 0);
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            kernel_16x6(&A[i], &B[j * K], &C[j * M + i], M, N, K);
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

void print_mat(float* mat, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", mat[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_rand(float* mat, const int M, const int N) {
    for (int i = 0; i < M * N; i++) {
        *mat++ = rand() / (float)RAND_MAX;
    }
}

void init_const(float* mat, const float value, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            *mat++ = value;
        }
    }
}

void compare_mats(float* mat1, float* mat2, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-3) {
                printf("MISMATCH! Element[%d][%d] %f != %f\n", i, j, mat1[j * M + i],
                       mat2[j * M + i]);
                return;
            }
        }
    }
    printf("MATCH!\n");
    return;
}

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
    const int M = MDIM;
    const int N = NDIM;
    const int K = KDIM;
    float* A = (float*)_mm_malloc(M * K * sizeof(float), MEM_ALIGN);
    float* B = (float*)_mm_malloc(K * N * sizeof(float), MEM_ALIGN);
    float* C = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    float* C_ref = (float*)_mm_malloc(M * N * sizeof(float), MEM_ALIGN);
    init_rand(A, M, K);
    init_rand(B, K, N);

#ifdef TEST
    matmul_naive(A, B, C_ref, M, N, K);
#endif
    double FLOP = 2 * (double)M * N * K;

    for (int i = 0; i < NITER; i++) {
        init_const(C, 0.0, M, N);
        uint64_t start = timer();
        matmul_kernel(A, B, C, M, N, K);
        uint64_t end = timer();

        double exec_time = (end - start) * 1e-9;
        double FLOPS = FLOP / exec_time;

        printf("Exec. time = %.3fms\n", exec_time * 1000);
        printf("GFLOPS = %.3f\n", FLOPS / 1e9);
#ifdef TEST
        compare_mats(C, C_ref, M, N);
#endif
        printf("\n");
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0;
}