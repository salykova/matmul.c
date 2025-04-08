#include "helper.h"
#include <immintrin.h>

#ifdef TEST
    #include "matmul_naive.h"
#endif

#ifndef MATMUL_VER
    #define MATMUL_VER matmul_kernel
#endif

#define EXT .h

#define STR(x)             #x
#define STRINGIFY_MACRO(x) STR(x)
#define EXPAND(x)          x
#define CONCAT(n1, n2)     STRINGIFY_MACRO(EXPAND(n1)EXPAND(n2))

#include CONCAT(MATMUL_VER, EXT)

#define MEM_ALIGN 64

#ifndef MDIM
    #define MDIM 1000
#endif

#ifndef NDIM
    #define NDIM 1000
#endif

#ifndef KDIM
    #define KDIM 1000
#endif

#ifndef NITER
    #define NITER 100
#endif

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
        MATMUL_VER(A, B, C, M, N, K);
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