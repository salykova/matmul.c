#include "helper_matrix.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef OPENBLAS
    #include <cblas.h>
#else
    #include "matmul.h"
#endif

#define MEMALIGN 64

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int MINSIZE = 200;
    int MAXSIZE = 8000;
    int NPTS = 40;
    int WARMUP = 5;

    if (argc > 4) {
        MINSIZE = atoi(argv[1]);
        MAXSIZE = atoi(argv[2]);
        NPTS = atoi(argv[3]);
        WARMUP = atoi(argv[4]);
    }

    printf("================\n");
    printf("MINSIZE = %i\nMAXSIZE = %i\nNPTS = %i\nWARMUP = %i\n", MINSIZE, MAXSIZE, NPTS, WARMUP);
    printf("================\n");
    int avg_gflops[NPTS];
    int min_gflops[NPTS];
    int max_gflops[NPTS];
    int matsizes[NPTS];

    double delta_size = (double)(MAXSIZE - MINSIZE) / (NPTS - 1);
    for (int i = 0; i < NPTS - 1; i++) {
        matsizes[i] = MINSIZE + i * delta_size;
    }
    matsizes[NPTS - 1] = MAXSIZE;

    // Warm-up
    printf("Warm-up:\n");
    float* A = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    float* B = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    float* C = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    for (int j = 0; j < WARMUP; j++) {
        fflush(stdout);
        printf("\r%i / %i", j + 1, WARMUP);
        init_const(C, 0.0, MAXSIZE, MAXSIZE);
        int m = MAXSIZE;
        int n = MAXSIZE;
        int k = MAXSIZE;
#ifdef OPENBLAS
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k, 0, C, m);
#else
        matmul(A, B, C, m, n, k);
#endif
    }
    printf("\n");
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    printf("==========================\n");
#ifdef OPENBLAS
    printf("Benchmarking OpenBLAS ...\n");
#else
    printf("Benchmarking matmul.c ...\n");
#endif
    printf("==========================\n");
    for (int i = 0; i < NPTS; i++) {
        int matsize = matsizes[i];

        int m = matsize;
        int n = matsize;
        int k = matsize;

        float* A = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* B = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* C = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);

        init_rand(A, matsize, matsize);
        init_rand(B, matsize, matsize);

        double FLOP = 2 * (double)matsize * matsize * matsize;
        double avg_exec_time = 0;
        double max_exec_time = 0;
        double min_exec_time = 1e69;
        int n_iter = (int)(40000 / matsize);
        n_iter = n_iter > 0 ? n_iter : 1;

        for (int j = 0; j < n_iter; j++) {
            init_const(C, 0.0, matsize, matsize);
            uint64_t start = timer();
#ifdef OPENBLAS
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k, 0, C, m);
#else
            matmul(A, B, C, m, n, k);
#endif
            uint64_t end = timer();
            float exec_time = (end - start) * 1e-9;
            max_exec_time = exec_time > max_exec_time ? exec_time : max_exec_time;
            min_exec_time = exec_time < min_exec_time ? exec_time : min_exec_time;
            avg_exec_time += exec_time;
        }

        avg_exec_time /= n_iter;
        avg_gflops[i] = (int)(FLOP / avg_exec_time / 1e9);
        max_gflops[i] = (int)(FLOP / min_exec_time / 1e9);
        min_gflops[i] = (int)(FLOP / max_exec_time / 1e9);

        printf("matsize = %i, PEAK GFLOPS = %i\n", matsize, max_gflops[i]);

        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
    }
    printf("\n================\n");
    FILE* fptr;
#ifdef OPENBLAS
    const char* filename = "benchmark_openblas.txt";
#else
    const char* filename = "benchmark_matmul.txt";
#endif
    fptr = fopen(filename, "w");
    if (fptr == NULL) {
        printf("Error opening the file %s\n", filename);
        return -1;
    }
    for (int i = 0; i < NPTS; i++) {
        fprintf(fptr, "%i %i %i %i\n", matsizes[i], min_gflops[i], max_gflops[i], avg_gflops[i]);
    }
    fclose(fptr);
    printf("Saved in %s\n", filename);
    return 0;
}