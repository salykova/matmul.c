#include "utils.h"
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef OPENBLAS
    #include <cblas.h>
#else
    #include "matmul.h"
#endif

#define MEMALIGN  64
#define max(x, y) ((x) > (y) ? (x) : (y))

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int MINSIZE = 200;
    int STEPSIZE = 200;
    int NPTS = 40;
    int WNITER = 5;
    int NITER_START = 1001;
    int NITER_END = 5;

    if (argc > 6) {
        MINSIZE = atoi(argv[1]);
        STEPSIZE = atoi(argv[2]);
        NPTS = atoi(argv[3]);
        WNITER = atoi(argv[4]);
        NITER_START = atoi(argv[5]);
        NITER_END = atoi(argv[6]);
    }

    assert(NPTS > 0 && MINSIZE > 0 && STEPSIZE > 0 && NPTS > 0 && (NITER_START >= NITER_END));

    // Warm-up
    int wmatsize = MINSIZE + (int)(NPTS / 2) * STEPSIZE;
    printf("================\n");
    printf("Warm-up: m=n=k=%i\n", wmatsize);
    float* A = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);
    float* B = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);
    float* C = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);
    int m = wmatsize;
    int n = wmatsize;
    int k = wmatsize;
    for (int j = 0; j < WNITER; j++) {
        fflush(stdout);
        printf("\r%i / %i", j + 1, WNITER);
        init_const(C, 0.0, wmatsize * wmatsize);
#ifdef OPENBLAS
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
#else
        matmul(A, B, C, m, n, k);
#endif
    }
    printf("\n");
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    // Benchmark
    printf("==========================\n");
#ifdef OPENBLAS
    printf("Benchmarking OpenBLAS\n");
#else
    printf("Benchmarking matmul.c\n");
#endif
    printf("==========================\n");

    int* gflops_median_all = (int*)malloc(NPTS * sizeof(int));
    int* gflops_max_all = (int*)malloc(NPTS * sizeof(int));
    int* matsizes = (int*)malloc(NPTS * sizeof(int));
    for (int i = 0; i < NPTS; i++) {
        matsizes[i] = MINSIZE + i * STEPSIZE;
    }

    for (int i = 0; i < NPTS; i++) {
        int matsize = matsizes[i];
        int m = matsize;
        int n = matsize;
        int k = matsize;

        float* A = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* B = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* C = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);

        init_rand(A, matsize * matsize);
        init_rand(B, matsize * matsize);

        int n_iter =
            max(1, scedule_niter(matsize, NITER_START, NITER_END, MINSIZE, matsizes[NPTS - 1]));
        float* runtimes = (float*)malloc(n_iter * sizeof(float));
        double FLOP = 2 * (double)matsize * matsize * matsize;

        for (int j = 0; j < n_iter; j++) {
            init_const(C, 0.0, matsize * matsize);
            uint64_t start = timer();
#ifdef OPENBLAS
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k, 0, C, m);
#else
            matmul(A, B, C, m, n, k);
#endif
            uint64_t end = timer();
            runtimes[j] = (end - start) * 1e-9;
        }

        qsort(runtimes, n_iter, sizeof(float), compare_floats);
        float runtime_median = n_iter % 2 == 0 ?
                                   (runtimes[n_iter / 2] + runtimes[n_iter / 2 - 1]) / 2 :
                                   runtimes[n_iter / 2];
        int gflops_median = (int)(FLOP / (double)runtime_median / 1e9);
        int gflops_max = (int)(FLOP / (double)runtimes[0] / 1e9);
        gflops_max_all[i] = gflops_max;
        gflops_median_all[i] = gflops_median;

        printf("m=n=k=%i | PEAK/MEDIAN GFLOPS = %i/%i\n", matsize, gflops_max, gflops_median);
        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
        free(runtimes);
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
        fprintf(fptr, "%i %i %i\n", matsizes[i], gflops_max_all[i], gflops_median_all[i]);
    }
    fclose(fptr);
    printf("Saved in %s\n", filename);

    free(gflops_max_all);
    free(gflops_median_all);
    free(matsizes);
    return 0;
}