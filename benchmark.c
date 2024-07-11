#include "matmul.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEMALIGN 64

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

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main(int argc, char* argv[]) {
    int MINSIZE;
    int MAXSIZE;
    int NPTS;
    int WARMUP;
    if (argc < 5) {
        MINSIZE = 200;
        MAXSIZE = 5000;
        NPTS = 50;
        WARMUP = 15;
    }
    else {
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
    int global_max_gflops = 0;
    int mat_sizes[NPTS];

    double delta_size = (double)(MAXSIZE - MINSIZE) / (NPTS - 1);
    for (int i = 0; i < NPTS - 1; i++) {
        mat_sizes[i] = MINSIZE + i * delta_size;
    }
    mat_sizes[NPTS - 1] = MAXSIZE;

    // Warmup
    printf("Warm-up:\n");
    float* A = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    float* B = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    float* C = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEMALIGN);
    for (int j = 0; j < WARMUP; j++) {
        fflush(stdout);
        printf("\r%i / %i", j + 1, WARMUP);
        init_const(C, 0.0, MAXSIZE, MAXSIZE);
        matmul(A, B, C, MAXSIZE, MAXSIZE, MAXSIZE);
    }
    printf("\n");
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    printf("Benchmark:\n");
    for (int i = 0; i < NPTS; i++) {
        printf("\r%i / %i", i + 1, NPTS);
        fflush(stdout);
        int mat_size = mat_sizes[i];
        float* A = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEMALIGN);
        float* B = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEMALIGN);
        float* C = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEMALIGN);

        init_rand(A, mat_size, mat_size);
        init_rand(B, mat_size, mat_size);

        double FLOP = 2 * (double)mat_size * mat_size * mat_size;
        double avg_exec_time = 0;
        double max_exec_time = 0;
        double min_exec_time = 1e69;
        int n_iter = (int)(200000 / mat_size);
        for (int j = 0; j < n_iter; j++) {
            init_const(C, 0.0, mat_size, mat_size);
            uint64_t start = timer();
            matmul(A, B, C, mat_size, mat_size, mat_size);
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
        global_max_gflops = global_max_gflops > max_gflops[i] ? global_max_gflops : max_gflops[i];

        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
    }
    printf("\n================\n");
    FILE* fptr;
    const char* filename = "benchmark_c.txt";
    fptr = fopen(filename, "w");
    if (fptr == NULL) {
        printf("Error opening the file %s\n", filename);
        return -1;
    }
    for (int i = 0; i < NPTS; i++) {
        fprintf(fptr, "%i %i %i %i\n", mat_sizes[i], min_gflops[i], max_gflops[i], avg_gflops[i]);
    }
    fclose(fptr);
    printf("PEAK GFLOPS = %i\n", global_max_gflops);
    printf("Benchmark results were saved in %s\n", filename);
    return 0;
}