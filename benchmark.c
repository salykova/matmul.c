#include "matmul.h"
#include "utils.h"
#include <assert.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define MEMALIGN  64
#define max(x, y) ((x) > (y) ? (x) : (y))

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int minsize = 200;
    int stepsize = 200;
    int npts = 40;
    int wniter = 5;
    int niter_start = 1001;
    int niter_end = 5;
    char* save_dir = "benchmark_results";

    if (argc > 7) {
        minsize = atoi(argv[1]);
        stepsize = atoi(argv[2]);
        npts = atoi(argv[3]);
        wniter = atoi(argv[4]);
        niter_start = atoi(argv[5]);
        niter_end = atoi(argv[6]);
        save_dir = argv[7];
    }

    assert(wniter >= 0 && npts > 0 && minsize > 0 && stepsize > 0 && (niter_start >= niter_end));

    // Warm-up
    int wmatsize = minsize + (int)(npts / 2) * stepsize;
    int m = wmatsize;
    int n = wmatsize;
    int k = wmatsize;
    float* A = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);
    float* B = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);
    float* C = (float*)_mm_malloc(wmatsize * wmatsize * sizeof(float), MEMALIGN);

    printf("================\n");
    printf("Warm-up: m=n=k=%i\n", wmatsize);

    for (int j = 0; j < wniter; j++) {
        fflush(stdout);
        printf("\r%i / %i", j + 1, wniter);
        matmul(A, B, C, m, n, k);
    }

    printf("\n");
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    // Benchmark
    int* gflops_all = (int*)malloc(npts * sizeof(int));
    int* matsizes = (int*)malloc(npts * sizeof(int));
    for (int i = 0; i < npts; i++) {
        matsizes[i] = minsize + i * stepsize;
    }

    printf("==========================\n");
    printf("Benchmark\n");
    printf("==========================\n");

    for (int i = 0; i < npts; i++) {
        int matsize = matsizes[i];
        int m = matsize;
        int n = matsize;
        int k = matsize;

        float* A = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* B = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
        float* C = (float*)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);

        init_rand(A, matsize * matsize);
        init_rand(B, matsize * matsize);

        int n_iter = max(1,
                         get_niter(matsize, niter_start, niter_end, minsize, matsizes[npts - 1]));

        uint64_t start = timer();
        for (int j = 0; j < n_iter; j++) {
            matmul(A, B, C, m, n, k);
        }
        uint64_t end = timer();

        double exec_time = (end - start) * 1e-9 / n_iter;
        double FLOP = 2 * (double)matsize * matsize * matsize;
        int gflops = (int)(FLOP / exec_time / 1e9);
        gflops_all[i] = gflops;

        printf("m=n=k=%i | GFLOPS = %i\n", matsize, gflops);
        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
    }
    printf("\n================\n");

    struct stat buffer;
    if (stat(save_dir, &buffer) == -1) {
        int status = mkdir(save_dir, 0700);
        if (status != 0) {
            printf("Error creating directory %s\n", save_dir);
            return -1;
        }
    }

    char* filename = "matmul.txt";
    char* save_path = malloc(strlen(save_dir) + strlen(filename) + 2);
    strcpy(save_path, save_dir);
    strcat(save_path, "/");
    strcat(save_path, filename);

    FILE* fptr;
    fptr = fopen(save_path, "w");
    if (fptr == NULL) {
        printf("Error opening file %s\n", filename);
        return -1;
    }
    for (int i = 0; i < npts; i++) {
        fprintf(fptr, "%i %i\n", matsizes[i], gflops_all[i]);
    }
    fclose(fptr);
    printf("Saved in %s\n", save_path);

    free(gflops_all);
    free(matsizes);
    return 0;
}