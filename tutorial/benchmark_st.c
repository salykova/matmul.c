// clang-17 -O2 -mno-avx512f -march=native benchmark_st.c -o benchmark_st.out && ./benchmark_st.out
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

#define MR 16
#define NR 6

#define MC MR * 64
#define NC NR * 256
#define KC 2000

#ifndef NITER
#define NITER 200
#endif

#ifndef MINSIZE
#define MINSIZE 200
#endif

#ifndef MAXSIZE
#define MAXSIZE 2000
#endif

#ifndef NPTS
#define NPTS 50
#endif

#define min(x, y) ((x) < (y) ? (x) : (y))

static float blockA_packed[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static float blockB_packed[NC * KC] __attribute__((aligned(MEM_ALIGN)));

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

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[6][2];
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_masks[2];
    if (m != 16) {
        const unsigned int bit_mask = 65535;
        packed_masks[0] = _mm256_setr_epi32(
            bit_mask << (m + 15), bit_mask << (m + 14), bit_mask << (m + 13), bit_mask << (m + 12),
            bit_mask << (m + 11), bit_mask << (m + 10), bit_mask << (m + 9), bit_mask << (m + 8));
        packed_masks[1] = _mm256_setr_epi32(
            bit_mask << (m + 7), bit_mask << (m + 6), bit_mask << (m + 5), bit_mask << (m + 4),
            bit_mask << (m + 3), bit_mask << (m + 2), bit_mask << (m + 1), bit_mask << m);
        for (int j = 0; j < n; j++) {
            C_buffer[j][0] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
            C_buffer[j][1] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j][0] = _mm256_loadu_ps(&C[j * M]);
            C_buffer[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
        }
    }
    for (int p = 0; p < k; p++) {
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
    if (m != 16) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_buffer[j][0]);
            _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_buffer[j][1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j][0]);
            _mm256_storeu_ps(&C[j * M + 8], C_buffer[j][1]);
        }
    }
}

void matmul_cache(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        const int nr = min(NR, nc - jr);
                        kernel_16x6(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                                    &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
                    }
                }
            }
        }
    }
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

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
    int avg_gflops[NPTS];
    int min_gflops[NPTS];
    int max_gflops[NPTS];

    int mat_sizes[NPTS];
    double size_increment = (double)(MAXSIZE - MINSIZE) / (NPTS - 1);
    for (int i = 0; i < NPTS - 1; i++) {
        mat_sizes[i] = MINSIZE + i * size_increment;
    }
    mat_sizes[NPTS - 1] = MAXSIZE;

    // Warmup
    printf("Warm-up:\n");
    float* A = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
    float* B = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
    float* C = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
    for (int j = 0; j < 20; j++) {
        fflush(stdout);
        printf("\r%i / %i", j + 1, 20);
        init_const(C, 0.0, MAXSIZE, MAXSIZE);
        matmul_cache(A, B, C, MAXSIZE, MAXSIZE, MAXSIZE);
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
        float* A = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);
        float* B = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);
        float* C = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);

        init_rand(A, mat_size, mat_size);
        init_rand(B, mat_size, mat_size);

        double FLOP = 2 * (double)mat_size * mat_size * mat_size;
        double avg_exec_time = 0;
        double max_exec_time = 0;
        double min_exec_time = 1e69;

        for (int j = 0; j < NITER; j++) {
            init_const(C, 0.0, mat_size, mat_size);
            uint64_t start = timer();
            matmul_cache(A, B, C, mat_size, mat_size, mat_size);
            uint64_t end = timer();
            float exec_time = (end - start) * 1e-9;
            max_exec_time = exec_time > max_exec_time ? exec_time : max_exec_time;
            min_exec_time = exec_time < min_exec_time ? exec_time : min_exec_time;
            avg_exec_time += exec_time;
        }

        avg_exec_time /= NITER;
        avg_gflops[i] = (int)(FLOP / avg_exec_time / 1e9);
        max_gflops[i] = (int)(FLOP / min_exec_time / 1e9);
        min_gflops[i] = (int)(FLOP / max_exec_time / 1e9);

        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
    }
    printf("\n================\n");
    FILE* fptr;
    const char* filename = "benchmark_c.txt";
    fptr = fopen(filename, "w");
    if (fptr == NULL) {
        printf("Error opening the file %s", filename);
        return -1;
    }
    for (int i = 0; i < NPTS; i++) {
        fprintf(fptr, "%i %i %i %i\n", mat_sizes[i], min_gflops[i], max_gflops[i], avg_gflops[i]);
    }
    fclose(fptr);
    printf("Benchmark results were saved in %s\n", filename);
    return 0;
}