// clang-17 -O2 -mno-avx512f -DTEST -march=native -DNITER=100000 matmul_cache.c -o matmul_cache.out &&
// ./matmul_cache.out
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

#define MR 16
#define NR 6

#define MC MR * 64
#define NC NR * 256
#define KC 2000

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
#define NITER 5
#endif

#define min(x, y) ((x) < (y) ? (x) : (y))

static float blockA_packed[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static float blockB_packed[NC * KC] __attribute__((aligned(MEM_ALIGN)));

void pack_panelB(float* B, float* blockB_packed, const int nr, const int kb, const int K) {
  for (int p = 0; p < kb; p++) {
    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }
    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0;
    }
  }
}

void pack_blockB(float* B, float* blockB_packed, const int nb, const int kb, const int K) {
  for (int j = 0; j < nb; j += NR) {
    const int nr = min(NR, nb - j);
    pack_panelB(&B[j * K], &blockB_packed[j * kb], nr, kb, K);
  }
}

void pack_panelA(float* A, float* blockA_packed, const int mr, const int kb, const int M) {
  for (int p = 0; p < kb; p++) {
    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }
    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0;
    }
  }
}

void pack_blockA(float* A, float* blockA_packed, const int mb, const int kb, const int M) {
  for (int i = 0; i < mb; i += MR) {
    const int mr = min(MR, mb - i);
    pack_panelA(&A[i], &blockA_packed[i * kb], mr, kb, M);
  }
}

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n, const int k,
                 const int M) {

  __m256 C_buffer[2][6];
  __m256 b_packFloat8;
  __m256 a0_packFloat8;
  __m256 a1_packFloat8;
  __m256i masks[2];

  if (m != MR) {
    const unsigned int bit_mask = 65535;
    masks[0] = _mm256_setr_epi32(bit_mask << (m + 15), bit_mask << (m + 14), bit_mask << (m + 13), bit_mask << (m + 12),
                                 bit_mask << (m + 11), bit_mask << (m + 10), bit_mask << (m + 9), bit_mask << (m + 8));
    masks[1] = _mm256_setr_epi32(bit_mask << (m + 7), bit_mask << (m + 6), bit_mask << (m + 5), bit_mask << (m + 4),
                                 bit_mask << (m + 3), bit_mask << (m + 2), bit_mask << (m + 1), bit_mask << m);

    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_maskload_ps(&C[j * M], masks[0]);
      C_buffer[1][j] = _mm256_maskload_ps(&C[j * M + 8], masks[1]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_loadu_ps(&C[j * M]);
      C_buffer[1][j] = _mm256_loadu_ps(&C[j * M + 8]);
    }
  }
  for (int p = 0; p < k; p++) {

    a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
    a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
    C_buffer[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][0]);
    C_buffer[1][0] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][0]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
    C_buffer[0][1] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][1]);
    C_buffer[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
    C_buffer[0][2] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][2]);
    C_buffer[1][2] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][2]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
    C_buffer[0][3] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][3]);
    C_buffer[1][3] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][3]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
    C_buffer[0][4] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][4]);
    C_buffer[1][4] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][4]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
    C_buffer[0][5] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][5]);
    C_buffer[1][5] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][5]);

    blockA_packed += MR;
    blockB_packed += NR;
  }
  if (m != MR) {
    for (int j = 0; j < n; j++) {
      _mm256_maskstore_ps(&C[j * M], masks[0], C_buffer[0][j]);
      _mm256_maskstore_ps(&C[j * M + 8], masks[1], C_buffer[1][j]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      _mm256_storeu_ps(&C[j * M], C_buffer[0][j]);
      _mm256_storeu_ps(&C[j * M + 8], C_buffer[1][j]);
    }
  }
}

void matmul_cache(float* A, float* B, float* C, const int M, const int N, const int K) {
  for (int j = 0; j < N; j += NC) {
    const int nb = min(NC, N - j);
    for (int p = 0; p < K; p += KC) {
      const int kb = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nb, kb, K);
      for (int i = 0; i < M; i += MC) {
        const int mb = min(MC, M - i);
        pack_blockA(&A[p * M + i], blockA_packed, mb, kb, M);
        for (int jr = 0; jr < nb; jr += NR) {
          const int nr = min(NR, nb - jr);
          for (int ir = 0; ir < mb; ir += MR) {
            const int mr = min(MR, mb - ir);
            kernel_16x6(&blockA_packed[ir * kb], &blockB_packed[jr * kb], &C[(j + jr) * M + (i + ir)], mr, nr, kb, M);
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
      if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-4) {
        printf("MISMATCH! Element[%d][%d] %f != %f\n", i, j, mat1[j * M + i], mat2[j * M + i]);
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
    matmul_cache(A, B, C, M, N, K);
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