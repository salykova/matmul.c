#include <immintrin.h>
#include <stdint.h>

static int8_t mask_32[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
static int mask_16[16]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[6][2];
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_masks[2];
    if (m != 16) {
        packed_masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m]));
        packed_masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m + 8]));
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

void kernel_8x12(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[12];
    __m256 b_packFloat8;
    __m256 a_packFloat8;
    __m256i packed_mask;
    if (m != 8) {
        packed_mask = _mm256_loadu_si256((__m256i*)&mask_16[8 - m]);
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_maskload_ps(&C[j * M], packed_mask);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_loadu_ps(&C[j * M]);
        }
    }
    for (int p = 0; p < k; p++) {
        a_packFloat8 = _mm256_loadu_ps(blockA_packed);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[1] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[2] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[3] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[4] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[5] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 6);
        C_buffer[6] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[6]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 7);
        C_buffer[7] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[7]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 8);
        C_buffer[8] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[8]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 9);
        C_buffer[9] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[9]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 10);
        C_buffer[10] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[10]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 11);
        C_buffer[11] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[11]);

        blockA_packed += 8;
        blockB_packed += 12;
    }
    if (m != 8) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_mask, C_buffer[j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j]);
        }
    }
}

void kernel_8x13(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[13];
    __m256 b_packFloat8;
    __m256 a_packFloat8;
    __m256i packed_mask;
    if (m != 8) {
        packed_mask = _mm256_loadu_si256((__m256i*)&mask_16[8 - m]);
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_maskload_ps(&C[j * M], packed_mask);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_loadu_ps(&C[j * M]);
        }
    }
    for (int p = 0; p < k; p++) {
        a_packFloat8 = _mm256_loadu_ps(blockA_packed);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[1] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[2] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[3] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[4] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[5] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 6);
        C_buffer[6] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[6]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 7);
        C_buffer[7] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[7]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 8);
        C_buffer[8] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[8]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 9);
        C_buffer[9] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[9]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 10);
        C_buffer[10] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[10]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 11);
        C_buffer[11] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[11]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 12);
        C_buffer[12] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[12]);

        blockA_packed += 8;
        blockB_packed += 13;
    }
    if (m != 8) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_mask, C_buffer[j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j]);
        }
    }
}

void kernel_8x14(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M) {
    __m256 C_buffer[14];
    __m256 b_packFloat8;
    __m256 a_packFloat8;
    __m256i packed_mask;
    if (m != 8) {
        packed_mask = _mm256_loadu_si256((__m256i*)&mask_16[8 - m]);
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_maskload_ps(&C[j * M], packed_mask);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_loadu_ps(&C[j * M]);
        }
    }
    for (int p = 0; p < k; p++) {
        a_packFloat8 = _mm256_loadu_ps(blockA_packed);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[1] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[2] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[3] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[4] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[5] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 6);
        C_buffer[6] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[6]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 7);
        C_buffer[7] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[7]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 8);
        C_buffer[8] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[8]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 9);
        C_buffer[9] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[9]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 10);
        C_buffer[10] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[10]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 11);
        C_buffer[11] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[11]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 12);
        C_buffer[12] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[12]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 13);
        C_buffer[13] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[13]);

        blockA_packed += 8;
        blockB_packed += 14;
    }
    if (m != 8) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_mask, C_buffer[j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j]);
        }
    }
}

void kernel_8x8(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                const int k, const int M) {
    __m256 C_buffer[8];
    __m256 b_packFloat8;
    __m256 a_packFloat8;
    __m256i packed_mask;
    if (m != 8) {
        packed_mask = _mm256_loadu_si256((__m256i*)&mask_16[8 - m]);
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_maskload_ps(&C[j * M], packed_mask);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[j] = _mm256_loadu_ps(&C[j * M]);
        }
    }
    for (int p = 0; p < k; p++) {
        a_packFloat8 = _mm256_loadu_ps(blockA_packed);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[1] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[2] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[3] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[4] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[5] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[5]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 6);
        C_buffer[6] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[6]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 7);
        C_buffer[7] = _mm256_fmadd_ps(a_packFloat8, b_packFloat8, C_buffer[7]);

        blockA_packed += 8;
        blockB_packed += 8;
    }
    if (m != 8) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * M], packed_mask, C_buffer[j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * M], C_buffer[j]);
        }
    }
}