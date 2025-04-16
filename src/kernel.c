#include "kernel.h"

static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

inline void fma_loop_00(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline void fma_loop_01(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* C_accum_10,
                        __m256* C_accum_11,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        *C_accum_10 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_10);
        *C_accum_11 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_11);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline void fma_loop_02(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* C_accum_10,
                        __m256* C_accum_11,
                        __m256* C_accum_20,
                        __m256* C_accum_21,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        *C_accum_10 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_10);
        *C_accum_11 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_11);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        *C_accum_20 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_20);
        *C_accum_21 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_21);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline void fma_loop_03(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* C_accum_10,
                        __m256* C_accum_11,
                        __m256* C_accum_20,
                        __m256* C_accum_21,
                        __m256* C_accum_30,
                        __m256* C_accum_31,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        *C_accum_10 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_10);
        *C_accum_11 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_11);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        *C_accum_20 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_20);
        *C_accum_21 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_21);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        *C_accum_30 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_30);
        *C_accum_31 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_31);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline void fma_loop_04(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* C_accum_10,
                        __m256* C_accum_11,
                        __m256* C_accum_20,
                        __m256* C_accum_21,
                        __m256* C_accum_30,
                        __m256* C_accum_31,
                        __m256* C_accum_40,
                        __m256* C_accum_41,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        *C_accum_10 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_10);
        *C_accum_11 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_11);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        *C_accum_20 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_20);
        *C_accum_21 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_21);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        *C_accum_30 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_30);
        *C_accum_31 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_31);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        *C_accum_40 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_40);
        *C_accum_41 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_41);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline void fma_loop_05(float* blockA_packed,
                        float* blockB_packed,
                        __m256* C_accum_00,
                        __m256* C_accum_01,
                        __m256* C_accum_10,
                        __m256* C_accum_11,
                        __m256* C_accum_20,
                        __m256* C_accum_21,
                        __m256* C_accum_30,
                        __m256* C_accum_31,
                        __m256* C_accum_40,
                        __m256* C_accum_41,
                        __m256* C_accum_50,
                        __m256* C_accum_51,
                        __m256* a0_packFloat8,
                        __m256* a1_packFloat8,
                        __m256* b_packFloat8,
                        int kc) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        *C_accum_00 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_00);
        *C_accum_01 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_01);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        *C_accum_10 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_10);
        *C_accum_11 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_11);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        *C_accum_20 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_20);
        *C_accum_21 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_21);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        *C_accum_30 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_30);
        *C_accum_31 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_31);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        *C_accum_40 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_40);
        *C_accum_41 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_41);

        *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        *C_accum_50 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum_50);
        *C_accum_51 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum_51);

        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline static void build_masks(__m256i* packed_mask_0, __m256i* packed_mask_1, int mr) {
    *packed_mask_0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
    *packed_mask_1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));
}

inline void maskload_accum_00(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
}

inline void maskload_accum_01(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256* C_accum_10,
                              __m256* C_accum_11,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
    *C_accum_10 = _mm256_maskload_ps(&C[M], packed_mask_0);
    *C_accum_11 = _mm256_maskload_ps(&C[M + 8], packed_mask_1);
}

inline void maskload_accum_02(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256* C_accum_10,
                              __m256* C_accum_11,
                              __m256* C_accum_20,
                              __m256* C_accum_21,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
    *C_accum_10 = _mm256_maskload_ps(&C[M], packed_mask_0);
    *C_accum_11 = _mm256_maskload_ps(&C[M + 8], packed_mask_1);
    *C_accum_20 = _mm256_maskload_ps(&C[2 * M], packed_mask_0);
    *C_accum_21 = _mm256_maskload_ps(&C[2 * M + 8], packed_mask_1);
}

inline void maskload_accum_03(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256* C_accum_10,
                              __m256* C_accum_11,
                              __m256* C_accum_20,
                              __m256* C_accum_21,
                              __m256* C_accum_30,
                              __m256* C_accum_31,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
    *C_accum_10 = _mm256_maskload_ps(&C[M], packed_mask_0);
    *C_accum_11 = _mm256_maskload_ps(&C[M + 8], packed_mask_1);
    *C_accum_20 = _mm256_maskload_ps(&C[2 * M], packed_mask_0);
    *C_accum_21 = _mm256_maskload_ps(&C[2 * M + 8], packed_mask_1);
    *C_accum_30 = _mm256_maskload_ps(&C[3 * M], packed_mask_0);
    *C_accum_31 = _mm256_maskload_ps(&C[3 * M + 8], packed_mask_1);
}

inline void maskload_accum_04(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256* C_accum_10,
                              __m256* C_accum_11,
                              __m256* C_accum_20,
                              __m256* C_accum_21,
                              __m256* C_accum_30,
                              __m256* C_accum_31,
                              __m256* C_accum_40,
                              __m256* C_accum_41,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
    *C_accum_10 = _mm256_maskload_ps(&C[M], packed_mask_0);
    *C_accum_11 = _mm256_maskload_ps(&C[M + 8], packed_mask_1);
    *C_accum_20 = _mm256_maskload_ps(&C[2 * M], packed_mask_0);
    *C_accum_21 = _mm256_maskload_ps(&C[2 * M + 8], packed_mask_1);
    *C_accum_30 = _mm256_maskload_ps(&C[3 * M], packed_mask_0);
    *C_accum_31 = _mm256_maskload_ps(&C[3 * M + 8], packed_mask_1);
    *C_accum_40 = _mm256_maskload_ps(&C[4 * M], packed_mask_0);
    *C_accum_41 = _mm256_maskload_ps(&C[4 * M + 8], packed_mask_1);
}

inline void maskload_accum_05(float* C,
                              __m256* C_accum_00,
                              __m256* C_accum_01,
                              __m256* C_accum_10,
                              __m256* C_accum_11,
                              __m256* C_accum_20,
                              __m256* C_accum_21,
                              __m256* C_accum_30,
                              __m256* C_accum_31,
                              __m256* C_accum_40,
                              __m256* C_accum_41,
                              __m256* C_accum_50,
                              __m256* C_accum_51,
                              __m256i packed_mask_0,
                              __m256i packed_mask_1,
                              int M) {
    *C_accum_00 = _mm256_maskload_ps(C, packed_mask_0);
    *C_accum_01 = _mm256_maskload_ps(&C[8], packed_mask_1);
    *C_accum_10 = _mm256_maskload_ps(&C[M], packed_mask_0);
    *C_accum_11 = _mm256_maskload_ps(&C[M + 8], packed_mask_1);
    *C_accum_20 = _mm256_maskload_ps(&C[2 * M], packed_mask_0);
    *C_accum_21 = _mm256_maskload_ps(&C[2 * M + 8], packed_mask_1);
    *C_accum_30 = _mm256_maskload_ps(&C[3 * M], packed_mask_0);
    *C_accum_31 = _mm256_maskload_ps(&C[3 * M + 8], packed_mask_1);
    *C_accum_40 = _mm256_maskload_ps(&C[4 * M], packed_mask_0);
    *C_accum_41 = _mm256_maskload_ps(&C[4 * M + 8], packed_mask_1);
    *C_accum_50 = _mm256_maskload_ps(&C[5 * M], packed_mask_0);
    *C_accum_51 = _mm256_maskload_ps(&C[5 * M + 8], packed_mask_1);
}

inline void load_accum_00(float* C, __m256* C_accum_00, __m256* C_accum_01, int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
}

inline void load_accum_01(float* C,
                          __m256* C_accum_00,
                          __m256* C_accum_01,
                          __m256* C_accum_10,
                          __m256* C_accum_11,
                          int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
    *C_accum_10 = _mm256_loadu_ps(&C[M]);
    *C_accum_11 = _mm256_loadu_ps(&C[M + 8]);
}

inline void load_accum_02(float* C,
                          __m256* C_accum_00,
                          __m256* C_accum_01,
                          __m256* C_accum_10,
                          __m256* C_accum_11,
                          __m256* C_accum_20,
                          __m256* C_accum_21,
                          int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
    *C_accum_10 = _mm256_loadu_ps(&C[M]);
    *C_accum_11 = _mm256_loadu_ps(&C[M + 8]);
    *C_accum_20 = _mm256_loadu_ps(&C[2 * M]);
    *C_accum_21 = _mm256_loadu_ps(&C[2 * M + 8]);
}

inline void load_accum_03(float* C,
                          __m256* C_accum_00,
                          __m256* C_accum_01,
                          __m256* C_accum_10,
                          __m256* C_accum_11,
                          __m256* C_accum_20,
                          __m256* C_accum_21,
                          __m256* C_accum_30,
                          __m256* C_accum_31,
                          int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
    *C_accum_10 = _mm256_loadu_ps(&C[M]);
    *C_accum_11 = _mm256_loadu_ps(&C[M + 8]);
    *C_accum_20 = _mm256_loadu_ps(&C[2 * M]);
    *C_accum_21 = _mm256_loadu_ps(&C[2 * M + 8]);
    *C_accum_30 = _mm256_loadu_ps(&C[3 * M]);
    *C_accum_31 = _mm256_loadu_ps(&C[3 * M + 8]);
}

inline void load_accum_04(float* C,
                          __m256* C_accum_00,
                          __m256* C_accum_01,
                          __m256* C_accum_10,
                          __m256* C_accum_11,
                          __m256* C_accum_20,
                          __m256* C_accum_21,
                          __m256* C_accum_30,
                          __m256* C_accum_31,
                          __m256* C_accum_40,
                          __m256* C_accum_41,
                          int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
    *C_accum_10 = _mm256_loadu_ps(&C[M]);
    *C_accum_11 = _mm256_loadu_ps(&C[M + 8]);
    *C_accum_20 = _mm256_loadu_ps(&C[2 * M]);
    *C_accum_21 = _mm256_loadu_ps(&C[2 * M + 8]);
    *C_accum_30 = _mm256_loadu_ps(&C[3 * M]);
    *C_accum_31 = _mm256_loadu_ps(&C[3 * M + 8]);
    *C_accum_40 = _mm256_loadu_ps(&C[4 * M]);
    *C_accum_41 = _mm256_loadu_ps(&C[4 * M + 8]);
}

inline void load_accum_05(float* C,
                          __m256* C_accum_00,
                          __m256* C_accum_01,
                          __m256* C_accum_10,
                          __m256* C_accum_11,
                          __m256* C_accum_20,
                          __m256* C_accum_21,
                          __m256* C_accum_30,
                          __m256* C_accum_31,
                          __m256* C_accum_40,
                          __m256* C_accum_41,
                          __m256* C_accum_50,
                          __m256* C_accum_51,
                          int M) {
    *C_accum_00 = _mm256_loadu_ps(C);
    *C_accum_01 = _mm256_loadu_ps(&C[8]);
    *C_accum_10 = _mm256_loadu_ps(&C[M]);
    *C_accum_11 = _mm256_loadu_ps(&C[M + 8]);
    *C_accum_20 = _mm256_loadu_ps(&C[2 * M]);
    *C_accum_21 = _mm256_loadu_ps(&C[2 * M + 8]);
    *C_accum_30 = _mm256_loadu_ps(&C[3 * M]);
    *C_accum_31 = _mm256_loadu_ps(&C[3 * M + 8]);
    *C_accum_40 = _mm256_loadu_ps(&C[4 * M]);
    *C_accum_41 = _mm256_loadu_ps(&C[4 * M + 8]);
    *C_accum_50 = _mm256_loadu_ps(&C[5 * M]);
    *C_accum_51 = _mm256_loadu_ps(&C[5 * M + 8]);
}

inline void store_accum_00(float* C, __m256* C_accum_00, __m256* C_accum_01, int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
}

inline void store_accum_01(float* C,
                           __m256* C_accum_00,
                           __m256* C_accum_01,
                           __m256* C_accum_10,
                           __m256* C_accum_11,
                           int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
    _mm256_storeu_ps(&C[M], *C_accum_10);
    _mm256_storeu_ps(&C[M + 8], *C_accum_11);
}

inline void store_accum_02(float* C,
                           __m256* C_accum_00,
                           __m256* C_accum_01,
                           __m256* C_accum_10,
                           __m256* C_accum_11,
                           __m256* C_accum_20,
                           __m256* C_accum_21,
                           int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
    _mm256_storeu_ps(&C[M], *C_accum_10);
    _mm256_storeu_ps(&C[M + 8], *C_accum_11);
    _mm256_storeu_ps(&C[2 * M], *C_accum_20);
    _mm256_storeu_ps(&C[2 * M + 8], *C_accum_21);
}

inline void store_accum_03(float* C,
                           __m256* C_accum_00,
                           __m256* C_accum_01,
                           __m256* C_accum_10,
                           __m256* C_accum_11,
                           __m256* C_accum_20,
                           __m256* C_accum_21,
                           __m256* C_accum_30,
                           __m256* C_accum_31,
                           int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
    _mm256_storeu_ps(&C[M], *C_accum_10);
    _mm256_storeu_ps(&C[M + 8], *C_accum_11);
    _mm256_storeu_ps(&C[2 * M], *C_accum_20);
    _mm256_storeu_ps(&C[2 * M + 8], *C_accum_21);
    _mm256_storeu_ps(&C[3 * M], *C_accum_30);
    _mm256_storeu_ps(&C[3 * M + 8], *C_accum_31);
}

inline void store_accum_04(float* C,
                           __m256* C_accum_00,
                           __m256* C_accum_01,
                           __m256* C_accum_10,
                           __m256* C_accum_11,
                           __m256* C_accum_20,
                           __m256* C_accum_21,
                           __m256* C_accum_30,
                           __m256* C_accum_31,
                           __m256* C_accum_40,
                           __m256* C_accum_41,
                           int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
    _mm256_storeu_ps(&C[M], *C_accum_10);
    _mm256_storeu_ps(&C[M + 8], *C_accum_11);
    _mm256_storeu_ps(&C[2 * M], *C_accum_20);
    _mm256_storeu_ps(&C[2 * M + 8], *C_accum_21);
    _mm256_storeu_ps(&C[3 * M], *C_accum_30);
    _mm256_storeu_ps(&C[3 * M + 8], *C_accum_31);
    _mm256_storeu_ps(&C[4 * M], *C_accum_40);
    _mm256_storeu_ps(&C[4 * M + 8], *C_accum_41);
}

inline void store_accum_05(float* C,
                           __m256* C_accum_00,
                           __m256* C_accum_01,
                           __m256* C_accum_10,
                           __m256* C_accum_11,
                           __m256* C_accum_20,
                           __m256* C_accum_21,
                           __m256* C_accum_30,
                           __m256* C_accum_31,
                           __m256* C_accum_40,
                           __m256* C_accum_41,
                           __m256* C_accum_50,
                           __m256* C_accum_51,
                           int M) {
    _mm256_storeu_ps(C, *C_accum_00);
    _mm256_storeu_ps(&C[8], *C_accum_01);
    _mm256_storeu_ps(&C[M], *C_accum_10);
    _mm256_storeu_ps(&C[M + 8], *C_accum_11);
    _mm256_storeu_ps(&C[2 * M], *C_accum_20);
    _mm256_storeu_ps(&C[2 * M + 8], *C_accum_21);
    _mm256_storeu_ps(&C[3 * M], *C_accum_30);
    _mm256_storeu_ps(&C[3 * M + 8], *C_accum_31);
    _mm256_storeu_ps(&C[4 * M], *C_accum_40);
    _mm256_storeu_ps(&C[4 * M + 8], *C_accum_41);
    _mm256_storeu_ps(&C[5 * M], *C_accum_50);
    _mm256_storeu_ps(&C[5 * M + 8], *C_accum_51);
}

inline void maskstore_accum_00(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
}

inline void maskstore_accum_01(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256* C_accum_10,
                               __m256* C_accum_11,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
    _mm256_maskstore_ps(&C[M], packed_mask_0, *C_accum_10);
    _mm256_maskstore_ps(&C[M + 8], packed_mask_1, *C_accum_11);
}

inline void maskstore_accum_02(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256* C_accum_10,
                               __m256* C_accum_11,
                               __m256* C_accum_20,
                               __m256* C_accum_21,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
    _mm256_maskstore_ps(&C[M], packed_mask_0, *C_accum_10);
    _mm256_maskstore_ps(&C[M + 8], packed_mask_1, *C_accum_11);
    _mm256_maskstore_ps(&C[2 * M], packed_mask_0, *C_accum_20);
    _mm256_maskstore_ps(&C[2 * M + 8], packed_mask_1, *C_accum_21);
}

inline void maskstore_accum_03(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256* C_accum_10,
                               __m256* C_accum_11,
                               __m256* C_accum_20,
                               __m256* C_accum_21,
                               __m256* C_accum_30,
                               __m256* C_accum_31,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
    _mm256_maskstore_ps(&C[M], packed_mask_0, *C_accum_10);
    _mm256_maskstore_ps(&C[M + 8], packed_mask_1, *C_accum_11);
    _mm256_maskstore_ps(&C[2 * M], packed_mask_0, *C_accum_20);
    _mm256_maskstore_ps(&C[2 * M + 8], packed_mask_1, *C_accum_21);
    _mm256_maskstore_ps(&C[3 * M], packed_mask_0, *C_accum_30);
    _mm256_maskstore_ps(&C[3 * M + 8], packed_mask_1, *C_accum_31);
}

inline void maskstore_accum_04(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256* C_accum_10,
                               __m256* C_accum_11,
                               __m256* C_accum_20,
                               __m256* C_accum_21,
                               __m256* C_accum_30,
                               __m256* C_accum_31,
                               __m256* C_accum_40,
                               __m256* C_accum_41,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
    _mm256_maskstore_ps(&C[M], packed_mask_0, *C_accum_10);
    _mm256_maskstore_ps(&C[M + 8], packed_mask_1, *C_accum_11);
    _mm256_maskstore_ps(&C[2 * M], packed_mask_0, *C_accum_20);
    _mm256_maskstore_ps(&C[2 * M + 8], packed_mask_1, *C_accum_21);
    _mm256_maskstore_ps(&C[3 * M], packed_mask_0, *C_accum_30);
    _mm256_maskstore_ps(&C[3 * M + 8], packed_mask_1, *C_accum_31);
    _mm256_maskstore_ps(&C[4 * M], packed_mask_0, *C_accum_40);
    _mm256_maskstore_ps(&C[4 * M + 8], packed_mask_1, *C_accum_41);
}

inline void maskstore_accum_05(float* C,
                               __m256* C_accum_00,
                               __m256* C_accum_01,
                               __m256* C_accum_10,
                               __m256* C_accum_11,
                               __m256* C_accum_20,
                               __m256* C_accum_21,
                               __m256* C_accum_30,
                               __m256* C_accum_31,
                               __m256* C_accum_40,
                               __m256* C_accum_41,
                               __m256* C_accum_50,
                               __m256* C_accum_51,
                               __m256i packed_mask_0,
                               __m256i packed_mask_1,
                               int M) {
    _mm256_maskstore_ps(C, packed_mask_0, *C_accum_00);
    _mm256_maskstore_ps(&C[8], packed_mask_1, *C_accum_01);
    _mm256_maskstore_ps(&C[M], packed_mask_0, *C_accum_10);
    _mm256_maskstore_ps(&C[M + 8], packed_mask_1, *C_accum_11);
    _mm256_maskstore_ps(&C[2 * M], packed_mask_0, *C_accum_20);
    _mm256_maskstore_ps(&C[2 * M + 8], packed_mask_1, *C_accum_21);
    _mm256_maskstore_ps(&C[3 * M], packed_mask_0, *C_accum_30);
    _mm256_maskstore_ps(&C[3 * M + 8], packed_mask_1, *C_accum_31);
    _mm256_maskstore_ps(&C[4 * M], packed_mask_0, *C_accum_40);
    _mm256_maskstore_ps(&C[4 * M + 8], packed_mask_1, *C_accum_41);
    _mm256_maskstore_ps(&C[5 * M], packed_mask_0, *C_accum_50);
    _mm256_maskstore_ps(&C[5 * M + 8], packed_mask_1, *C_accum_51);
}

void kernel_16x6_load_accum(float* blockA_packed,
                            float* blockB_packed,
                            float* C,
                            int mr,
                            int nr,
                            int kc,
                            int M) {
    __m256 C_accum_00 = {};
    __m256 C_accum_01 = {};
    __m256 C_accum_10 = {};
    __m256 C_accum_11 = {};
    __m256 C_accum_20 = {};
    __m256 C_accum_21 = {};
    __m256 C_accum_30 = {};
    __m256 C_accum_31 = {};
    __m256 C_accum_40 = {};
    __m256 C_accum_41 = {};
    __m256 C_accum_50 = {};
    __m256 C_accum_51 = {};

    __m256 b_packFloat8 = {};
    __m256 a0_packFloat8 = {};
    __m256 a1_packFloat8 = {};
    __m256i packed_mask_0 = {};
    __m256i packed_mask_1 = {};

    if (mr != 16) {
        build_masks(&packed_mask_0, &packed_mask_1, mr);
        switch (nr) {
        case 1 :
            maskload_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
            fma_loop_00(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
            break;
        case 2 :
            maskload_accum_01(C,
                              &C_accum_00,
                              &C_accum_01,
                              &C_accum_10,
                              &C_accum_11,
                              packed_mask_0,
                              packed_mask_1,
                              M);
            fma_loop_01(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_01(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 3 :
            maskload_accum_02(C,
                              &C_accum_00,
                              &C_accum_01,
                              &C_accum_10,
                              &C_accum_11,
                              &C_accum_20,
                              &C_accum_21,
                              packed_mask_0,
                              packed_mask_1,
                              M);
            fma_loop_02(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_02(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 4 :
            maskload_accum_03(C,
                              &C_accum_00,
                              &C_accum_01,
                              &C_accum_10,
                              &C_accum_11,
                              &C_accum_20,
                              &C_accum_21,
                              &C_accum_30,
                              &C_accum_31,
                              packed_mask_0,
                              packed_mask_1,
                              M);
            fma_loop_03(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_03(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 5 :
            maskload_accum_04(C,
                              &C_accum_00,
                              &C_accum_01,
                              &C_accum_10,
                              &C_accum_11,
                              &C_accum_20,
                              &C_accum_21,
                              &C_accum_30,
                              &C_accum_31,
                              &C_accum_40,
                              &C_accum_41,
                              packed_mask_0,
                              packed_mask_1,
                              M);
            fma_loop_04(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_04(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               &C_accum_40,
                               &C_accum_41,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 6 :
            maskload_accum_05(C,
                              &C_accum_00,
                              &C_accum_01,
                              &C_accum_10,
                              &C_accum_11,
                              &C_accum_20,
                              &C_accum_21,
                              &C_accum_30,
                              &C_accum_31,
                              &C_accum_40,
                              &C_accum_41,
                              &C_accum_50,
                              &C_accum_51,
                              packed_mask_0,
                              packed_mask_1,
                              M);
            fma_loop_05(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &C_accum_50,
                        &C_accum_51,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_05(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               &C_accum_40,
                               &C_accum_41,
                               &C_accum_50,
                               &C_accum_51,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            load_accum_00(C, &C_accum_00, &C_accum_01, M);
            fma_loop_00(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_00(C, &C_accum_00, &C_accum_01, M);
            break;
        case 2 :
            load_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
            fma_loop_01(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
            break;
        case 3 :
            load_accum_02(C,
                          &C_accum_00,
                          &C_accum_01,
                          &C_accum_10,
                          &C_accum_11,
                          &C_accum_20,
                          &C_accum_21,
                          M);
            fma_loop_02(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_02(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           M);
            break;
        case 4 :
            load_accum_03(C,
                          &C_accum_00,
                          &C_accum_01,
                          &C_accum_10,
                          &C_accum_11,
                          &C_accum_20,
                          &C_accum_21,
                          &C_accum_30,
                          &C_accum_31,
                          M);
            fma_loop_03(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_03(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           M);
            break;
        case 5 :
            load_accum_04(C,
                          &C_accum_00,
                          &C_accum_01,
                          &C_accum_10,
                          &C_accum_11,
                          &C_accum_20,
                          &C_accum_21,
                          &C_accum_30,
                          &C_accum_31,
                          &C_accum_40,
                          &C_accum_41,
                          M);
            fma_loop_04(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_04(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           &C_accum_40,
                           &C_accum_41,
                           M);

            break;
        case 6 :
            load_accum_05(C,
                          &C_accum_00,
                          &C_accum_01,
                          &C_accum_10,
                          &C_accum_11,
                          &C_accum_20,
                          &C_accum_21,
                          &C_accum_30,
                          &C_accum_31,
                          &C_accum_40,
                          &C_accum_41,
                          &C_accum_50,
                          &C_accum_51,
                          M);
            fma_loop_05(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &C_accum_50,
                        &C_accum_51,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_05(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           &C_accum_40,
                           &C_accum_41,
                           &C_accum_50,
                           &C_accum_51,
                           M);
            break;
        }
    }
}

void kernel_16x6_zero_init_accum(float* blockA_packed,
                                 float* blockB_packed,
                                 float* C,
                                 int mr,
                                 int nr,
                                 int kc,
                                 int M) {
    __m256 C_accum_00 = {};
    __m256 C_accum_01 = {};
    __m256 C_accum_10 = {};
    __m256 C_accum_11 = {};
    __m256 C_accum_20 = {};
    __m256 C_accum_21 = {};
    __m256 C_accum_30 = {};
    __m256 C_accum_31 = {};
    __m256 C_accum_40 = {};
    __m256 C_accum_41 = {};
    __m256 C_accum_50 = {};
    __m256 C_accum_51 = {};

    __m256 b_packFloat8 = {};
    __m256 a0_packFloat8 = {};
    __m256 a1_packFloat8 = {};
    __m256i packed_mask_0 = {};
    __m256i packed_mask_1 = {};

    if (mr != 16) {
        build_masks(&packed_mask_0, &packed_mask_1, mr);
        switch (nr) {
        case 1 :
            fma_loop_00(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_00(C, &C_accum_00, &C_accum_01, packed_mask_0, packed_mask_1, M);
            break;
        case 2 :
            fma_loop_01(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_01(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 3 :
            fma_loop_02(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_02(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 4 :
            fma_loop_03(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_03(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 5 :
            fma_loop_04(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_04(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               &C_accum_40,
                               &C_accum_41,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        case 6 :
            fma_loop_05(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &C_accum_50,
                        &C_accum_51,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            maskstore_accum_05(C,
                               &C_accum_00,
                               &C_accum_01,
                               &C_accum_10,
                               &C_accum_11,
                               &C_accum_20,
                               &C_accum_21,
                               &C_accum_30,
                               &C_accum_31,
                               &C_accum_40,
                               &C_accum_41,
                               &C_accum_50,
                               &C_accum_51,
                               packed_mask_0,
                               packed_mask_1,
                               M);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            fma_loop_00(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_00(C, &C_accum_00, &C_accum_01, M);
            break;
        case 2 :
            fma_loop_01(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_01(C, &C_accum_00, &C_accum_01, &C_accum_10, &C_accum_11, M);
            break;
        case 3 :
            fma_loop_02(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_02(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           M);
            break;
        case 4 :
            fma_loop_03(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_03(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           M);
            break;
        case 5 :
            fma_loop_04(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_04(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           &C_accum_40,
                           &C_accum_41,
                           M);

            break;
        case 6 :
            fma_loop_05(blockA_packed,
                        blockB_packed,
                        &C_accum_00,
                        &C_accum_01,
                        &C_accum_10,
                        &C_accum_11,
                        &C_accum_20,
                        &C_accum_21,
                        &C_accum_30,
                        &C_accum_31,
                        &C_accum_40,
                        &C_accum_41,
                        &C_accum_50,
                        &C_accum_51,
                        &a0_packFloat8,
                        &a1_packFloat8,
                        &b_packFloat8,
                        kc);
            store_accum_05(C,
                           &C_accum_00,
                           &C_accum_01,
                           &C_accum_10,
                           &C_accum_11,
                           &C_accum_20,
                           &C_accum_21,
                           &C_accum_30,
                           &C_accum_31,
                           &C_accum_40,
                           &C_accum_41,
                           &C_accum_50,
                           &C_accum_51,
                           M);
            break;
        }
    }
}