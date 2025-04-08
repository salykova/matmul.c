#include <immintrin.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
    #define NTHREADS 20
#endif

#define MC (16 * NTHREADS * 10)
#define NC (6 * NTHREADS * 60)
#define KC 1000

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

void kernel_16x6(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 int mr,
                 int nr,
                 int kc,
                 int m) {
    __m256 C00;
    __m256 C10;
    __m256 C01;
    __m256 C11;
    __m256 C02;
    __m256 C12;
    __m256 C03;
    __m256 C13;
    __m256 C04;
    __m256 C14;
    __m256 C05;
    __m256 C15;

    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_mask0;
    __m256i packed_mask1;
    if (mr != 16) {
        packed_mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
        packed_mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));
        switch (nr) {
        case 1 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            break;
        case 2 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            break;
        case 3 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            break;
        case 4 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);
                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            break;
        case 5 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
                C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
                C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            break;
        default :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);
            C05 = _mm256_maskload_ps(&C[5 * m], packed_mask0);
            C15 = _mm256_maskload_ps(&C[5 * m + 8], packed_mask1);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
                C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
                C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
                C05 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C05);
                C15 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C15);

                blockA_packed += 16;
                blockB_packed += 6;
            }

            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            _mm256_maskstore_ps(&C[5 * m], packed_mask0, C05);
            _mm256_maskstore_ps(&C[5 * m + 8], packed_mask1, C15);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            break;
        case 2 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            break;
        case 3 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            break;
        case 4 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            break;
        case 5 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
                C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
                C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            break;
        default :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);
            C05 = _mm256_loadu_ps(&C[5 * m]);
            C15 = _mm256_loadu_ps(&C[5 * m + 8]);

            for (int p = 0; p < kc; p++) {
                a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
                a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
                C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
                C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
                C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
                C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
                C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
                C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
                C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
                C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
                C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
                C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

                b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
                C05 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C05);
                C15 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C15);

                blockA_packed += 16;
                blockB_packed += 6;
            }
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            _mm256_storeu_ps(&C[5 * m], C05);
            _mm256_storeu_ps(&C[5 * m + 8], C15);
            break;
        }
    }
}


void pack_panelB(float* B, float* blockB_packed, int nr, int kc, int k) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * k + p];
        }
        for (int j = nr; j < 6; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, int nc, int kc, int k) {
#pragma omp parallel for num_threads(NTHREADS) schedule(guided)
    for (int j = 0; j < nc; j += 6) {
        int nr = min(6, nc - j);
        pack_panelB(&B[j * k], &blockB_packed[j * kc], nr, kc, k);
    }
}

void pack_panelA(float* A, float* blockA_packed, int mr, int kc, int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < 16; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, int mc, int kc, int M) {
#pragma omp parallel for num_threads(NTHREADS) schedule(guided)
    for (int i = 0; i < mc; i += 16) {
        int mr = min(16, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void matmul_parallel(float* A, float* B, float* C, int m, int n, int k) {
    for (int j = 0; j < n; j += NC) {
        int nc = min(NC, n - j);
        for (int p = 0; p < k; p += KC) {
            int kc = min(KC, k - p);
            pack_blockB(&B[j * k + p], blockB_packed, nc, kc, k);
            for (int i = 0; i < m; i += MC) {
                int mc = min(MC, m - i);
                pack_blockA(&A[p * m + i], blockA_packed, mc, kc, m);
#pragma omp parallel for num_threads(NTHREADS) schedule(guided)
                for (int jr = 0; jr < nc; jr += 6) {
                    for (int ir = 0; ir < mc; ir += 16) {
                        int nr = min(6, nc - jr);
                        int mr = min(16, mc - ir);
                        kernel_16x6(&blockA_packed[ir * kc],
                                    &blockB_packed[jr * kc],
                                    &C[(j + jr) * m + (i + ir)],
                                    mr,
                                    nr,
                                    kc,
                                    m);
                    }
                }
            }
        }
    }
}