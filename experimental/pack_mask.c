// #include <immintrin.h>

// packed_masks[0] = _mm256_sllv_epi32(_mm256_set1_epi32(65535), _mm256_add_epi32(_mm256_set1_epi32(m), _mm256_set_epi32(8, 9, 10, 11, 12, 13, 14, 15)));
// packed_masks[1] = _mm256_sllv_epi32(_mm256_set1_epi32(65535), _mm256_add_epi32(_mm256_set1_epi32(m), _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));