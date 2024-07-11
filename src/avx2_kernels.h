void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M);
void kernel_8x12(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M);
void kernel_8x13(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M);
void kernel_8x14(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                 const int k, const int M);
void kernel_8x8(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n,
                const int k, const int M);