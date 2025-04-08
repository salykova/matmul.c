# Advanced GEMM Optimization on Modern x86-64 Multi-Core Processors

Tutorial: [https://salykova.github.io/matmul-cpu](https://salykova.github.io/matmul-cpu)

For example, to benchmark matmul implementation from `matmul_pad.h`, specify `-DMATMUL_VER` accordingly:

```bash
gcc -O3 -march=native -fopenmp -mno-avx512f -DMATMUL_VER=matmul_pad runner.c -o runner.out && ./runner.out
```

To test matmul implementation for correctness, add `-DTEST`:

```bash
gcc -O3 -march=native -fopenmp -mno-avx512f -DTEST -DMATMUL_VER=matmul_pad runner.c -o runner.out && ./runner.out
```