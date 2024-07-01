# High-Performance Matrix Multiplication in C
<p align="center">
  <img src="assets/logo.jpeg" width="70%" alt="cpu-burn">
</p>

## Key Features
- Simple, portable and scalable C code
- Step by step [tutorial](https://salykova.github.io/matmul-cpu)
- Targets x86 processors with AVX and FMA instructions (=all modern Intel Core and AMD Ryzen CPUs)
- Faster than NumPy when fine-tuned for Ryzen 7700
- Efficiently parallelized with just 3 lines of OpenMP directives
- Follows the [BLIS](https://github.com/flame/blis) design
- Works for arbitrary matrix sizes
- Intuitive API `void matmul(float* A, float* B, float* C, const int M, const int N, const int K)`
## Performance

<p align="center">
  <img src="assets/benchmark_mt.png" alt="cpu-burn" width="80%">
</p>


<p align="center">
  <img src="assets/benchmark_mt2.png" alt="cpu-burn" width="80%">
</p>

To reproduce the results, run:
```bash
python benchmark_numpy.py

clang-17 -O2 -mno-avx512f -march=native -fopenmp benchmark.c -o benchmark.out && ./benchmark.out

python plot_benchmark.py
```