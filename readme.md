# High-Performance Matrix Multiplication in C
<p align="center">
  <img src="assets/logo.jpeg" width="300" height="300" alt="cpu-burn">
</p>

> **Important!** Please don’t expect peak performance without fine-tuning the hyperparameters, such as the *number of threads, kernel and block sizes*, unless you are running it on a Ryzen 7700(X). More on this in the [tutorial](https://salykova.github.io/matmul-cpu).

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

Tested on:
- CPU: Ryzen 7 7700 8 Cores, 16 Threads
- RAM: 32GB DDR5 6000 MHz CL36
- Numpy 1.26.4
- Compiler: clang-17
- Compiler flags: -O2 -mno-avx512f -march=native
- OS: Ubuntu 22.04.4 LTS

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
