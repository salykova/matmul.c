# High-Performance Matrix Multiplication on CPUs

> **Important!** The code should be compiled with Clang. GCC causes the program to run 1.5 to 2 times slower on my machine. Please don’t expect peak performance without fine-tuning the hyperparameters, such as the *number of threads, kernel and block sizes*, unless you are running it on a Ryzen 7700(X). More on this in the [tutorial](https://salykova.github.io/matmul-cpu).

## Key Features
- Simple and scalable C code (<150 LOC)
- Step by step [tutorial](https://salykova.github.io/matmul-cpu)
- Targets x86 processors with AVX and FMA instructions (=all modern Intel Core and AMD Ryzen CPUs)
- Faster than OpenBLAS when fine-tuned for Ryzen 7700
- Efficiently parallelized with just 3 lines of OpenMP directives
- Follows the [BLIS](https://github.com/flame/blis) design
- Works for arbitrary matrix sizes
- Intuitive API `void matmul(float* A, float* B, float* C, const int M, const int N, const int K)`

## How to use
For quick testing, fine-tuning, and prototyping, use the standalone file `matmul.c` in the main folder:
```
clang-17 -O2 -mno-avx512f -fopenmp -march=native matmul.c -o matmul.out && ./matmul.out
```
To verify the numerial accuracy of the implementation, add `-DTEST`:
```
clang-17 -O2 -mno-avx512f -fopenmp -march=native -DTEST matmul.c -o matmul.out && ./matmul.out
```

## Performance
<p align="center">
  <img src="assets/matmul_perf.png" alt="mt1" width="80%">
</p>

Tested on:
- CPU: Ryzen 7 7700 8 Cores, 16 Threads
- RAM: 32GB DDR5 6000 MHz CL36
- Numpy 1.26.4
- Compiler: `clang-17`
- Compiler flags: `-O2 -mno-avx512f -march=native`
- OS: Ubuntu 22.04.4 LTS

To reproduce the results, run:
```bash
python benchmark_numpy.py

clang-17 -O2 -mno-avx512f -march=native -fopenmp benchmark.c -o benchmark.out && ./benchmark.out

python plot_benchmark.py
```
<p align="center">
  <img src="assets/htop.png" alt="htop" width="80%">
</p>
