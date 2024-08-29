# High-Performance Matrix Multiplication on CPU

> **Important!** If you compile the code with GCC, use the implementation from `matmul_gcc.c`. If CLANG - it's recommended to use more compact implementation from `matmul.c`. Please don’t expect peak performance without fine-tuning the hyperparameters, such as the *number of threads, kernel and block sizes*, unless you run it on a Ryzen 7700(X). More on this in the [tutorial](https://salykova.github.io/matmul-cpu).

>In the current implementation, only 1 out of 5 loops is parallelized (the 2nd loop around the micro-kernel). For manycore processors (more than 16 cores), consider utilizing nested parallelism and parallelizing 2-3 loops to increase performance (e.g., the 5th, 3rd, and 2nd loops around the micro-kernel).

## Key Features
- Step by step [tutorial](https://salykova.github.io/matmul-cpu)
- Simple and scalable C code (<150 LOC)
- Supports arbitrary matrix sizes
- Faster than NumPy with OpenBLAS and MKL backends on Ryzen 7700
- Efficiently parallelized with just 3 lines of OpenMP directives
- Targets x86 processors with AVX2 and FMA3 instructions (=all modern Intel Core and AMD Ryzen CPUs)
- Follows the [BLIS](https://github.com/flame/blis) design
- Intuitive API `void matmul(float* A, float* B, float* C, const int M, const int N, const int K)`

## Installation
Install the following packages via `apt` if you are using a Debian-based Linux distribution
```bash
sudo apt-get install clang cmake build-essential python3-dev python3-pip libomp-dev
```
Create the virtual environment using `pip` or `conda` e.g.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
and install the Python dependencies
```bash
python -m pip install -r requirements.txt
```

## Usage
For quick testing, fine-tuning, and prototyping, use the standalone file `matmul.c` in the main folder:
```
clang -O2 -mno-avx512f -fopenmp -march=native matmul.c -o matmul.out && ./matmul.out
```
To verify the numerial accuracy, add `-DTEST`:
```
clang -O2 -mno-avx512f -fopenmp -march=native -DTEST matmul.c -o matmul.out && ./matmul.out
```

## Performance

Tested on:
- CPU: Ryzen 7 7700 8 Cores, 16 Threads
- RAM: 32GB DDR5 6000 MHz CL36
- Numpy 1.26.4
- Compiler: `clang-17`
- Compiler flags: `-O2 -mno-avx512f -march=native`
- OS: Ubuntu 22.04.4 LTS

<p align="center">
  <img src="assets/perf_vs_openblas.png" alt="openblas" width="85%">
</p>

<p align="center">
  <img src="assets/perf_vs_mkl.png" alt="mkl" width="85%">
</p>

To benchmark the code, compile `benchmark.c` using `clang`. Parameters `NTHREADS`, `MR`, `NR` , `MC`, `NC`, `KC` can be defined in [CMakeLists.txt](https://github.com/salykova/matmul.c/blob/main/src/CMakeLists.txt) or via command line as shown below:
```bash
export CC=/usr/bin/clang
cmake -B build -S . -DMR=16 -DNR=6 -DNTHREADS=16
cmake --build build
```
To reproduce the results, run:
```bash
python benchmark_numpy.py

./build/benchmark MINSIZE MAXSIZE NPTS WARMUP

python plot_benchmark.py
```
If not manually specified, default values are `MINSIZE=200`, `MAXSIZE=5000`, `NPTS=50`, `WARMUP=15`.