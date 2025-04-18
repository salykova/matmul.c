# Advanced GEMM Optimization on Modern Multi-Core x86 Processors

> **Important note:** in the current implementation, the *multithreading strategy, number of threads and tile sizes* have been specifically optimized for AMD Ryzen 7 9700X and Intel Core Ultra 265 processors to achieve maximum performance. Depending on your CPU, you may need to fine-tune these parameters and choose an alternative parallelization strategy for optimal performance. More details can be found in the [tutorial](https://salykova.github.io/matmul-cpu). For instance, on many-core server processors, it’s recommended to use nested parallelism and to parallelize multiple loops around the micro-kernel.

## Key Features

- Performance comparable to modern BLAS libraries
- Simple and compact implementation in C, no assembly code
- Step by step, beginner-friendly [tutorial](https://salykova.github.io/matmul-cpu)
- Multithreading via OpenMP
- High-level design follows [BLIS](https://github.com/flame/blis)

## Installation

Install the following packages via `apt` if you are using a Debian-based Linux distribution

```bash
sudo apt-get install cmake build-essential gnuplot libomp-dev
```

## Performance

Test environment:

- CPU: AMD Ryzen 7 9700X
- RAM: 32GB DDR5 6000 MHz CL36
- OpenBLAS v.0.3.26
- Compiler: GCC 13.3.0
- OS: Ubuntu Ubuntu 24.04.1 LTS

<p align="center">
  <img src="assets/9700x.png" alt="openblas" width="80%">
</p>

To benchmark the implementation, run
```bash
bash scripts/benchmark.sh NTHREADS 0
```

for AMD, or

```bash
bash scripts/benchmark.sh NTHREADS 1
```

for Intel CPUs. Set `NTHREADS` according to your CPU and fine-tune the tile sizes `MC, NC, KC`. The benchmark parameters such as `MINSIZE`, `STEPSIZE`, `NPTS` and etc. can be adjusted in `benchmark.sh`.

## Tests
```bash
bash scripts/test.sh NTHREADS 0/1
```
