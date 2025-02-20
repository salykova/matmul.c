# High-Performance FP32 Matrix Multiplication on x86 CPUs

> **Important note:** Please don’t expect peak performance without fine-tuning hyperparameters such as the *number of threads, kernel size and block sizes*, unless you're running it on a Ryzen 9700X. Current parallelization strategy is optimized for desktop CPUs. For many-core server processors, consider using nested parallelism and parallelizing 2-3 loops to increase the performance.

## Key Features

- Optimized for x86 desktop CPUs with FMA3 and AVX2 instructions
- Faster than OpenBLAS and MKL
- Step by step, beginner-friendly [tutorial](https://salykova.github.io/matmul-cpu)
- Simple and compact implementation
- Multithreading via OpenMP
- High-level design follows [BLIS](https://github.com/flame/blis)

## Installation

Install the following packages via `apt` if you are using a Debian-based Linux distribution
```bash
sudo apt-get install cmake build-essential gnuplot libomp-dev
```

### Optional

To benchmark OpenBLAS, make sure you've installed it according to the [installation guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide). During installation, choose `TARGET` corresponding to your CPU and disable AVX512 instructions. For instance, if you're using Zen4/5 CPUs, compile OpenBLAS with:
```bash
make TARGET=ZEN
```
Otherwise, OpenBLAS defaults to AVX512 instructions available on Zen4/5 CPUs.

## Performance

Test enviroment:
- CPU: AMD Ryzen 7 9700X
- RAM: 32GB DDR5 6000 MHz CL36
- OpenBLAS v.0.3.26
- Compiler: GCC 13.3.0
- OS: Ubuntu Ubuntu 24.04.1 LTS

<p align="center">
  <img src="assets/9700x.png" alt="openblas" width="80%">
</p>

To benchmark the custom implementation, run
```bash
bash scripts/benchmark.sh NTHREADS
```
Set `NTHREADS` according to your CPU. For example, for Ryzen 9700X  use `NTHREADS=16`. The benchmark parameters such as `MINSIZE`, `STEPSIZE`, `NPTS` and etc. can be adjusted in `benchmark.sh`.

### Optional

Benchmark OpenBLAS using the following command
```bash
bash scripts/benchmark_openblas.sh PATH_TO_OPENBLAS
```

Run the following command to plot benchmark results:
```bash
bash scripts/plot_data.sh
```

## Tests
```bash
bash scripts/test.sh NTHREADS
```
