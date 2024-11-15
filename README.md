# Fast, Multi-threaded Matrix Multiplication in C from Scratch

> **Important!** Please don’t expect peak performance without fine-tuning the hyperparameters, such as the *number of threads, kernel and block sizes*, unless you run it on a Ryzen 7700(X). More on this in the [tutorial](https://salykova.github.io/matmul-cpu).

>In the current implementation, only 1 out of 5 loops is parallelized (the 2nd loop around the micro-kernel). For manycore processors (more than 16 cores), consider utilizing nested parallelism and parallelizing 2-3 loops to increase performance (e.g., the 5th, 3rd, and 2nd loops around the micro-kernel).

## Key Features
- Step by step, beginner-friendly [tutorial](https://salykova.github.io/matmul-cpu)
- Simple and scalable C code
- Supports arbitrary matrix sizes
- Faster than OpenBLAS and MKL on Ryzen 7700
- Efficiently parallelized with just 3 lines of OpenMP directives
- Targets x86 processors with AVX2 and FMA3 instructions (=all modern Intel Core and AMD Ryzen CPUs)
- Follows the [BLIS](https://github.com/flame/blis) design

## Installation
Install the following packages via `apt` if you are using a Debian-based Linux distribution
```bash
sudo apt-get install cmake build-essential python3-dev python3-pip libomp-dev
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
### Optional:
If you want to benchmark OpenBLAS, please, install OpenBLAS following the [installation guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide).


## Performance

Test enviroment:
- CPU: Ryzen 7 7700 8 Cores, 16 Threads
- RAM: 32GB DDR5 6000 MHz CL36
- OpenBLAS v.0.3.26
- Compiler: `gcc 11.4.0`
- Compiler flags: `-O3 -mno-avx512f -march=native`
- OS: Ubuntu 22.04.4 LTS

<p align="center">
  <img src="assets/perf_vs_openblas.png" alt="openblas" width="85%">
</p>

To benchmark the code, run
```bash
cmake -B build -S . -DOPENBLAS=OFF
cmake --build build
./build/benchmark MINSIZE MAXSIZE NPTS WARMUP
```
If not manually specified, default values are `MINSIZE=200`, `MAXSIZE=8000`, `NPTS=40`, `WARMUP=5`.

To benchmark OpenBLAS, run
```bash
cmake -B build -S . -DOPENBLAS=ON -DOPENBLAS_PATH=path/to/OpenBLAS/
cmake --build build
./build/benchmark MINSIZE MAXSIZE NPTS WARMUP
```
Or you can use
```bash
bash benchmark.sh /path/to/OpenBLAS
```
to benchmark both the code and OpenBLAS.

For the visualization of the results, simply run
```python
python plot_benchmark.py
```

## Tests
```bash
bash test.sh
```