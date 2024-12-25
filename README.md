# High-Performance FP32 Matrix Multiplication on CPU

> **Important note:** Please don’t expect peak performance without fine-tuning hyperparameters such as the *number of threads, kernel size and block sizes*, unless you're running it on a Ryzen 9700X. Current parallelization strategy is optimized for Intel and AMD x86 desktop CPUs. For many-core server processors, consider using nested parallelism and parallelizing 2-3 loops to increase the performance (e.g., the 5th, 3rd, and 2nd loops around the kernel).

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
To benchmark OpenBLAS, start by installing it according to the [installation guide](https://github.com/OpenMathLib/OpenBLAS/wiki/Installation-Guide). During the installation, ensure you set an appropriate TARGET and disable AVX512 instructions. For instance, if you're using Zen4/5 CPUs, compile OpenBLAS with:
```bash
make TARGET=ZEN
```
Otherwise, OpenBLAS defaults to AVX512 instructions available on Zen4/5 CPUs.


## Performance

Test enviroment:
- CPU: AMD Ryzen 7 9700X | Ryzen 9 7900X
- Locked CPU clock frequency: 4.5GHz
- RAM: 32GB DDR5 6000 MHz CL36
- OpenBLAS v.0.3.26
- Compiler: GCC 13.3.0
- OS: Ubuntu Ubuntu 24.04.1 LTS

<p align="center">
  <img src="assets/perf_ryzen_9700x.png" alt="openblas" width="85%">
</p>

<p align="center">
  <img src="assets/perf_ryzen_7900x.png" alt="openblas" width="85%">
</p>
First, lock CPU clock frequency:

```bash
sudo cpupower frequency-set -u CLK
sudo cpupower frequency-set -d CLK
```
Ensure that clock frequency is stable and doesn't vary during the benchmark. You can check the CPU clock frequency by running the command
```bash
watch -n 1 grep \"cpu MHz\" /proc/cpuinfo
```

To benchmark the matmul implementation, run
```bash
cmake -B build -S . -DOPENBLAS=OFF -DNTHREADS=X
cmake --build build
./build/benchmark MINSIZE MAXSIZE NPTS WARMUP
```
and set `-DNTHREADS` according to your CPU. On Ryzen 9700X I use `-DNTHREADS=16`. If not manually specified, default benchmark parameters are `MINSIZE=200`, `MAXSIZE=8000`, `NPTS=40`, `WARMUP=5`.

To benchmark OpenBLAS, run
```bash
cmake -B build -S . -DOPENBLAS=ON -DOPENBLAS_PATH=path/to/OpenBLAS/
cmake --build build
./build/benchmark MINSIZE MAXSIZE NPTS WARMUP
```
Or you can use
```bash
bash benchmark.sh /path/to/OpenBLAS NTHREADS
```
to benchmark both the code and OpenBLAS.

For the visualization of the results, simply run
```python
python plot_benchmark.py
```

## Tests
```bash
bash test.sh NTHREADS
```