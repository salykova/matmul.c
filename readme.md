# High-Performance Matrix Multiplication in C
<p align="center">
  <img src="assets/logo.jpeg" width="75%" alt="cpu-burn">
</p>

## Key Features
- Simple, portable and scalable C code
- Step by step [tutorial](https://salykova.github.io/matmul-cpu)
- Targets x86 processors with AVX and FMA instructions (=all modern Intel Core and AMD Ryzen CPUs)
- Faster than NumPy when fine-tuned for Ryzen 7700
- Efficiently parallelized with just 3 lines of OpenMP directives
- Follows the [BLIS](https://github.com/flame/blis) design
- Works for arbitrary matrix sizes
## Performance

<p align="center">
  <img src="assets/benchmark_mt.png" alt="cpu-burn" width="80%">
</p>


<p align="center">
  <img src="assets/benchmark_mt2.png" alt="cpu-burn" width="80%">
</p>
