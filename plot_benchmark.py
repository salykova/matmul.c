import glob
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    try:
        blas_lib = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"].upper()
    except KeyError:
        blas_lib = "NumPy"

    benchmark_data = glob.glob("*.txt")
    assert (
        "benchmark_c.txt" in benchmark_data or "benchmark_numpy.txt" in benchmark_data
    ), "First, run benchmark.c and/or benchmark_numpy.py to create data"

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(10, 8))

    if "benchmark_c.txt" in benchmark_data:
        mat_sizes, min_gflops_c, max_gflops_c, avg_gflops_c = np.loadtxt("benchmark_c.txt").T
        plt.plot(mat_sizes, avg_gflops_c, "-*", label="matmul.c MEAN")
        plt.plot(mat_sizes, max_gflops_c, "-*", label="matmul.c PEAK")

    if "benchmark_numpy.txt" in benchmark_data:
        mat_sizes, min_gflops_numpy, max_gflops_numpy, avg_gflops_numpy = np.loadtxt("benchmark_numpy.txt").T
        plt.plot(mat_sizes, avg_gflops_numpy, "-*", label="NumPy MEAN")
        plt.plot(mat_sizes, max_gflops_numpy, "-*", label="NumPy PEAK")

    ax.set_xlabel("M=N=K", fontsize=16)
    ax.set_ylabel("GFLOP/S", fontsize=16)
    ax.set_title(f"NumPy ({blas_lib} backend) vs matmul.c", fontsize=18)
    ax.legend(fontsize=12)
    ax.grid()
    plt.show()
    fig.savefig("benchmark.png")
