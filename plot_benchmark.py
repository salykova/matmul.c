import glob
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    benchmark_data = glob.glob("*.txt")
    assert (
        "benchmark_matmul.txt" in benchmark_data or "benchmark_openblas.txt" in benchmark_data
    ), "No benchmark data was found, first run benchmark!"

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(10, 8))

    if "benchmark_openblas.txt" in benchmark_data:
        mat_sizes, min_gflops_c, max_gflops_c, avg_gflops_c = np.loadtxt("benchmark_openblas.txt").T
        plt.plot(mat_sizes, max_gflops_c, linewidth=2, label="cblas_sgemm C API")

    if "benchmark_matmul.txt" in benchmark_data:
        mat_sizes, min_gflops_numpy, max_gflops_numpy, avg_gflops_numpy = np.loadtxt("benchmark_matmul.txt").T
        plt.plot(mat_sizes, max_gflops_numpy, linewidth=2, label="matmul.c")

    ax.set_xlabel("M=N=K", fontsize=16)
    ax.set_ylabel("GFLOPS", fontsize=16)
    ax.set_title("OpenBLAS v0.3.26 vs matmul.c (Ryzen 7700 CLK LOCK=4.5GHz)", fontsize=19)
    ax.legend(fontsize=16)
    ax.grid()
    plt.show()
    fig.savefig("benchmark.png")
