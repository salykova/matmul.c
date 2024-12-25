import glob
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    cpu_model = "Unknown CPU"
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                cpu_model = line[line.find(":")+1:].strip()

    benchmark_data = glob.glob("*.txt")
    assert (
        "benchmark_matmul.txt" in benchmark_data or "benchmark_openblas.txt" in benchmark_data
    ), "No benchmark data was found. Please run the benchmark first!"

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(10, 9))
    if "benchmark_matmul.txt" in benchmark_data:
        mat_sizes, _, max_gflops_matmul, _ = np.loadtxt("benchmark_matmul.txt").T
        plt.plot(mat_sizes, max_gflops_matmul, linewidth=2, label="matmul.c")

    if "benchmark_openblas.txt" in benchmark_data:
        mat_sizes, _, max_gflops_openblas, _ = np.loadtxt("benchmark_openblas.txt").T
        plt.plot(mat_sizes, max_gflops_openblas, linewidth=2, label="OpenBLAS v0.3.26")


    ax.set_xlabel("M=N=K", fontsize=16)
    ax.set_ylabel("GFLOPS", fontsize=16)
    ax.set_title(cpu_model, fontsize=19)
    ax.legend(fontsize=16)
    ax.grid()
    plt.show()
    fig.savefig("benchmark.png")
