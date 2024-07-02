import glob
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    benchmark_data = glob.glob("*.txt")
    assert (
        "benchmark_c.txt" in benchmark_data or "benchmark_numpy.txt" in benchmark_data
    ), "First, run benchmark.c and/or benchmark_numpy.py to create data"

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(10, 8))

    if "benchmark_c.txt" in benchmark_data:
        mat_sizes, min_gflops_c, max_gflops_c, avg_gflops_c = np.loadtxt("benchmark_c.txt").T
        plt.plot(mat_sizes, avg_gflops_c, "-*", label="matmul.c AVG")
        plt.plot(mat_sizes, max_gflops_c, "-*", label="matmul.c PEAK")
        # plt.plot(mat_sizes, min_gflops_c, "-*", label="C MIN")
        # ax.fill_between(mat_sizes, min_gflops_c, max_gflops_c, alpha=0.2)

    ax.set_xlabel("M=N=K", fontsize=16)
    ax.set_ylabel("GFLOP/S", fontsize=16)
    ax.set_title("matmul_llama.c, Ryzen 7700 (8C, 16T)", fontsize=18)
    ax.legend(fontsize=12)
    ax.grid()
    plt.show()
    fig.savefig("benchmark.png")
