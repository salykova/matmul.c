import os
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-MINSIZE", "--MIN-SIZE", type=int, default=200)
parser.add_argument("-MAXSIZE", "--MAX-SIZE", type=int, default=5000)
parser.add_argument("-WARMUP", "--WARMUP", type=int, default=15)
parser.add_argument("-NPTS", "--NUM-PTS", type=int, default=50)
parser.add_argument("-ST", "--SINGLE-THREAD", action="store_true")
parser.add_argument("-SHOWFIG", "--SHOW-FIG", action="store_true")
parser.add_argument("-SAVEFIG", "--SAVE-FIG", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    MAX_SIZE = args.MAX_SIZE
    MIN_SIZE = args.MIN_SIZE
    NUM_PTS = args.NUM_PTS
    SHOW_FIG = args.SHOW_FIG
    SAVE_FIG = args.SAVE_FIG
    if args.SINGLE_THREAD:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import matplotlib.pyplot as plt

    MAT_SIZES = np.linspace(MIN_SIZE, MAX_SIZE, args.NUM_PTS, endpoint=True, dtype=int)
    avg_flops = []
    max_flops = []
    min_flops = []

    # Warmup
    A = np.random.randn(MAX_SIZE, MAX_SIZE).astype(np.float32)
    B = np.random.randn(MAX_SIZE, MAX_SIZE).astype(np.float32)
    for _ in tqdm(range(args.WARMUP), desc="Warmup"):
        C = A @ B

    for i in tqdm(range(len(MAT_SIZES)), desc="Benchmark"):
        FLOP = 2 * MAT_SIZES[i] ** 3
        avg_exec_time = 0
        min_exec_time = np.inf
        max_exec_time = -np.inf
        A = np.random.randn(MAT_SIZES[i], MAT_SIZES[i]).astype(np.float32)
        B = np.random.randn(MAT_SIZES[i], MAT_SIZES[i]).astype(np.float32)
        n_iter = int(200_000 / MAT_SIZES[i])
        for _ in range(n_iter):
            start = time.perf_counter()
            C = A @ B
            end = time.perf_counter()
            exec_time = end - start
            min_exec_time = exec_time if exec_time < min_exec_time else min_exec_time
            max_exec_time = exec_time if exec_time > max_exec_time else max_exec_time
            avg_exec_time += exec_time

        avg_exec_time /= n_iter
        avg_flops.append(FLOP / avg_exec_time)
        max_flops.append(FLOP / min_exec_time)
        min_flops.append(FLOP / max_exec_time)

    avg_gflops = (np.array(avg_flops) / 1e9).astype(int)
    max_gflops = (np.array(max_flops) / 1e9).astype(int)
    min_gflops = (np.array(min_flops) / 1e9).astype(int)
    np.savetxt(
        "benchmark_numpy.txt",
        np.vstack((MAT_SIZES, min_gflops, max_gflops, avg_gflops)).T,
        fmt="%i",
    )

    if SAVE_FIG or SHOW_FIG:
        plt.rc("font", size=12)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(MAT_SIZES, avg_gflops, "--*", label="MEAN")
        plt.plot(MAT_SIZES, max_gflops, "--*", label="PEAK")
        ax.set_xlabel("M=N=K", fontsize=16)
        ax.set_ylabel("GFLOP/S", fontsize=16)
        title_threading = "SINGLE-THREADED" if args.SINGLE_THREAD else "MUTLI-THREADED"
        ax.set_title(f"NumPy(=OpenBLAS) {title_threading}, RYZEN 7700 (8C/16T)", fontsize=18)
        ax.legend(fontsize=12)
        ax.grid()
        if args.SHOW_FIG:
            plt.show()
        if args.SAVE_FIG:
            fig.savefig("benchmark_numpy.png")
