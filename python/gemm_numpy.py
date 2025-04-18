import time
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--minsize", type=int, default=500)
parser.add_argument("--stepsize", type=int, default=500)
parser.add_argument("--npts", type=int, default=16)
parser.add_argument("--wniter", type=int, default=20)
parser.add_argument("--niter_start", type=int, default=1000)
parser.add_argument("--niter_end", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="benchmark_data")


def get_niter(matsize, niter_start, niter_end, matsize_start, matsize_end):
    if matsize_end == matsize_start or niter_start == niter_end:
        return niter_start
    a = float((niter_end - niter_start) * (matsize_start * matsize_end)) / (matsize_start - matsize_end)
    b = niter_start - a / matsize_start
    return round(a / matsize + b)


if __name__ == "__main__":
    args = parser.parse_args()
    dtype = np.float32

    sep = 30
    if args.wniter > 0:
        wmatsize = args.minsize + (args.npts // 2) * args.stepsize
        m = n = k = wmatsize
        A = np.random.randn(m, k).astype(dtype)
        B = np.random.randn(k, n).astype(dtype)
        C = np.zeros((k, n)).astype(dtype)

        print(sep * "=")
        print("Warm-up".center(sep))
        print(sep * "=")

        for i in range(args.wniter):
            np.matmul(A, B, out=C)
            print(f"m=n=k={wmatsize}: {i + 1}/{args.wniter}", end="\r")
        print("\n")

    gflops_all = []
    matsizes = [args.minsize + i * args.stepsize for i in range(args.npts)]

    print(sep * "=")
    print("Benchmark: NumPy GEMM".center(sep))
    print(sep * "=")

    for matsize in matsizes:
        m = n = k = matsize
        A = np.random.randn(m, k).astype(dtype)
        B = np.random.randn(k, n).astype(dtype)
        C = np.zeros((m, n)).astype(dtype)

        n_iter = get_niter(matsize, args.niter_start, args.niter_end, matsizes[0], matsizes[-1])
        st = time.perf_counter()
        for _ in range(n_iter):
            np.matmul(A, B, out=C)
        et = time.perf_counter()

        elapsed_time_s = (et - st) / n_iter
        flop = 2 * m * n * k
        gflops = flop / elapsed_time_s / 1e9
        gflops_all.append(gflops)
        print(f"m=n=k={matsize} | GFLOPS = {gflops:1.0f}")

    save_dir_path = Path.cwd() / args.save_dir
    Path.mkdir(save_dir_path, exist_ok=True)
    bench_data_path = save_dir_path / "GEMM-NumPy.txt"
    with open(bench_data_path, mode="w") as f:
        f.writelines([f"{matsize} {int(gflops)}\n" for (matsize, gflops) in zip(matsizes, gflops_all)])
