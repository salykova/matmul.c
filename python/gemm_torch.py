import time
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--minsize", type=int, default=200)
parser.add_argument("--stepsize", type=int, default=200)
parser.add_argument("--npts", type=int, default=40)
parser.add_argument("--wniter", type=int, default=5)
parser.add_argument("--niter_start", type=int, default=1001)
parser.add_argument("--niter_end", type=int, default=5)
parser.add_argument("--save_dir", type=str, default="benchmark_data")


def get_niter(matsize, niter_start, niter_end, matsize_start, matsize_end):
    if matsize_end == matsize_start or niter_start == niter_end:
        return niter_start
    a = float((niter_end - niter_start) * (matsize_start * matsize_end)) / (matsize_start - matsize_end)
    b = niter_start - a / matsize_start
    return round(a / matsize + b)


if __name__ == "__main__":
    args = parser.parse_args()
    dtype = torch.float32

    sep = 30
    if args.wniter:
        wmatsize = args.minsize + (args.npts // 2) * args.stepsize
        m = n = k = wmatsize

        A = torch.rand((m, k), dtype=dtype)
        B = torch.rand((k, n), dtype=dtype)
        C = torch.zeros((m, n), dtype=dtype)

        print(sep * "=")
        print("Warm-up".center(sep))
        print(sep * "=")

        for i in range(args.wniter):
            torch.matmul(A, B, out=C)
            print(f"m=n=k={wmatsize}: {i + 1}/{args.wniter}", end="\r")
        print("\n")

    gflops_all = []
    matsizes = [args.minsize + i * args.stepsize for i in range(args.npts)]

    print(sep * "=")
    print("Benchmark: PyTorch GEMM".center(sep))
    print(sep * "=")

    for matsize in matsizes:
        m = n = k = matsize
        A = torch.rand((m, k), dtype=dtype)
        B = torch.rand((k, n), dtype=dtype)
        C = torch.zeros((m, n), dtype=dtype)

        n_iter = get_niter(matsize, args.niter_start, args.niter_end, matsizes[0], matsizes[-1])
        st = time.perf_counter()
        for _ in range(n_iter):
            torch.matmul(A, B, out=C)
        et = time.perf_counter()

        elapsed_time_s = (et - st) / n_iter
        flop = 2 * m * n * k
        gflops = flop / elapsed_time_s / 1e9
        gflops_all.append(gflops)
        print(f"m=n=k={matsize} | GFLOPS = {gflops:1.0f}")

    save_dir_path = Path.cwd() / args.save_dir
    Path.mkdir(save_dir_path, exist_ok=True)
    bench_data_path = save_dir_path / "GEMM-PyTorch.txt"
    with open(bench_data_path, mode="w") as f:
        f.writelines([f"{matsize} {int(gflops)}\n" for (matsize, gflops) in zip(matsizes, gflops_all)])
