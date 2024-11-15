#!/bin/bash

rm -r build
witer=5
npts=40
minsize=200
maxsize=8000

cmake -B build -S . -DOPENBLAS=OFF -DNTHREADS=${2}
cmake --build build
./build/benchmark ${minsize} ${maxsize} ${npts} ${witer}

cmake -B build -S . -DOPENBLAS=ON -DOPENBLAS_PATH=${1}
cmake --build build
./build/benchmark ${minsize} ${maxsize} ${npts} ${witer}

python plot_benchmark.py