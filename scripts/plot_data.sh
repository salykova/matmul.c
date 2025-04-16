#!/bin/bash

BENCH_DIR="benchmark_data"

rm -r $PWD/build
cmake -B $PWD/build -S $PWD -DNTHREADS=1
cmake --build $PWD/build -t plot_data
$PWD/build/plot_data ${BENCH_DIR}