#!/bin/bash

rm -r $PWD/build

MINSIZE=200;
STEPSIZE=200;
NPTS=40;
WNITER=5;
NITER_START=1001;
NITER_END=5;

cmake -B $PWD/build -S $PWD -DOPENBLAS=OFF -DNTHREADS=${1}
cmake --build $PWD/build -t benchmark
$PWD/build/benchmark ${MINSIZE} ${STEPSIZE} ${NPTS} ${WNITER} ${NITER_START} ${NITER_END}

python plot_benchmark.py