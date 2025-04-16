#!/bin/bash

MINSIZE=200;
STEPSIZE=200;
NPTS=40;
WNITER=5;
NITER_START=1001;
NITER_END=5;
SAVEDIR="benchmark_data"

rm -r $PWD/build
cmake -B $PWD/build -S $PWD -DNTHREADS=${1} -DINTEL_PROC=${2}
cmake --build $PWD/build -t benchmark
$PWD/build/benchmark ${MINSIZE} ${STEPSIZE} ${NPTS} ${WNITER} ${NITER_START} ${NITER_END} ${SAVEDIR}