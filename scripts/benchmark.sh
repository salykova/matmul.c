#!/bin/bash

MINSIZE=200;
STEPSIZE=200;
NPTS=50;
WNITER=5;
NITER_START=1001;
NITER_END=5;

rm -r $PWD/build
cmake -B $PWD/build -S $PWD -DOPENBLAS=OFF -DNTHREADS=${1}
cmake --build $PWD/build -t benchmark
$PWD/build/benchmark ${MINSIZE} ${STEPSIZE} ${NPTS} ${WNITER} ${NITER_START} ${NITER_END}