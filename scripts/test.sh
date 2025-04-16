#!/bin/bash

step=1
minsize=4
maxsize=512

rm -r build
cmake -B $PWD/build -S . -DNTHREADS=${1} -DINTEL_PROC=${2}
cmake --build $PWD/build -t test_matmul
$PWD/build/test_matmul ${minsize} ${maxsize} ${step}