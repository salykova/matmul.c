#!/bin/bash

step=1
minsize=32
maxsize=512

rm -r build
cmake -B $PWD/build -S . -DOPENBLAS=OFF -DNTHREADS=${1}
cmake --build $PWD/build -t test_matmul
$PWD/build/test_matmul ${minsize} ${maxsize} ${step}