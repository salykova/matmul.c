#!/bin/bash

rm -r build
step=1
minsize=32
maxsize=1024

cmake -B build -S . -DOPENBLAS=OFF
cmake --build build
./build/test ${minsize} ${maxsize} ${step}

# cmake -B build -S . -DOPENBLAS=ON -DOPENBLAS_PATH=${1}
# cmake --build build
# ./build/test ${minsize} ${maxsize} ${step}