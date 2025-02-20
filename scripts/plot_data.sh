#!/bin/bash

rm -r $PWD/build
cmake -B $PWD/build -S $PWD
cmake --build $PWD/build -t plot_data
$PWD/build/plot_data