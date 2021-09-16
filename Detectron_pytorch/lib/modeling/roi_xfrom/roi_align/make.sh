#!/usr/bin/env bash

CUDA_PATH=/home/tengyao/opt/cuda-10.1

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_75

cd ../
python build.py
