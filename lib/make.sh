#!/usr/bin/env bash


CUDA_PATH=/home/tengyao/opt/cuda-10.1

python3 setup.py build_ext --inplace
rm -rf build
