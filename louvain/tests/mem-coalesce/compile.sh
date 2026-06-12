#!/bin/bash
# sm_86 = RTX 3050/3090 (Ampere); use sm_75 for the Colab T4.
nvcc -O3 -arch=sm_86 mem_coalesce.cu            -o mem_coalesce
nvcc -O3 -arch=sm_86 mem_coalesce_fat.cu        -o mem_coalesce_fat
nvcc -O3 -arch=sm_86 mem_coalesce_fat_sparse.cu -o mem_coalesce_fat_sparse
