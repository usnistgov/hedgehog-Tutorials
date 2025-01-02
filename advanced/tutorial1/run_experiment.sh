#!/bin/bash

TUTORIAL_EXECUTABLE=/path/to/advanced_tutorial1

GPU_IDS=("-d 0 -d 1 -d 2 -d 3 -d 4 -d 5 -d 6 -d 7 -d 8" "-d 0 -d 1 -d 2 -d 3" "-d 0")

BLOCK_SIZES=(8192 16384)
MATRIX_SIZES=(131072 262144)

PRODUCT_THREADS=4
ADDITION_THREADS=100

IS_FIRST=1

for matrix_size in "${MATRIX_SIZES[@]}"; do
  for block_size in "${BLOCK_SIZES[@]}"; do
    # Varying number of GPUs
    for GPU_ID in "${GPU_IDS[@]}"; do
      echo "Running $matrix_size with block size $block_size and GPU IDS $GPU_ID"
      if [ $IS_FIRST -eq 1 ]; then
        IGNORE_HEADER=""
        IS_FIRST=0
      else
        IGNORE_HEADER="-z"
      fi

      $TUTORIAL_EXECUTABLE -n $matrix_size -m $matrix_size -p $matrix_size -b $block_size -x $PRODUCT_THREADS -a $ADDITION_THREADS $GPU_ID $IGNORE_HEADER >> results.txt
    done
  done
done