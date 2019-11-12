//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_CUDA_UTILS_H
#define HEDGEHOG_TUTORIALS_CUDA_UTILS_H


#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

// Utils macro to convert row based intex to column based
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, [[maybe_unused]]const char *file, [[maybe_unused]]const int line) {
  if (cudaSuccess != err) {
    std::cerr << "checkCudaErrors() Cuda error = "
              << err
              << "\"" << cudaGetErrorString(err) << " \" from "
              << file << ":" << line << std::endl;
    cudaDeviceReset();
    exit(43);
  }
}
// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cublasStatus_t err, const char *file, const int line) {
  if (CUBLAS_STATUS_SUCCESS != err) {
    std::cerr << "checkCudaErrors() Status Error = "
              << err << " from "
              << file << ":" << line << std::endl;
    cudaDeviceReset();
    exit(44);
  }
}
#endif


#endif //HEDGEHOG_TUTORIALS_CUDA_UTILS_H
