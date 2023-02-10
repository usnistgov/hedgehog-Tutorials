//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
#define HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
#include <hedgehog/hedgehog.h>
#include "../data/unified_matrix_block_data.h"

template<class Type>
class CudaProductTask
    : public hh::AbstractCUDATask<
        1,
        std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>,
                  std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>, UnifiedMatrixBlockData<Type, 'p'>> {
 private:
  cublasHandle_t handle_ = {};

 public:
  explicit CudaProductTask(size_t numberThreads = 1) : hh::AbstractCUDATask<
      1,
      std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>,
                std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>, UnifiedMatrixBlockData<Type, 'p'>>(
      "CUDA Product Task",
      numberThreads,
      false,
      false) {}

  void initializeCuda() override {
    checkCudaErrors(cublasCreate_v2(&handle_));
    checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
  }

  void shutdownCuda() override {
    checkCudaErrors(cublasDestroy_v2(handle_));
  }

  void execute(std::shared_ptr<std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>,
                                         std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>> ptr) override {
    Type alpha = 1.,
        beta = 0.;

    auto matA = ptr->first;
    auto matB = ptr->second;
    auto res = std::dynamic_pointer_cast<UnifiedMatrixBlockData<Type, 'p'>>(this->getManagedMemory());

    res->rowIdx(matA->rowIdx());
    res->colIdx(matB->colIdx());
    res->blockSizeHeight(matA->blockSizeHeight());
    res->blockSizeWidth(matB->blockSizeWidth());
    res->leadingDimension(matA->blockSizeHeight());
    res->ttl(1);

    checkCudaErrors(cudaMemPrefetchAsync(res->blockData(),
                                         sizeof(Type) * res->blockSizeHeight() * res->blockSizeWidth(),
                                         this->deviceId(),
                                         this->stream()));

    matA->synchronizeEvent();
    matB->synchronizeEvent();

    if constexpr (std::is_same<Type, float>::value) {
      checkCudaErrors(
          cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (float *) matA->blockData(), matA->leadingDimension(),
                         (float *) matB->blockData(), matB->leadingDimension(), &beta,
                         (float *) res->blockData(), res->leadingDimension())
      );
    } else if (std::is_same<Type, double>::value) {
      checkCudaErrors(
          cublasDgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (double *) matA->blockData(), matA->leadingDimension(),
                         (double *) matB->blockData(), matB->leadingDimension(), &beta,
                         (double *) res->blockData(), res->leadingDimension())
      );
    } else {
      std::cerr << "The matrix can't be multiplied" << std::endl;
      exit(43);
    }

    checkCudaErrors(cudaStreamSynchronize(this->stream()));

    matA->returnToMemoryManager();
    matB->returnToMemoryManager();

    cudaMemPrefetchAsync(res->blockData(),
                         sizeof(Type) * res->blockSizeHeight() * res->blockSizeWidth(),
                         cudaCpuDeviceId,
                         this->stream());

    res->recordEvent(this->stream());

    this->addResult(res);
  }

  std::shared_ptr<hh::AbstractTask<
      1,
      std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>,
                std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>, UnifiedMatrixBlockData<Type, 'p'>>>
  copy() override {
    return std::make_shared<CudaProductTask>(this->numberThreads());
  }

};

#endif //HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
