//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
#define HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
#include <hedgehog/hedgehog.h>
#include "../data/unified_matrix_block_data.h"

template <class Type>
class CudaProductTask : public hh::AbstractCUDATask<MatrixBlockData<Type, 'p', Order::Column>, std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>> {
 private:
  cublasHandle_t handle_ = {};
  Type *tempGpuResult_;
  size_t blockSizeHeight_;
  size_t blockSizeWidth_;

 public:
  explicit CudaProductTask(size_t blockSizeHeight, size_t blockSizeWidth, size_t numberThreads = 1) :
  hh::AbstractCUDATask<
      MatrixBlockData<Type, 'p', Order::Column>,
          std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>>("CUDA Product Task", numberThreads, false, true),
          blockSizeHeight_(blockSizeHeight), blockSizeWidth_(blockSizeWidth) {}

  void initializeCuda() override {
    checkCudaErrors(cublasCreate_v2(&handle_));
    checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    checkCudaErrors(cudaMallocManaged((void **)&tempGpuResult_, sizeof(Type) * blockSizeHeight_ * blockSizeWidth_));
  }

  void shutdownCuda() override {
    checkCudaErrors(cublasDestroy_v2(handle_));
    checkCudaErrors(cudaFree(tempGpuResult_));
  }

  void execute(std::shared_ptr<std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>> ptr) override {
    Type alpha = 1.,
         beta = 0.;

    auto matA = ptr->first;
    auto matB = ptr->second;
//    auto res = this->getManagedMemory();

//    res->rowIdx(matA->rowIdx());
//    res->colIdx(matB->colIdx());
//    res->blockSizeHeight(matA->blockSizeHeight());
//    res->blockSizeWidth(matB->blockSizeWidth());
//    res->leadingDimension(matA->blockSizeHeight());
//    res->ttl(1);

    checkCudaErrors(cudaMemPrefetchAsync(tempGpuResult_, sizeof(Type) * blockSizeHeight_ * blockSizeWidth_, this->deviceId(), this->stream()));

    Type *resultData = new Type[matA->blockSizeHeight() * matB->blockSizeWidth()];
    auto res = std::make_shared<MatrixBlockData<Type, 'p', Order::Column>>(matA->rowIdx(), matB->colIdx(), matA->blockSizeHeight(), matB->blockSizeWidth(), resultData, resultData);

    matA->synchronizeEvent();
    matB->synchronizeEvent();

    if constexpr(std::is_same<Type, float>::value) {
      checkCudaErrors(
          cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (float *) matA->blockData(), matA->leadingDimension(),
                         (float *) matB->blockData(), matB->leadingDimension(), &beta,
                         (float *) tempGpuResult_, blockSizeHeight_)
      );
    } else if (std::is_same<Type, double>::value) {
      checkCudaErrors(
          cublasDgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (double *) matA->blockData(), matA->leadingDimension(),
                         (double *) matB->blockData(), matB->leadingDimension(), &beta,
                         (double *) tempGpuResult_, blockSizeHeight_)
      );
    } else {
      std::cerr << "The matrix can't be multiplied" << std::endl;
      exit(43);
    }

    checkCudaErrors(cudaMemPrefetchAsync(tempGpuResult_, sizeof(Type) * blockSizeHeight_ * blockSizeWidth_, cudaCpuDeviceId, this->stream()));

    checkCudaErrors(cudaStreamSynchronize(this->stream()));

    matA->returnToMemoryManager();
    matB->returnToMemoryManager();

    std::copy(tempGpuResult_, tempGpuResult_ + blockSizeWidth_ * blockSizeHeight_, resultData);

    this->addResult(res);
  }

  std::shared_ptr<hh::AbstractTask <
                  MatrixBlockData<Type, 'p', Order::Column>, std::pair<std::shared_ptr<UnifiedMatrixBlockData<Type, 'a'>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'b'>>>>>

  copy() override {
    return std::make_shared<CudaProductTask>(blockSizeHeight_, blockSizeWidth_, this->numberThreads());
  }


};

#endif //HEDGEHOG_TUTORIALS_CUDA_PRODUCT_TASK_H
