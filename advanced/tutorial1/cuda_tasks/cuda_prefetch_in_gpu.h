//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_CUDA_PREFETCH_IN_GPU_H
#define HEDGEHOG_TUTORIALS_CUDA_PREFETCH_IN_GPU_H

#include "../data/unified_matrix_block_data.h"

template<class MatrixType, char Id>
class CudaPrefetchInGpu : public hh::AbstractCUDATask<1,
                                                      UnifiedMatrixBlockData<MatrixType, Id>,
                                                      UnifiedMatrixBlockData<MatrixType, Id>> {
 private:
  size_t blockTTL_ = 0;

 public:
  explicit CudaPrefetchInGpu(size_t blockTTL)
      : hh::AbstractCUDATask<1, UnifiedMatrixBlockData<MatrixType, Id>, UnifiedMatrixBlockData<MatrixType, Id>>(
      "Prefetch In GPU",
      1,
      false,
      false), blockTTL_(blockTTL) {}

  void execute(std::shared_ptr<UnifiedMatrixBlockData<MatrixType, Id> > ptr) override {
    auto managedMemory =
        std::dynamic_pointer_cast<UnifiedMatrixBlockData<MatrixType, Id>>(this->getManagedMemory());

    managedMemory->rowIdx(ptr->rowIdx());
    managedMemory->colIdx(ptr->colIdx());
    managedMemory->blockSizeHeight(ptr->blockSizeHeight());
    managedMemory->blockSizeWidth(ptr->blockSizeWidth());
    managedMemory->leadingDimension(ptr->leadingDimension());
    managedMemory->fullMatrixData(ptr->fullMatrixData());
    managedMemory->blockData(ptr->blockData());

    managedMemory->ttl(blockTTL_);

    checkCudaErrors(cudaMemPrefetchAsync(managedMemory->blockData(),
                                         sizeof(MatrixType) * managedMemory->blockSizeHeight()
                                             * managedMemory->blockSizeWidth(),
                                         this->deviceId(),
                                         this->stream()));

    managedMemory->recordEvent(this->stream());
    checkCudaErrors(cudaGetLastError());
    this->addResult(managedMemory);
  }

  std::shared_ptr<hh::AbstractTask<1, UnifiedMatrixBlockData<MatrixType, Id>, UnifiedMatrixBlockData<MatrixType, Id>>>
  copy() override {
    return std::make_shared<CudaPrefetchInGpu<MatrixType, Id>>(blockTTL_);
  }

};

#endif //HEDGEHOG_TUTORIALS_CUDA_PREFETCH_IN_GPU_H
