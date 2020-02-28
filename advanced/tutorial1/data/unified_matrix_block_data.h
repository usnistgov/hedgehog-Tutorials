//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_UNIFIED_MATRIX_BLOCK_DATA_H
#define HEDGEHOG_TUTORIALS_UNIFIED_MATRIX_BLOCK_DATA_H

#include <hedgehog/hedgehog.h>
#include "matrix_block_data.h"
template<class Type, char Id>
class UnifiedMatrixBlockData : public MatrixBlockData<Type, Id, Order::Column>,
                               public hh::MemoryData<UnifiedMatrixBlockData<Type, Id>> {
 private:
  size_t
      ttl_ = 0;

  cudaEvent_t
      event_ = {};

  bool releaseMemory_ = false;
  bool eventCreated_ = false;

 public:
  UnifiedMatrixBlockData() : releaseMemory_(false){}

  explicit UnifiedMatrixBlockData(size_t blockSize) : releaseMemory_(true) {
    checkCudaErrors(cudaMallocManaged((void **) this->adressBlockData(), sizeof(Type) * blockSize * blockSize));
  }

  UnifiedMatrixBlockData(
      size_t rowIdx, size_t colIdx,
      size_t blockSizeHeight, size_t blockSizeWidth, size_t leadingDimension,
  Type *fullMatrixData, Type *blockData)
  : MatrixBlockData<Type, Id, Order::Column>(
      rowIdx, colIdx,
      blockSizeHeight, blockSizeWidth, leadingDimension,
      fullMatrixData, blockData) {}

  template<char OldId>
  explicit UnifiedMatrixBlockData(UnifiedMatrixBlockData<Type, OldId> &o) {
    this->rowIdx_ = o.rowIdx_;
    this->colIdx_ = o.colIdx_;
    this->blockSizeHeight_ = o.blockSizeHeight_;
    this->blockSizeWidth_ = o.blockSizeWidth_;
    this->leadingDimension_ = o.leadingDimension_;
    this->fullMatrixData_ = o.fullMatrixData_;
    this->blockData_ = o.blockData_;
  }

  template<char OldId>
  explicit UnifiedMatrixBlockData(std::shared_ptr<UnifiedMatrixBlockData<Type, OldId>> &o) {
    this->rowIdx_ = o->getRowIdx();
    this->colIdx_ = o->getColIdx();
    this->blockSizeHeight_ = o->getBlockSizeHeight();
    this->blockSizeWidth_ = o->getBlockSizeWidth();
    this->leadingDimension_ = o->getLeadingDimension();
    this->fullMatrixData_ = o->getFullMatrixData();
    this->blockData_ = o->getBlockData();
  }

  virtual ~UnifiedMatrixBlockData() {
    if (releaseMemory_) {
      checkCudaErrors(cudaFree(this->blockData()));
    }

    if (eventCreated_) {
      checkCudaErrors(cudaEventDestroy(event_));
    }
  }

  Type **adressBlockData() { return &this->blockData_; }

  void recordEvent(cudaStream_t stream) {
    if (!eventCreated_) {
      checkCudaErrors(cudaEventCreate(&event_));
      eventCreated_ = true;
    }
    checkCudaErrors(cudaEventRecord(event_, stream));
  }

  void synchronizeEvent() {
    checkCudaErrors(cudaEventSynchronize(event_));
  }


  void ttl(size_t ttl) { ttl_ = ttl; }
  void used() override { --this->ttl_; }
  bool canBeRecycled() override { return this->ttl_ == 0; }

  friend std::ostream &operator<<(std::ostream &os, UnifiedMatrixBlockData const &data) {
    os << "UnifiedMatrixBlockData " << Id << " position Grid: (" << data.rowIdx_ << ", " << data.colIdx_ << ")"
       << std::endl;
    os << "Block: (" << data.blockSizeHeight() << ", " << data.blockSizeWidth() << ") leadingDimension="
       << data.leadingDimension_
       << " ttl: " << data.ttl_ << std::endl;
    return os;
  }
};


#endif //HEDGEHOG_TUTORIALS_UNIFIED_MATRIX_BLOCK_DATA_H
