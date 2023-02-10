#ifndef HEDGEHOG_TUTORIALS_VECTOR_DECOMPOSITION_H_
#define HEDGEHOG_TUTORIALS_VECTOR_DECOMPOSITION_H_

#include <hedgehog/hedgehog.h>

#include "final_reduction_task.h"

template<class DataType>
class VectorDecomposition :
    public hh::AbstractTask<1,
                            std::vector<DataType>,
                            std::pair<typename std::vector<DataType>::const_iterator,
                                      typename std::vector<DataType>::const_iterator>> {
 private:
  size_t const blockSize_ = 0;
  FinalReductionTask<DataType> *const finalReductionType_ = nullptr;
 public:
  explicit VectorDecomposition(std::size_t const blockSize, FinalReductionTask<DataType> *finalReductionType) :
      hh::AbstractTask<1,
                       std::vector<DataType>,
                       std::pair<typename std::vector<DataType>::const_iterator,
                                 typename std::vector<DataType>::const_iterator>>("Vector Decomposition"),
      blockSize_(blockSize), finalReductionType_(finalReductionType) {
    assert(finalReductionType_ != nullptr);
  }

  ~VectorDecomposition() override = default;

  void execute(std::shared_ptr<std::vector<DataType>> v) override {
    if (v->size() < blockSize_) {
      throw std::runtime_error("The number of blocks is larger than the given vector of data.");
    }
    if (v->empty()) {
      finalReductionType_->numberReductedBlocks(1);
      this->addResult(std::make_shared<std::pair<typename std::vector<DataType>::const_iterator,
                                                 typename std::vector<DataType>::const_iterator>>(v->cbegin(),
                                                                                                  v->cend()));
    } else {
      finalReductionType_->numberReductedBlocks((size_t) std::ceil((double) v->size() / (double) (blockSize_)));
      auto begin = v->begin();
      auto endIt = v->begin();
      long endPosition = 0;
      while (endIt != v->end()) {
        endPosition = (long) std::min((size_t) endPosition + blockSize_, v->size());
        endIt = v->begin() + endPosition;
        this->addResult(std::make_shared<std::pair<typename std::vector<DataType>::const_iterator,
                                                   typename std::vector<DataType>::const_iterator>>(begin, endIt));
        begin = endIt;
      }
    }
  }
};

#endif //HEDGEHOG_TUTORIALS_VECTOR_DECOMPOSITION_H_
