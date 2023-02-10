//
// Created by anb22 on 9/26/22.
//

#ifndef HEDGEHOG_TUTORIALS_FINAL_REDUCTION_TASK_H_
#define HEDGEHOG_TUTORIALS_FINAL_REDUCTION_TASK_H_

#include <hedgehog/hedgehog.h>
#include "functional"

template<class DataType>
class VectorDecomposition;

template<class DataType>
class FinalReductionTask : public hh::AbstractTask<1, DataType, DataType> {
 private:
  friend VectorDecomposition<DataType>;
  std::shared_ptr<DataType> stored_ = nullptr;
  std::function<DataType(DataType const &, DataType const &)> const lambda_;
  std::size_t numberReductedBlocks_ = 0;
 public:
  explicit FinalReductionTask(std::function<DataType(DataType const &, DataType const &)> lambda) :
      hh::AbstractTask<1, DataType, DataType>("Final Reduction"), lambda_(std::move(lambda)) {}
  ~FinalReductionTask() override = default;

  void execute(std::shared_ptr<DataType> data) override {
    --numberReductedBlocks_;
    if (stored_ == nullptr) { stored_ = data; }
    else { *stored_ = lambda_(*data, *stored_); };
    if(numberReductedBlocks_ == 0) { this->addResult(stored_); }
  }

  void clean() override {
    stored_ = nullptr;
    numberReductedBlocks_ = 0;
  }

 private:
  void numberReductedBlocks(size_t numberReductedBlocks) {
    numberReductedBlocks_ = numberReductedBlocks;
  }

};

#endif //HEDGEHOG_TUTORIALS_FINAL_REDUCTION_TASK_H_
