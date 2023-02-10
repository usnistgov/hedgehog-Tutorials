#ifndef HEDGEHOG_TUTORIALS_REDUCTION_TASK_H_
#define HEDGEHOG_TUTORIALS_REDUCTION_TASK_H_

#include <hedgehog/hedgehog.h>
#include <functional>

template<class DataType>
class ReductionTask : public hh::AbstractTask<
    1,
    std::pair<typename std::vector<DataType>::const_iterator, typename std::vector<DataType>::const_iterator>,
    DataType> {
 private:
  std::function<DataType(DataType const &, DataType const &)> const lambda_;
 public:
  ReductionTask(size_t const numberThreads, std::function<DataType(DataType const &, DataType const &)> lambda)
      : hh::AbstractTask<
      1,
      std::pair<typename std::vector<DataType>::const_iterator, typename std::vector<DataType>::const_iterator>,
      DataType
  >("Reduction task", numberThreads), lambda_(std::move(lambda)) {}

  void execute(std::shared_ptr<std::pair<typename std::vector<DataType>::const_iterator,
                                         typename std::vector<DataType>::const_iterator>> data) override {
    if (data->first == data->second) { this->addResult(std::make_shared<DataType>()); }
    else if (data->second - data->first == 1) { this->addResult(std::make_shared<DataType>(*(data->first))); }
    else {

      this->addResult(std::make_shared<DataType>(std::reduce(data->first + 1,
                                                             data->second,
                                                             *(data->first),
                                                             lambda_)));
    }
  }

  std::shared_ptr<hh::AbstractTask<1,
                                   std::pair<typename std::vector<DataType>::const_iterator,
                                             typename std::vector<DataType>::const_iterator>,
                                   DataType>> copy() override {
    return std::make_shared<ReductionTask<DataType>>(this->numberThreads(), lambda_);
  }
};

#endif //HEDGEHOG_TUTORIALS_REDUCTION_TASK_H_
