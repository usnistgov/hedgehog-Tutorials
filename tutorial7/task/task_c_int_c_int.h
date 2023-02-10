
#ifndef NEW_HEDGEHOG_TUTORIALS_TASK_C_INT_C_INT_H_
#define NEW_HEDGEHOG_TUTORIALS_TASK_C_INT_C_INT_H_

#include <hedgehog/hedgehog.h>

class TaskCIntCInt : public hh::AbstractTask<1, int const, int const> {
 public:
  explicit TaskCIntCInt(std::string const &name) : hh::AbstractTask<1, int const, int const>(name) {}
  void execute(std::shared_ptr<int const> data) override { this->addResult(data); }
};

#endif //NEW_HEDGEHOG_TUTORIALS_TASK_C_INT_C_INT_H_
