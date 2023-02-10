
#ifndef NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_H_
#define NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_H_

#include <hedgehog/hedgehog.h>

class TaskIntInt : public hh::AbstractTask<1, int, int> {
 public:
  explicit TaskIntInt(std::string const &name) : hh::AbstractTask<1, int, int>(name) {}
  void execute(std::shared_ptr<int> data) override { this->addResult(data); }
};

#endif //NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_H_
