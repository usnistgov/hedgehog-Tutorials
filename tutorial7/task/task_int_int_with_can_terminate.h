
#ifndef NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_WITH_CAN_TERMINATE_H_
#define NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_WITH_CAN_TERMINATE_H_

#include <hedgehog/hedgehog.h>

class TaskIntIntWithCanTerminate : public hh::AbstractTask<1, int, int> {
 public:
  explicit TaskIntIntWithCanTerminate(std::string const &name) : hh::AbstractTask<1, int, int>(name) {}
  void execute(std::shared_ptr<int> data) override { this->addResult(data); }

  bool canTerminate() const override { return false;}
};

#endif //NEW_HEDGEHOG_TUTORIALS_TASK_INT_INT_WITH_CAN_TERMINATE_H_
