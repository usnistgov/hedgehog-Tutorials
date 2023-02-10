
#ifndef HEDGEHOG_TUTORIALS_ABSTRACT_PRIORITY_QUEUE_TASK_H_
#define HEDGEHOG_TUTORIALS_ABSTRACT_PRIORITY_QUEUE_TASK_H_

#include "../implementor/multi_priority_queue_receivers.h"

template<size_t Separator, class ...AllTypes>
class AbstractPriorityQueueTask : public hh::AbstractTask<Separator, AllTypes...> {
 public:
  explicit AbstractPriorityQueueTask(std::string const &name, size_t const numberThreads, bool const automaticStart)
      : hh::AbstractTask<Separator, AllTypes...>(
      std::make_shared<hh::core::CoreTask<Separator, AllTypes...>>(
          this,
          name, numberThreads, automaticStart,
          std::make_shared<hh::core::implementor::DefaultSlot>(),
          std::make_shared<MPQR<Separator, AllTypes...>>(),
          std::make_shared<hh::tool::DME<Separator, AllTypes...>>(this),
          std::make_shared<hh::core::implementor::DefaultNotifier>(),
          std::make_shared<hh::tool::MDS<Separator, AllTypes...>>()
      )
  ) {}
};

#endif //HEDGEHOG_TUTORIALS_ABSTRACT_PRIORITY_QUEUE_TASK_H_
