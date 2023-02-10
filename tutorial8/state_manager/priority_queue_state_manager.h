
#ifndef HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_STATE_MANAGER_H_
#define HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_STATE_MANAGER_H_

#include "../implementor/multi_priority_queue_receivers.h"

template<size_t Separator, class ...AllTypes>
class PriorityQueueStateManager : public hh::StateManager<Separator, AllTypes...> {
 public:
  explicit PriorityQueueStateManager(std::shared_ptr<hh::AbstractState<Separator, AllTypes...>> const &state,
                                     std::string const &name, bool const automaticStart)
      : hh::StateManager<Separator, AllTypes...>(
      std::make_shared<hh::core::CoreStateManager<Separator, AllTypes...>>(
  this,
  state, name, automaticStart,
  std::make_shared<hh::core::implementor::DefaultSlot>(),
      std::make_shared<MPQR<Separator, AllTypes...>>(),
  std::make_shared<hh::tool::DME<Separator, AllTypes...>>(state.get()),
  std::make_shared<hh::core::implementor::DefaultNotifier>(),
      std::make_shared<hh::tool::MDS<Separator, AllTypes...>>()
  ), state
  ) {}
};

#endif //HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_STATE_MANAGER_H_
