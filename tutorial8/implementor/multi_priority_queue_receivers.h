
#ifndef HEDGEHOG_TUTORIALS_MULTI_PRIORITY_QUEUE_RECEIVERS_H_
#define HEDGEHOG_TUTORIALS_MULTI_PRIORITY_QUEUE_RECEIVERS_H_

#include "priority_queue_receiver.h"

template<class ...Inputs>
class MultiPriorityQueueReceivers : public PriorityQueueReceiver<Inputs> ... {
 public:
  explicit MultiPriorityQueueReceivers() : PriorityQueueReceiver<Inputs>()... {}
  ~MultiPriorityQueueReceivers() override = default;
};

template<class Inputs>
struct MultiPriorityQueueReceiversTypeDeducer;

template<class ...Inputs>
struct MultiPriorityQueueReceiversTypeDeducer<std::tuple<Inputs...>> {
  using type = MultiPriorityQueueReceivers<Inputs...>;
};

template<class TupleInputs>
using MultiPriorityQueueReceiversTypeDeducer_t = typename MultiPriorityQueueReceiversTypeDeducer<TupleInputs>::type;

template<size_t Separator, class ...AllTypes>
using MPQR = MultiPriorityQueueReceiversTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

#endif //HEDGEHOG_TUTORIALS_MULTI_PRIORITY_QUEUE_RECEIVERS_H_
