#include <utility>

#include "task/abstract_priority_queue_task.h"
#include "state_manager/priority_queue_state_manager.h"

class IntIntPriorityQueueTask : public AbstractPriorityQueueTask<1, int, int> {
 public:
  IntIntPriorityQueueTask() : AbstractPriorityQueueTask("IntPriorityQueue", 1, false) {}
  void execute(std::shared_ptr<int> data) override {
    std::cout << *data << std::endl;
    this->addResult(data);
  }
};

class IntState : public hh::AbstractState<1, int, int> {
 public:
  void execute(std::shared_ptr<int> data) override {
    std::cout << *data << std::endl;
    this->addResult(data);
  }
};

int main() {
  {
    hh::Graph<1, int, int> g;
    auto task = std::make_shared<IntIntPriorityQueueTask>();
    g.inputs(task);
    g.outputs(task);

    for (int i = 100; i >= 0; --i) { g.pushData(std::make_shared<int>(i)); }

    g.executeGraph();
    g.finishPushingData();
    g.waitForTermination();
  }

  {
    hh::Graph<1, int, int> g;

    auto sm =
        std::make_shared<PriorityQueueStateManager<1, int, int>>(
            std::make_shared<IntState>(),
            "Priority State Manager",
            false);

    g.inputs(sm);
    g.outputs(sm);

    for (int i = 100; i >= 0; --i) { g.pushData(std::make_shared<int>(i)); }

    g.executeGraph();
    g.finishPushingData();
    g.waitForTermination();
  }
}