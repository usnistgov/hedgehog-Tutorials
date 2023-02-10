
#ifndef HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_RECEIVER_H_
#define HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_RECEIVER_H_

#include <hedgehog/hedgehog.h>

template<class Input>
class PriorityQueueReceiver : public hh::core::implementor::ImplementorReceiver<Input> {
 private:
  std::unique_ptr<std::priority_queue<std::shared_ptr<Input>>> const
      queue_ = nullptr; ///< Queue storing to be processed data

  std::unique_ptr<std::set<hh::core::abstraction::SenderAbstraction<Input> *>> const
      senders_ = nullptr; ///< List of senders attached to this receiver

  size_t
      maxSize_ = 0; ///< Maximum size attained by the queue

 public:
  explicit PriorityQueueReceiver()
      : queue_(std::make_unique<std::priority_queue<std::shared_ptr<Input>>>()),
  senders_(std::make_unique<std::set<hh::core::abstraction::SenderAbstraction<Input> *>>()) {}
  virtual ~PriorityQueueReceiver() = default;
  void receive(std::shared_ptr<Input> const data) final {
    queue_->push(data);
    maxSize_ = std::max(queue_->size(), maxSize_);
  }
  [[nodiscard]] std::shared_ptr<Input> getInputData() override {
    assert(!queue_->empty());
    auto front = queue_->top();
    queue_->pop();
    return front;
  }
  [[nodiscard]] size_t numberElementsReceived() const override { return queue_->size(); }
  [[nodiscard]] size_t maxNumberElementsReceived() const override { return maxSize_; }
  [[nodiscard]] bool empty() const override { return queue_->empty(); }
  [[nodiscard]] std::set<hh::core::abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override {
    return *senders_;
  }
  void addSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override { senders_->insert(sender); }
  void removeSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override { senders_->erase(sender); }
};

#endif //HEDGEHOG_TUTORIALS_PRIORITY_QUEUE_RECEIVER_H_
