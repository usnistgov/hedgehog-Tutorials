//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_ADDITION_TASK_H
#define HEDGEHOG_TUTORIALS_ADDITION_TASK_H

#include <hedgehog/hedgehog.h>

#include "../data/unified_matrix_block_data.h"

template<class Type>
class AdditionTask : public hh::AbstractTask<
    1, std::pair<
        std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
        std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>,
    MatrixBlockData<Type, 'c', Order::Column>> {

 public:
  explicit AdditionTask(size_t numberThreads) :
      hh::AbstractTask<
          1, std::pair<
              std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
              std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>,
          MatrixBlockData<Type, 'c', Order::Column>>("Addition Task", numberThreads) {}

  virtual ~AdditionTask() = default;

 public:
  void execute(std::shared_ptr<std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
                                         std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>> ptr) override {
    auto c = ptr->first;
    auto p = ptr->second;

    p->synchronizeEvent();

    assert(c->blockSizeWidth() == p->blockSizeWidth());
    assert(c->blockSizeHeight() == p->blockSizeHeight());

    for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
      for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
        c->blockData()[j * c->leadingDimension() + i] += p->blockData()[j * p->leadingDimension() + i];
      }
    }

    p->returnToMemoryManager();

    this->addResult(c);
  }

  std::shared_ptr<hh::AbstractTask<1,
                                   std::pair<
                                       std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
                                       std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>,
                                   MatrixBlockData<Type, 'c', Order::Column>>> copy() override {
    return std::make_shared<AdditionTask>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TUTORIALS_ADDITION_TASK_H
