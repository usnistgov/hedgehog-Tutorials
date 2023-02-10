//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_MANAGER_H
#define HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_MANAGER_H
#include <hedgehog/hedgehog.h>
#include "../data/data_type.h"
#include "../data/unified_matrix_block_data.h"
#include "../state/partial_computation_state.h"

template<class Type>
class PartialComputationStateManager
    : public hh::StateManager<
        2,
        MatrixBlockData<Type, 'c', Order::Column>,
        UnifiedMatrixBlockData<Type, 'p'>,
        std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
                  std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>
    > {
 public:
  explicit PartialComputationStateManager(std::shared_ptr<PartialComputationState<Type>> const &state) :
      hh::StateManager<2,
                       MatrixBlockData<Type, 'c', Order::Column>,
                       UnifiedMatrixBlockData<Type, 'p'>,
                       std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>,
                                 std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>>(
          state, "Partial Computation State Manager") {}



  [[nodiscard]] bool canTerminate() const override {
    this->state()->lock();
    auto ret = std::dynamic_pointer_cast<PartialComputationState<Type>>(this->state())->isDone();
    this->state()->unlock();
    return ret;
  }
};
#endif //HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_MANAGER_H
