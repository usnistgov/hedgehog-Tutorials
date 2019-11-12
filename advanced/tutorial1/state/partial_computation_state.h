//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_H
#define HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_H
#include <hedgehog/hedgehog.h>
#include <ostream>
#include "../data/unified_matrix_block_data.h"

template<class Type>
class PartialComputationState
    : public hh::AbstractState<
        std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>,
        MatrixBlockData<Type, 'c', Order::Column>, UnifiedMatrixBlockData<Type, 'p'>
    > {
 private:
  size_t
      gridHeightResults_ = 0,
      gridWidthResults_ = 0;

  std::vector<std::vector<std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>>
      gridPartialProduct_ = {};

  std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>>
      gridMatrixC_ = {};

  size_t
      ttl_ = 0;

 public:
  PartialComputationState(size_t gridHeightResults, size_t gridWidthResults, size_t ttl)
      : gridHeightResults_(gridHeightResults), gridWidthResults_(gridWidthResults), ttl_(ttl) {
    gridPartialProduct_ =
        std::vector<std::vector<std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>>(
            gridHeightResults_ * gridWidthResults_);
    gridMatrixC_ =
        std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>>(
            gridHeightResults_ * gridWidthResults_,
            nullptr
        );
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();
    if (isPAvailable(i, j)) {
      auto res = std::make_shared<
          std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>
      >();
      res->first = ptr;
      res->second = partialProduct(i, j);
      this->push(res);
      --ttl_;
    } else {
      blockMatrixC(ptr);
    }
  }

  void execute(std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();
    if (isCAvailable(i, j)) {
      auto res = std::make_shared<
          std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>, std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>>>
      >();
      res->first = blockMatrixC(i, j);
      res->second = ptr;
      this->push(res);
      --ttl_;
    } else {
      partialProduct(ptr);
    }
  }

  bool isDone() { return ttl_ == 0; };

 private:
  bool isPAvailable(size_t i, size_t j) { return gridPartialProduct_[i * gridWidthResults_ + j].size() != 0; }
  bool isCAvailable(size_t i, size_t j) { return gridMatrixC_[i * gridWidthResults_ + j] != nullptr; }

  std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>> partialProduct(size_t i, size_t j) {
    assert(isPAvailable(i, j));
    std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>> p = gridPartialProduct_[i * gridWidthResults_ + j].back();
    gridPartialProduct_[i * gridWidthResults_ + j].pop_back();
    return p;
  }

  void partialProduct(std::shared_ptr<UnifiedMatrixBlockData<Type, 'p'>> p) {
    gridPartialProduct_[p->rowIdx() * gridWidthResults_ + p->colIdx()].push_back(p);
  }

  std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>> blockMatrixC(size_t i, size_t j) {
    assert(isCAvailable(i, j));
    auto c = gridMatrixC_[i * gridWidthResults_ + j];
    gridMatrixC_[i * gridWidthResults_ + j] = nullptr;
    return c;
  }

  void blockMatrixC(std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>> c) {
    assert(!isCAvailable(c->rowIdx(), c->colIdx()));
    gridMatrixC_[c->rowIdx() * gridWidthResults_ + c->colIdx()] = c;
  }

};

#endif //HEDGEHOG_TUTORIALS_PARTIAL_COMPUTATION_STATE_H
