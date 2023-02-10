// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


#ifndef TUTORIAL2_BLOCK_STATE_H
#define TUTORIAL2_BLOCK_STATE_H
#include <hedgehog/hedgehog.h>

#include "../data/triplet_matrix_block_data.h"
#include "../data/matrix_block_data.h"

template<class Type, Order Ord>
class BlockState : public hh::AbstractState<
    3,
    MatrixBlockData<Type, 'a', Ord>, MatrixBlockData<Type, 'b', Ord>, MatrixBlockData<Type, 'c', Ord>,
    TripletMatrixBlockData<Type, Ord>> {

 private:
  std::vector<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>> gridMatrixA_ = {};
  std::vector<std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>> gridMatrixB_ = {};
  std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>> gridMatrixC_ = {};

  size_t
      gridHeight_ = 0,
      gridWidth_ = 0;

 public:
  BlockState(size_t gridHeight, size_t gridWidth) : gridHeight_(gridHeight), gridWidth_(gridWidth) {
    gridMatrixA_ = std::vector<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>>
        (gridHeight_ * gridWidth_, nullptr);
    gridMatrixB_ = std::vector<std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>
        (gridHeight_ * gridWidth_, nullptr);
    gridMatrixC_ = std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>>
        (gridHeight_ * gridWidth_, nullptr);
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> ptr) override {
    std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> blockB = nullptr;
    std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> blockC = nullptr;
    auto
        rowIdx = ptr->rowIdx(),
        colIdx = ptr->colIdx();

    if ((blockB = matrixB(rowIdx, colIdx)) && (blockC = matrixC(rowIdx, colIdx))) {
      resetB(rowIdx, colIdx);
      resetC(rowIdx, colIdx);
      auto triplet = std::make_shared<TripletMatrixBlockData<Type, Ord>>();
      triplet->a_ = ptr;
      triplet->b_ = blockB;
      triplet->c_ = blockC;
      this->addResult(triplet);
    } else {
      matrixA(ptr);
    }
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> ptr) override {
    std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> blockA = nullptr;
    std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> blockC = nullptr;
    auto
        rowIdx = ptr->rowIdx(),
        colIdx = ptr->colIdx();

    if ((blockA = matrixA(rowIdx, colIdx)) && (blockC = matrixC(rowIdx, colIdx))) {
      resetA(rowIdx, colIdx);
      resetC(rowIdx, colIdx);
      auto triplet = std::make_shared<TripletMatrixBlockData<Type, Ord>>();
      triplet->a_ = blockA;
      triplet->b_ = ptr;
      triplet->c_ = blockC;
      this->addResult(triplet);
    } else {
      matrixB(ptr);
    }
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> ptr) override {
    std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> blockA = nullptr;
    std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> blockB = nullptr;
    auto
        rowIdx = ptr->rowIdx(),
        colIdx = ptr->colIdx();

    if ((blockA = matrixA(rowIdx, colIdx)) && (blockB = matrixB(rowIdx, colIdx))) {
      resetA(rowIdx, colIdx);
      resetB(rowIdx, colIdx);
      auto triplet = std::make_shared<TripletMatrixBlockData<Type, Ord>>();
      triplet->a_ = blockA;
      triplet->b_ = blockB;
      triplet->c_ = ptr;
      this->addResult(triplet);
    } else {
      matrixC(ptr);
    }
  }

 private:
  std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> matrixA(size_t i, size_t j) {
    return gridMatrixA_[i * gridWidth_ + j];
  }

  std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> matrixB(size_t i, size_t j) {
    return gridMatrixB_[i * gridWidth_ + j];
  }

  std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> matrixC(size_t i, size_t j) {
    return gridMatrixC_[i * gridWidth_ + j];
  }

  void matrixA(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> blockA) {
    gridMatrixA_[blockA->rowIdx() * gridWidth_ + blockA->colIdx()] = blockA;
  }

  void matrixB(std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> blockB) {
    gridMatrixB_[blockB->rowIdx() * gridWidth_ + blockB->colIdx()] = blockB;
  }

  void matrixC(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> blockC) {
    gridMatrixC_[blockC->rowIdx() * gridWidth_ + blockC->colIdx()] = blockC;
  }
  void resetA(size_t i, size_t j) { gridMatrixA_[i * gridWidth_ + j] = nullptr; }
  void resetB(size_t i, size_t j) { gridMatrixB_[i * gridWidth_ + j] = nullptr; }
  void resetC(size_t i, size_t j) { gridMatrixC_[i * gridWidth_ + j] = nullptr; }
};
#endif //TUTORIAL2_BLOCK_STATE_H
