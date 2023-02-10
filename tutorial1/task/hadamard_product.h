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


#ifndef TUTORIAL1_HADAMARD_PRODUCT_H
#define TUTORIAL1_HADAMARD_PRODUCT_H

#include <hedgehog/hedgehog.h>

#include "../data/data_type.h"
#include "../data/triplet_matrix_block_data.h"

template<class Type, Order Ord>
class HadamardProduct : public hh::AbstractTask<1, TripletMatrixBlockData<Type, Ord>, MatrixBlockData<Type, 'c', Ord>> {
 public:
  HadamardProduct(std::string const &name, size_t numberThreads)
      : hh::AbstractTask<1, TripletMatrixBlockData<Type, Ord>, MatrixBlockData<Type, 'c', Ord>>(name, numberThreads) {}

  void execute(std::shared_ptr<TripletMatrixBlockData<Type, Ord>> triplet) override {
    // Get the block from the triplet
    auto blockA = triplet->a_;
    auto blockB = triplet->b_;
    auto blockC = triplet->c_;

    // Get the data from the block for easy access
    auto dataA = blockA->blockData();
    auto dataB = blockB->blockData();
    auto dataC = blockC->blockData();

    // Get the block's dimensions and leading dimension
    size_t
        blockWidth = blockA->blockSizeWidth(),
        blockHeight = blockA->blockSizeHeight(),
        ldA = blockA->leadingDimension(),
        ldB = blockB->leadingDimension(),
        ldC = blockC->leadingDimension();

    // Do the computation
    if constexpr (Ord == Order::Row) {
      for (size_t i = 0; i < blockHeight; ++i) {
        for (size_t j = 0; j < blockWidth; ++j) {
          dataC[i * ldC + j] = dataA[i * ldA + j] * dataB[i * ldB + j];
        }
      }
    } else {
      for (size_t j = 0; j < blockWidth; ++j) {
        for (size_t i = 0; i < blockHeight; ++i) {
          dataC[j * ldC + i] = dataA[j * ldA + i] * dataB[j * ldB + i];
        }
      }
    }

    this->addResult(blockC);
  }

  std::shared_ptr<hh::AbstractTask<1, TripletMatrixBlockData<Type, Ord>, MatrixBlockData<Type, 'c', Ord>>>
  copy() override {
    return std::make_shared<HadamardProduct>(this->name(), this->numberThreads());
  }
};

#endif //TUTORIAL1_HADAMARD_PRODUCT_H
