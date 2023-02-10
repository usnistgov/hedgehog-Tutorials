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


#ifndef TUTORIAL4_PRODUCT_TASK_H
#define TUTORIAL4_PRODUCT_TASK_H
#include <cblas.h>
#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class ProductTask : public hh::AbstractTask<
    1,
    std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>,
    MatrixBlockData<Type, 'p', Ord>
    >{
 private:
  size_t
      countPartialComputation_ = 0;

 public:
  ProductTask(size_t numberExecutionThread, size_t countPartialComputation)
      : hh::AbstractTask<
      1,
      std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>,
      MatrixBlockData<Type, 'p', Ord>
  >("Product Task", numberExecutionThread), countPartialComputation_(countPartialComputation) {}

  void execute(std::shared_ptr<
      std::pair<
          std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>,
          std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>
      >> ptr) override {

    auto matA = ptr->first;
    auto matB = ptr->second;
    auto matP = new Type[matA->blockSizeHeight() * matB->blockSizeWidth()]();

    auto res = std::make_shared<MatrixBlockData<Type, 'p', Ord>>(
        matA->rowIdx(), matB->colIdx(),
        matA->blockSizeHeight(), matB->blockSizeWidth(),
        matP, matP);

    if constexpr(std::is_same<Type, float>::value) {
      cblas_sgemm(Ord == Order::Column ? CblasColMajor : CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), 1,
                  (float *) matA->blockData(), matA->leadingDimension(),
                  (float *) matB->blockData(), matB->leadingDimension(), 1,
                  (float *) matP, res->leadingDimension());
    } else if (std::is_same<Type, double>::value) {
      cblas_dgemm(Ord == Order::Column ? CblasColMajor : CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), 1,
                  (double *) matA->blockData(), matA->leadingDimension(),
                  (double *) matB->blockData(), matB->leadingDimension(), 1,
                  (double *) matP, res->leadingDimension());
    } else {
      std::cerr << "The matrix can't be multiplied" << std::endl;
      exit(43);
    }
    this->addResult(res);
  }
  std::shared_ptr<hh::AbstractTask<
      1,
      std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>,
      MatrixBlockData<Type, 'p', Ord>>
      > copy() override {
    return std::make_shared<ProductTask<Type, Ord>>(this->numberThreads(), countPartialComputation_);
  }
};

#endif //TUTORIAL4_PRODUCT_TASK_H
