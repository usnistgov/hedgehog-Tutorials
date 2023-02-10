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

//

#ifndef TUTORIAL6_MULTI_GPU_EXEC_PIPELINE_H
#define TUTORIAL6_MULTI_GPU_EXEC_PIPELINE_H

#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"

template<class MatrixType>
class MultiGPUExecPipeline : public hh::AbstractExecutionPipeline<
    2,
    MatrixBlockData<MatrixType, 'a', Order::Column>,
    MatrixBlockData<MatrixType, 'b', Order::Column>,
    MatrixBlockData<MatrixType, 'p', Order::Column>
> {
 private:
  size_t numberGraphDuplication_ = 0;

 public:
  MultiGPUExecPipeline(std::shared_ptr<hh::Graph<
      2,
      MatrixBlockData<MatrixType, 'a', Order::Column>, MatrixBlockData<MatrixType, 'b', Order::Column>,
      MatrixBlockData<MatrixType, 'p', Order::Column>
  >> const &graph, std::vector<int> const &deviceIds)
      : hh::AbstractExecutionPipeline<
      2,
      MatrixBlockData<MatrixType, 'a', Order::Column>, MatrixBlockData<MatrixType, 'b', Order::Column>,
      MatrixBlockData<MatrixType, 'p', Order::Column>
  >(graph, deviceIds, "Cuda Execution Pipeline"), numberGraphDuplication_(deviceIds.size()) {}
  virtual ~MultiGPUExecPipeline() = default;

  bool sendToGraph(
      std::shared_ptr<MatrixBlockData<MatrixType, 'a', Order::Column>> &data,
      size_t const &graphId) override {
    return data->colIdx() % numberGraphDuplication_ == graphId;
  }

  bool sendToGraph(
      std::shared_ptr<MatrixBlockData<MatrixType, 'b', Order::Column>> &data,
      size_t const &graphId) override {
    return data->rowIdx() % numberGraphDuplication_ == graphId;
  }
};

#endif //TUTORIAL6_MULTI_GPU_EXEC_PIPELINE_H
