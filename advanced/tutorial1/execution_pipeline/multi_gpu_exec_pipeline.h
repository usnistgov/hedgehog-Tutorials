//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_MULTI_GPU_EXEC_PIPELINE_H
#define HEDGEHOG_TUTORIALS_MULTI_GPU_EXEC_PIPELINE_H

#include <hedgehog/hedgehog.h>
#include "../data/unified_matrix_block_data.h"

template<class MatrixType>
class MultiGPUExecPipeline : public hh::AbstractExecutionPipeline<
    2,
    UnifiedMatrixBlockData<MatrixType, 'a'>,
    UnifiedMatrixBlockData<MatrixType, 'b'>,
    UnifiedMatrixBlockData<MatrixType, 'p'>> {
 private:
  size_t
      numberGraphDuplication_ = 0;

 public:
  MultiGPUExecPipeline(std::shared_ptr<
      hh::Graph<2,
                UnifiedMatrixBlockData<MatrixType, 'a'>,
                UnifiedMatrixBlockData<MatrixType, 'b'>,
                UnifiedMatrixBlockData<MatrixType, 'p'>>> const &graph,
                       std::vector<int> const &deviceIds) : hh::AbstractExecutionPipeline<
      2,
      UnifiedMatrixBlockData<MatrixType, 'a'>,
      UnifiedMatrixBlockData<MatrixType, 'b'>,
      UnifiedMatrixBlockData<MatrixType, 'p'>>(graph, deviceIds, "Cuda Execution Pipeline"),
                                                            numberGraphDuplication_(deviceIds.size()) {}
  virtual ~MultiGPUExecPipeline() = default;

  bool sendToGraph(
      [[maybe_unused]]std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'a'>> &data,
      [[maybe_unused]]size_t const &graphId) override {
    return data->colIdx() % numberGraphDuplication_ == graphId;
  }

  bool sendToGraph(
      [[maybe_unused]]std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'b'>> &data,
      [[maybe_unused]]size_t const &graphId) override {
    return data->rowIdx() % numberGraphDuplication_ == graphId;
  }
};
#endif //HEDGEHOG_TUTORIALS_MULTI_GPU_EXEC_PIPELINE_H
