//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_MULTI_GPU_EXEC_PIPELINE_H
#define HEDGEHOG_TUTORIALS_MULTI_GPU_EXEC_PIPELINE_H

#include <hedgehog/hedgehog.h>
#include "../data/unified_matrix_block_data.h"

template<class MatrixType>
class MultiGPUExecPipeline : public hh::AbstractExecutionPipeline<
    UnifiedMatrixBlockData<MatrixType, 'p'>,
    UnifiedMatrixBlockData<MatrixType, 'a'>,
    UnifiedMatrixBlockData<MatrixType, 'b'>> {
 private:
  size_t
      numberGraphDuplication_ = 0;

 public:
  MultiGPUExecPipeline(std::shared_ptr<
      hh::Graph<UnifiedMatrixBlockData<MatrixType, 'p'>,
          UnifiedMatrixBlockData<MatrixType, 'a'>,
          UnifiedMatrixBlockData<MatrixType, 'b'>>> const &graph,
                       size_t const &numberGraphDuplications,
                       std::vector<int> const &deviceIds) : hh::AbstractExecutionPipeline<
      UnifiedMatrixBlockData<MatrixType, 'p'>,
      UnifiedMatrixBlockData<MatrixType, 'a'>,
      UnifiedMatrixBlockData<MatrixType, 'b'>>("Cuda Execution Pipeline",
                                                       graph,
                                                       numberGraphDuplications,
                                                       deviceIds), numberGraphDuplication_(numberGraphDuplications) {}
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
