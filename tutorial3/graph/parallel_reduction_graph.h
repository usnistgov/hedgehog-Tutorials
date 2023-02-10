#ifndef HEDGEHOG_TUTORIALS_PARALLEL_REDUCTION_GRAPH_H_
#define HEDGEHOG_TUTORIALS_PARALLEL_REDUCTION_GRAPH_H_

#include <hedgehog/hedgehog.h>
#include "../task/final_reduction_task.h"
#include "../task/vector_decomposition.h"
#include "../task/reduction_task.h"

template<class DataType>
class ParallelReductionGraph : public hh::Graph<1, std::vector<DataType>, DataType> {
 public:
  ParallelReductionGraph(std::function<DataType(DataType const &, DataType const &)> const &lambda,
                         std::size_t blockSize,
                         std::size_t numberThreadsReduction = 1) : hh::Graph<1, std::vector<DataType>, DataType>(
      "Parallel reduction graph") {
    auto finalReduction = std::make_shared<FinalReductionTask<DataType>>(lambda);
    auto vectorDecomposition = std::make_shared<VectorDecomposition<DataType>>(blockSize, finalReduction.get());
    auto reductionTask = std::make_shared<ReductionTask<DataType>>(numberThreadsReduction, lambda);

    this->inputs(vectorDecomposition);
    this->edges(vectorDecomposition, reductionTask);
    this->edges(reductionTask, finalReduction);
    this->outputs(finalReduction);
  }
};

#endif //HEDGEHOG_TUTORIALS_PARALLEL_REDUCTION_GRAPH_H_
