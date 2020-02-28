//
// Created by tjb3 on 10/30/19.
//

#ifndef HEDGEHOG_TUTORIALS_UNIFIED_COMPUTATION_GRAPH_H
#define HEDGEHOG_TUTORIALS_UNIFIED_COMPUTATION_GRAPH_H

#include <hedgehog/hedgehog.h>

#include "../data/unified_matrix_block_data.h"

#include "../cuda_tasks/cuda_prefetch_in_gpu.h"
#include "../cuda_tasks/cuda_product_task.h"

#include "../state/unified_input_block_state.h"

template<class MatrixType>
class UnifiedComputationGraph : public hh::Graph<
    UnifiedMatrixBlockData<MatrixType, 'p'>,
    UnifiedMatrixBlockData<MatrixType, 'a'>, UnifiedMatrixBlockData<MatrixType, 'b'>> {
 public:
  UnifiedComputationGraph(size_t n, size_t m, size_t p, size_t blockSize, size_t numberThreadProduct, bool restrictInputMemory) :
      hh::Graph<UnifiedMatrixBlockData<MatrixType, 'p'>,
          UnifiedMatrixBlockData<MatrixType, 'a'>,
          UnifiedMatrixBlockData<MatrixType, 'b'>>("GPU Computation Graph") {

    size_t nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
    size_t mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
    size_t pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

    // Cuda tasks
    // Tasks
    auto copyInATask =
             std::make_shared<CudaPrefetchInGpu<MatrixType, 'a'>>(pBlocks);
    auto copyInBTask =
             std::make_shared<CudaPrefetchInGpu<MatrixType, 'b'>>(nBlocks);
    auto productTask =
        std::make_shared<CudaProductTask<MatrixType>>(numberThreadProduct);

    // MemoryManagers
    // Uses default constructor to not allocate memory as it is just going to manage the memory coming from main
    auto cudaMemoryManagerA =
             std::make_shared<hh::StaticMemoryManager<UnifiedMatrixBlockData<MatrixType, 'a'>>>(restrictInputMemory ? nBlocks + 4 : nBlocks * mBlocks);
    auto cudaMemoryManagerB =
        std::make_shared<hh::StaticMemoryManager<UnifiedMatrixBlockData<MatrixType, 'b'>>>(restrictInputMemory ? pBlocks + 4 : pBlocks * mBlocks);

    // Statically allocates group of memory
    auto cudaMemoryManagerProduct =
             std::make_shared<hh::StaticMemoryManager<UnifiedMatrixBlockData<MatrixType, 'p'>, size_t>>(8, blockSize);

    // Connect the memory manager
    productTask->connectMemoryManager(cudaMemoryManagerProduct);
    copyInATask->connectMemoryManager(cudaMemoryManagerA);
    copyInBTask->connectMemoryManager(cudaMemoryManagerB);

    // State
    auto stateInputBlock =
             std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);

    // StateManager
    std::shared_ptr<hh::StateManager<
        std::pair<
            std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'a'>>,
        std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'b'>>>,
        UnifiedMatrixBlockData<MatrixType, 'a'>, UnifiedMatrixBlockData<MatrixType, 'b'>>> stateManagerInputBlock =
                                                                                  std::make_shared<hh::StateManager<
                                                                                      std::pair<
                                                                                          std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'a'>>,
                                                                                      std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'b'>>>,
                                                                                      UnifiedMatrixBlockData<MatrixType, 'a'>, UnifiedMatrixBlockData<MatrixType, 'b'>>
                                                                              >("Input State Manager", stateInputBlock);


    // Copy the blocks to the device (NVIDIA GPU)
    this->input(copyInATask);
    this->input(copyInBTask);

    // Connect to the State manager to wait for compatible block of A and B
    this->addEdge(copyInATask, stateManagerInputBlock);
    this->addEdge(copyInBTask, stateManagerInputBlock);

    // Do the CUDA product task
    this->addEdge(stateManagerInputBlock, productTask);

    // Send Out the data
    this->output(productTask);
  }
};

#endif //HEDGEHOG_TUTORIALS_UNIFIED_COMPUTATION_GRAPH_H
