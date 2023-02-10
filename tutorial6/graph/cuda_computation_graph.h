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


#ifndef TUTORIAL6_CUDA_COMPUTATION_GRAPH_H
#define TUTORIAL6_CUDA_COMPUTATION_GRAPH_H

#include <hedgehog/hedgehog.h>

#include "../data/matrix_block_data.h"

#include "../cuda_tasks/cuda_copy_in_gpu.h"
#include "../cuda_tasks/cuda_copy_out_gpu.h"
#include "../cuda_tasks/cuda_product_task.h"

#include "../state/cuda_input_block_state.h"

template<class MatrixType>
class CUDAComputationGraph : public hh::Graph<
    2,
    MatrixBlockData<MatrixType, 'a', Order::Column>, MatrixBlockData<MatrixType, 'b', Order::Column>,
    MatrixBlockData<MatrixType, 'p', Order::Column>
> {
 public:
  CUDAComputationGraph(size_t n, size_t m, size_t p, size_t blockSize, size_t numberThreadProduct) :
      hh::Graph<
          2,
          MatrixBlockData<MatrixType, 'a', Order::Column>, MatrixBlockData<MatrixType, 'b', Order::Column>,
          MatrixBlockData<MatrixType, 'p', Order::Column>
      >("GPU Computation Graph") {
    size_t nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
    size_t mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
    size_t pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

    // Cuda tasks
    // Tasks
    auto copyInATask = std::make_shared<CudaCopyInGpu<MatrixType, 'a'>>(pBlocks, blockSize, n);
    auto copyInBTask = std::make_shared<CudaCopyInGpu<MatrixType, 'b'>>(nBlocks, blockSize, m);
    auto productTask = std::make_shared<CudaProductTask<MatrixType>>(p, numberThreadProduct);
    auto copyOutTask = std::make_shared<CudaCopyOutGpu<MatrixType>>(blockSize);

    // MemoryManagers
    auto cudaMemoryManagerA =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a'>, size_t>>(nBlocks + 4, blockSize);
    auto cudaMemoryManagerB =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b'>, size_t>>(pBlocks + 4, blockSize);
    auto cudaMemoryManagerProduct =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p'>, size_t>>(8, blockSize);

    // Connect the memory manager
    productTask->connectMemoryManager(cudaMemoryManagerProduct);
    copyInATask->connectMemoryManager(cudaMemoryManagerA);
    copyInBTask->connectMemoryManager(cudaMemoryManagerB);

    // State
    auto stateInputBlock = std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);

    // StateManager
    auto stateManagerInputBlock =
        std::make_shared<hh::StateManager<
            2,
            CudaMatrixBlockData<MatrixType, 'a'>, CudaMatrixBlockData<MatrixType, 'b'>,
            std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a'>>,
                      std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b'>>>
        >>(stateInputBlock, "Input State Manager");


    // Copy the blocks to the device (NVIDIA GPU)
    this->inputs(copyInATask);
    this->inputs(copyInBTask);

    // Connect to the State manager to wait for compatible block of A and B
    this->edges(copyInATask, stateManagerInputBlock);
    this->edges(copyInBTask, stateManagerInputBlock);

    // Do the CUDA product task
    this->edges(stateManagerInputBlock, productTask);

    // Send back the memory to the CPU
    this->edges(productTask, copyOutTask);

    // Send Out the data
    this->outputs(copyOutTask);
  }
};

#endif //TUTORIAL6_CUDA_COMPUTATION_GRAPH_H
