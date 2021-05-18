//
// Created by tjb3 on 10/30/19.
//

//#define HH_DISABLE_CHECK_CUDA

#include <hedgehog/hedgehog.h>
#include <random>
#include <cublasXt.h>

#include "../../utils/tclap/CmdLine.h"
#include "data/unified_matrix_block_data.h"
#include "data/data_type.h"
#include "utils/cuda_utils.h"

#include "graph/unified_computation_graph.h"
#include "execution_pipeline/multi_gpu_exec_pipeline.h"

#include "tasks/addition_task.h"

#include "state/output_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"

bool constexpr isTestResults = false;
bool constexpr doDebugPrintMatrices = false;

class SizeConstraint : public TCLAP::Constraint<size_t> {
 public:
  [[nodiscard]] std::string description() const override {
    return "Positive non null";
  }

  [[nodiscard]] std::string shortID() const override {
    return "NonNullInteger";
  }

  [[nodiscard]] bool check(size_t const &value) const override {
    return value > 0;
  }
};

template<class Type, char Id>
std::shared_ptr<UnifiedMatrixBlockData<Type, Id>>
allocateUnifiedMemory(size_t row, size_t col, size_t blockSizeHeight, size_t blockSizeWidth, std::uniform_real_distribution<Type> &unif,
                      std::mt19937_64 &rng) {
  Type *unifiedMem;

  checkCudaErrors(cudaMallocManaged((void **) &unifiedMem, blockSizeHeight * blockSizeWidth * sizeof(Type)));

  if constexpr (isTestResults) {
    std::for_each(unifiedMem, unifiedMem + (blockSizeHeight * blockSizeWidth),
                  [&unif, &rng](Type &val) { val = (Type) unif(rng); });
  }


  auto data = std::make_shared<UnifiedMatrixBlockData<Type, Id>>(row, col, blockSizeHeight, blockSizeWidth, blockSizeHeight, unifiedMem,
                                                            unifiedMem);
  return data;
}

template<class Type, char Id>
void copyBlock(std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> input, size_t blockSize, Type *output, size_t ld) {

  Type *outputLocation = &output[input->colIdx() * blockSize * ld +
                                 input->rowIdx() * blockSize];

  for (size_t i = 0; i < input->blockSizeWidth(); ++i) {
    for (size_t j = 0; j < input->blockSizeHeight(); ++j) {
      outputLocation[i * ld + j] = input->blockData()[i * input->leadingDimension() + j];
    }
  }
}

template<class Type, char Id>
void printBlock(std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> blk) {
  std::cout << "Block: " << blk->rowIdx() << ", " << blk->colIdx() << std::endl;
  for (size_t i = 0; i < blk->blockSizeHeight(); ++i) {
    for (size_t j = 0; j < blk->blockSizeWidth(); ++j) {
      std::cout << blk->blockData()[j * blk->leadingDimension() + i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<class Type>
void printMatrix(Type *matrix, size_t numRows, size_t numCols) {

  for (size_t i = 0; i < numRows; ++i) {
    for (size_t j = 0; j < numCols; ++j) {
      std::cout << matrix[j * numRows + i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

double computeMatrixMultiplicationGFLOPS(size_t n, size_t m, size_t p, double duration) {
  return ((double) n * (double) m * (double) p * 2. * 1.0e-9) / duration;
}

template<class MatrixType>
bool computeAndGetAccuracyMatrixMultiplication(MatrixType *matrixCResult,
                                                 MatrixType *matrixCBase,
                                                 size_t n,
                                                 size_t p) {
  auto
      *diff = new std::vector<MatrixType>(n * p, 0),
      *accuracyComputation = new std::vector<MatrixType>(n * p, 0),
      *sumBTRow = new std::vector<MatrixType>(p, 0);

  MatrixType
      epsilon = std::numeric_limits<MatrixType>::epsilon(),
      sumA = 0;

  bool
      accurate = false;

  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < n; ++i) {
      (*diff)[j * n + i] = std::fabs(matrixCBase[j * n + i] - matrixCResult[j * n + i]);;
    }
  }

  accurate = std::all_of(diff->begin(), diff->end(), [&epsilon](MatrixType &d) { return d <= 3.0 * epsilon; });

  //  Deallocate matrices metrics
  delete diff;
  delete accuracyComputation;
  delete sumBTRow;

  return accurate;

}

int matrixMultiplicationWithUnifiedMemory(int argc, char **argv) {
  using MatrixType = float;

  // Mersenne Twister Random Generator
  uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
  std::mt19937_64 rng(ss);

  // Choose your distribution depending on the type of MatrixType
  std::uniform_real_distribution<MatrixType> unif(0, 10);
  //  std::uniform_int_distribution<MatrixType> unif(0, 10);

  // Args
  size_t
      n = 0,
      m = 0,
      p = 0,
      nBlocks = 0,
      mBlocks = 0,
      pBlocks = 0,
      blockSize = 0,
      numberThreadProduct = 0,
      numberThreadAddition = 0,
      numRetry = 1;

  bool
    innerTraversal = false,
    restrictInputMemory = true,
    evaluateOutputTime = false;

  std::vector<double>
      resultTimes = {};

  std::vector<int>
      deviceIds = {};

  std::vector<double> runtimes;

  // Allocate matrices
  std::vector<std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'a'>>> aMatrixData;
  std::vector<std::shared_ptr<UnifiedMatrixBlockData<MatrixType, 'b'>>> bMatrixData;
  std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>> cMatrixData;

  // test matrices
  MatrixType *testA,
      *testB,
      *testResult,
      *testHedgehog;

  try {
    TCLAP::CmdLine cmd("Matrix Multiplication parameters", ' ', "0.1");
    SizeConstraint sc;
    TCLAP::ValueArg<size_t> nArg("n", "aheight", "Matrix A Height.", false, 12, &sc);
    cmd.add(nArg);
    TCLAP::ValueArg<size_t> mArg("m", "commondimension", "Common dimension.", false, 12, &sc);
    cmd.add(mArg);
    TCLAP::ValueArg<size_t> pArg("p", "bwidth", "Matrix B Width.", false, 12, &sc);
    cmd.add(pArg);
    TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 10, &sc);
    cmd.add(blockArg);
    TCLAP::ValueArg<size_t> productArg("x", "product", "Product task's number of threads.", false, 3, &sc);
    cmd.add(productArg);
    TCLAP::ValueArg<size_t> additionArg("a", "addition", "Addiction task's number of threads.", false, 3, &sc);
    cmd.add(additionArg);
    TCLAP::ValueArg<size_t> retryArg("r", "retry", "Number of retries.", false, 1, &sc);
    cmd.add(retryArg);
    TCLAP::MultiArg<int> deviceIdsArg(
        "d", "deviceids", "List of device Ids in which the computation will be decomposed.", false, "integer");
    cmd.add(deviceIdsArg);
    TCLAP::SwitchArg traversalArg("i", "inner-traversal", "Uses inner-traversal. Matrices should fit into GPU memory.", false);
    cmd.add(traversalArg);
    TCLAP::SwitchArg evaluateOutputTimeArg("o", "output-time", "Evaluates the output timing for first and average.", false);
    cmd.add(evaluateOutputTimeArg);



    cmd.parse(argc, argv);

    n = nArg.getValue();
    m = mArg.getValue();
    p = pArg.getValue();
    numberThreadAddition = additionArg.getValue();
    numberThreadProduct = productArg.getValue();
    numRetry = retryArg.getValue();
    innerTraversal = traversalArg.getValue();
    evaluateOutputTime = evaluateOutputTimeArg.getValue();

    restrictInputMemory = !innerTraversal;

    blockSize = blockArg.getValue();
    if (blockSize == 0) { blockSize = 1; }

    deviceIds = deviceIdsArg.getValue();
    if (deviceIds.empty()) {
      int numberGPUSAvailable = 0;
      checkCudaErrors(cudaGetDeviceCount(&numberGPUSAvailable));
      deviceIds.assign(numberGPUSAvailable, 0);
      std::iota(deviceIds.begin(), deviceIds.end(), 0);
    }

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  for (auto device :deviceIds) {
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cublasInit());
  }

  nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1),
  mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1),
  pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

  if (isTestResults) {
    testA = new MatrixType[m * n];
    testB = new MatrixType[m * p];
    testResult = new MatrixType[n * p];
    testHedgehog = new MatrixType[n * p];
  }

  size_t blockSizeRemainderAWidth = m % blockSize;
  size_t blockSizeRemainderAHeight = n % blockSize;

  size_t blockSizeRemainderBWidth = p % blockSize;
  size_t blockSizeRemainderBHeight = m % blockSize;

  size_t blockSizeRemainderCWidth = p % blockSize;
  size_t blockSizeRemainderCHeight = n % blockSize;

  // Allocate blocks and setup traversal
  if (innerTraversal) {
    for (size_t i = 0; i < nBlocks; ++i) {
      size_t blockSizeHeight = blockSize;
      if (i == nBlocks-1 && blockSizeRemainderAHeight > 0) {
        blockSizeHeight = blockSizeRemainderAHeight;
      }

      for (size_t j = 0; j < mBlocks; ++j) {
        size_t blockSizeWidth = blockSize;
        if (j == mBlocks-1 &&  blockSizeRemainderAWidth > 0) {
          blockSizeWidth = blockSizeRemainderAWidth;
        }
        aMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'a'>(i, j, blockSizeHeight, blockSizeWidth, unif, rng));
      }
    }
  }
  else
  {
    for (size_t j = 0; j < mBlocks; ++j) {
      size_t blockSizeWidth = blockSize;
      if (j == mBlocks-1 &&  blockSizeRemainderAWidth > 0) {
        blockSizeWidth = blockSizeRemainderAWidth;
      }
      for (size_t i = 0; i < nBlocks; ++i) {
        size_t blockSizeHeight = blockSize;
        if (i == nBlocks-1 && blockSizeRemainderAHeight > 0) {
          blockSizeHeight = blockSizeRemainderAHeight;
        }
        aMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'a'>(i, j, blockSizeHeight, blockSizeWidth, unif, rng));
      }
    }
  }

  if (innerTraversal) {
    for (size_t j = 0; j < pBlocks; ++j) {
      size_t blockSizeWidth = blockSize;
      if (j == pBlocks-1 &&  blockSizeRemainderBWidth > 0) {
        blockSizeWidth = blockSizeRemainderBWidth;
      }
      for (size_t i = 0; i < mBlocks; ++i) {
        size_t blockSizeHeight = blockSize;
        if (i == mBlocks-1 && blockSizeRemainderBHeight > 0) {
          blockSizeHeight = blockSizeRemainderBHeight;
        }
        bMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'b'>(i, j, blockSize, blockSize, unif, rng));
      }
    }
  }
  else {
    for (size_t i = 0; i < mBlocks; ++i) {
      size_t blockSizeHeight = blockSize;
      if (i == mBlocks-1 && blockSizeRemainderBHeight > 0) {
        blockSizeHeight = blockSizeRemainderBHeight;
      }
      for (size_t j = 0; j < pBlocks; ++j) {
        size_t blockSizeWidth = blockSize;
        if (j == pBlocks-1 &&  blockSizeRemainderBWidth > 0) {
          blockSizeWidth = blockSizeRemainderBWidth;
        }
        bMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'b'>(i, j, blockSizeHeight, blockSizeWidth, unif, rng));
      }
    }
  }

  for (size_t i = 0; i < nBlocks; ++i) {
    size_t blockSizeHeight = blockSize;
    if (i == nBlocks-1 && blockSizeRemainderCHeight > 0) {
      blockSizeHeight = blockSizeRemainderCHeight;
    }
    for (size_t j = 0; j < pBlocks; ++j) {
      size_t blockSizeWidth = blockSize;
      if (j == pBlocks-1 && blockSizeRemainderCWidth > 0) {
        blockSizeWidth = blockSizeRemainderCWidth;
      }
      MatrixType *c = new MatrixType[blockSizeHeight * blockSizeWidth]();
      if constexpr(isTestResults) {
        std::for_each(c, c + (blockSizeHeight * blockSizeWidth), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
      }

      auto blkData = std::make_shared<MatrixBlockData<MatrixType, 'c', Order::Column>>(i, j, blockSizeHeight, blockSizeWidth,
                                                                                       blockSizeHeight, c, c);
      cMatrixData.push_back(blkData);
    }
  }

  std::cout << "experiment,numGPUs,numThreadsProduct,numThreadsAddition,n,m,p,blockSize,time(s),gflops,first,avg";

  if constexpr (isTestResults) {
    std::cout << ",accurate";
  }
  std::cout << std::endl;

    for (size_t retryNum = 0; retryNum < numRetry; ++retryNum) {
    for (auto mat: aMatrixData) {
      checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), mat->blockSizeWidth()
                                                                 * mat->blockSizeHeight() * sizeof(MatrixType), cudaCpuDeviceId));
    }

    for (auto mat: bMatrixData) {
      checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), mat->blockSizeWidth()
                                                                  * mat->blockSizeHeight() * sizeof(MatrixType), cudaCpuDeviceId));
    }

    for (auto device :deviceIds) {
      checkCudaErrors(cudaSetDevice(device));
      checkCudaErrors(cudaDeviceSynchronize());
    }
    if constexpr (isTestResults) {
      if constexpr (doDebugPrintMatrices) {
        std::cout << "MatA" << std::endl;
      }
      for (auto mat : aMatrixData) {
        if constexpr (doDebugPrintMatrices) {
          printBlock<MatrixType, 'a'>(mat);
        }
        copyBlock<MatrixType, 'a'>(mat, blockSize, testA, n);
      }

      if constexpr (doDebugPrintMatrices) {
        std::cout << "MatB" << std::endl;
      }
      for (auto mat : bMatrixData) {
        if constexpr (doDebugPrintMatrices) {
          printBlock<MatrixType, 'b'>(mat);
        }
        copyBlock<MatrixType, 'b'>(mat, blockSize, testB, m);
      }
      if constexpr (doDebugPrintMatrices) {
        std::cout << "MatC" << std::endl;
      }
      for (auto mat : cMatrixData) {
        if constexpr (doDebugPrintMatrices) {
          printBlock<MatrixType, 'c'>(mat);
        }
        copyBlock<MatrixType, 'c'>(mat, blockSize, testResult, n);
      }

      if constexpr (doDebugPrintMatrices) {
        std::cout << "testA" << std::endl;
        printMatrix<MatrixType>(testA, n, m);
        std::cout << "testB" << std::endl;
        printMatrix<MatrixType>(testB, m, p);
        std::cout << "testResult" << std::endl;
        printMatrix<MatrixType>(testResult, n, p);
      }
    }


    // Graph
    // Global Graph
    auto matrixMultiplicationGraph =
        hh::Graph<MatrixBlockData<MatrixType, 'c', Order::Column>,
            UnifiedMatrixBlockData<MatrixType, 'a'>, UnifiedMatrixBlockData<MatrixType, 'b'>, MatrixBlockData<MatrixType, 'c', Order::Column>>
            ("Matrix Multiplication Graph");

    // GPU Graph
    auto cudaMatrixMultiplication = std::make_shared<UnifiedComputationGraph<MatrixType>>(n, m, p, blockSize,
                                                                                          numberThreadProduct, restrictInputMemory);

    // Execution Pipeline
    auto executionPipeline =
        std::make_shared<MultiGPUExecPipeline<MatrixType>>(cudaMatrixMultiplication, deviceIds.size(), deviceIds);

    // Tasks
    auto additionTask =
        std::make_shared<AdditionTask<MatrixType>>(numberThreadAddition);

    // State
    auto statePartialComputation =
        std::make_shared<PartialComputationState<MatrixType>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks);
    auto stateOutput =
        std::make_shared<OutputState<MatrixType>>(nBlocks, pBlocks, mBlocks);

    // State Manager
    auto stateManagerPartialComputation =
        std::make_shared<PartialComputationStateManager<MatrixType>>(statePartialComputation);
    auto stateManagerOutputBlock =
        std::make_shared<hh::StateManager<
            MatrixBlockData<MatrixType, 'c', Order::Column>,
            MatrixBlockData<MatrixType, 'c', Order::Column>>>("Output State Manager", stateOutput);

    // Build the graph
    // Send the blocks to GPU Graph
    matrixMultiplicationGraph.input(executionPipeline);
    matrixMultiplicationGraph.input(executionPipeline);
    matrixMultiplicationGraph.input(stateManagerPartialComputation);

    // Get back the blocks out the GPU Graph
    matrixMultiplicationGraph.addEdge(executionPipeline, stateManagerPartialComputation);

    // Use the same graph for the accumulation
    matrixMultiplicationGraph.addEdge(stateManagerPartialComputation, additionTask);
    matrixMultiplicationGraph.addEdge(additionTask, stateManagerPartialComputation);
    matrixMultiplicationGraph.addEdge(additionTask, stateManagerOutputBlock);
    matrixMultiplicationGraph.output(stateManagerOutputBlock);

    auto begin = std::chrono::system_clock::now();


    // Execute the graph
    matrixMultiplicationGraph.executeGraph();

    // Push the matrices
    for (auto data : aMatrixData) {
      matrixMultiplicationGraph.pushData(data);
    }

    for (auto data : bMatrixData) {
      matrixMultiplicationGraph.pushData(data);
    }

    for (auto data : cMatrixData) {
      matrixMultiplicationGraph.pushData(data);
    }

    // Notify push done
    matrixMultiplicationGraph.finishPushingData();

    if (evaluateOutputTime) {
      auto startFirstItem = std::chrono::high_resolution_clock::now();
      while (auto res = matrixMultiplicationGraph.getBlockingResult()) {
        auto endFirstItem = std::chrono::high_resolution_clock::now();
        auto durationItem =
            std::chrono::duration_cast<std::chrono::duration<double>>(endFirstItem - startFirstItem).count();
        resultTimes.push_back(durationItem);
        startFirstItem = std::chrono::high_resolution_clock::now();
      }
    }

    bool isAccurate = false;

    if constexpr (isTestResults) {
      while (auto res = matrixMultiplicationGraph.getBlockingResult()) {
        copyBlock<MatrixType, 'c'>(res, blockSize, testHedgehog, n);
//        printBlock<MatrixType, 'c'>(res);
      }

      if constexpr (doDebugPrintMatrices) {
        std::cout << "Hedgehog Final Results:" << std::endl;
        printMatrix(testHedgehog, n, p);
        std::cout << std::endl;
      }

      cublasXtHandle_t handle;
      checkCudaErrors(cublasXtCreate(&handle));

      checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), deviceIds.data()));
      checkCudaErrors(cublasXtSetBlockDim(handle, blockSize));

      MatrixType alpha = 1.0, beta = 1.0;

      if constexpr (std::is_same<MatrixType, double>::value) {
        checkCudaErrors(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, p, m, (double *) (&alpha),
                                          (double *) testA, n,
                                          (double *) testB, m,
                                          (double *) (&beta), (double *) testResult, n));
      } else {
        checkCudaErrors(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, p, m, (float *) (&alpha),
                                          (float *) testA, n,
                                          (float *) testB, m,
                                          (float *) (&beta), (float *) testResult, n));
      }

      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cublasXtDestroy(handle));

      if constexpr (doDebugPrintMatrices) {
        std::cout << "Test Final Results:" << std::endl;
        printMatrix(testResult, n, p);
        std::cout << std::endl;
      }

      isAccurate = computeAndGetAccuracyMatrixMultiplication<MatrixType>(testHedgehog, testResult, n, p);


    }

    // Wait for the graph to terminate
    matrixMultiplicationGraph.waitForTermination();

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin).count();
    double average = 0.0;
    if (evaluateOutputTime) {
      average = std::accumulate(resultTimes.begin(), resultTimes.end(), 0.0) / resultTimes.size();
    }

    runtimes.push_back(duration);

    std::string experimentName = "hedgehogGPUMatMul";
    if (innerTraversal)
    {
      experimentName += "Inner";
    } else {
      experimentName += "Outer";
    }

    if (!evaluateOutputTime) {
      experimentName += "NoOutput";
    }


    std::cout << experimentName << "," << deviceIds.size() << "," << numberThreadProduct << "," << numberThreadAddition
              << ","
              << n << "," << m << "," << p << "," << blockSize << "," << duration << "," <<
              computeMatrixMultiplicationGFLOPS(n, m, p, duration) << "," << (evaluateOutputTime ? resultTimes[0] : 0.0) << "," << average;

    if constexpr (isTestResults) {
      std::cout << "," << std::boolalpha << isAccurate;
    }
    std::cout << std::endl;

    matrixMultiplicationGraph
        .createDotFile("AdvancedTutorial1-" + std::to_string(innerTraversal) + "-" + std::to_string(n) + "-" + std::to_string(blockSize) +  "-" + std::to_string(deviceIds.size()) + "-" + std::to_string(numberThreadProduct) + "-" + std::to_string(retryNum) +  ".dot", hh::ColorScheme::EXECUTION,
                       hh::StructureOptions::QUEUE, hh::DebugOptions::NONE, true);

    resultTimes.clear();
  }


//  double avgRuntime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
//
//  std::cout << "Avg: " << avgRuntime << ", " << computeMatrixMultiplicationGFLOPS(n, m, p, avgRuntime);


  for (auto mat: aMatrixData) {
    checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), mat->blockSizeWidth()
                                                                * mat->blockSizeHeight() * sizeof(MatrixType), cudaCpuDeviceId));
  }

  for (auto mat: bMatrixData) {
    checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), mat->blockSizeWidth()
                                                                * mat->blockSizeHeight() * sizeof(MatrixType), cudaCpuDeviceId));
  }

  for (auto device :deviceIds) {
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  for (auto partialData : cMatrixData) {
    delete[] partialData->blockData();
    partialData->blockData(nullptr);
  }

  for (auto partialData : aMatrixData) {
    checkCudaErrors(cudaFree(partialData->blockData()));
    partialData->blockData(nullptr);
  }

  for (auto partialData : bMatrixData) {
    checkCudaErrors(cudaFree(partialData->blockData()));
    partialData->blockData(nullptr);
  }

  for (auto device :deviceIds) {
    cudaSetDevice(device);
    checkCudaErrors(cublasShutdown());
  }
  return 0;
}


int main(int argc, char *argv[]) {
  return matrixMultiplicationWithUnifiedMemory(argc, argv);
}
