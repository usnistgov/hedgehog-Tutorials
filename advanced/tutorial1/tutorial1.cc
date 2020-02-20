//
// Created by tjb3 on 10/30/19.
//

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
allocateUnifiedMemory(size_t row, size_t col, size_t blockSize, std::uniform_real_distribution<Type> &unif,
                      std::mt19937_64 &rng) {
  Type *unifiedMem;

  hh::checkCudaErrors(cudaMallocManaged((void **) &unifiedMem, blockSize * blockSize * sizeof(Type)));

  if constexpr (isTestResults) {
    std::for_each(unifiedMem, unifiedMem + (blockSize * blockSize),
                  [&unif, &rng](Type &val) { val = (Type) unif(rng); });
  }


  return std::make_shared<UnifiedMatrixBlockData<Type, Id>>(row, col, blockSize, blockSize, blockSize, unifiedMem,
                                                            unifiedMem);

}

template<class Type, char Id>
void copyBlock(std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> input, Type *output, size_t ld) {

  Type *outputLocation = &output[input->colIdx() * input->blockSizeWidth() * ld +
                                 input->rowIdx() * input->blockSizeHeight()];

  for (size_t col = 0; col < input->blockSizeWidth(); ++col) {
    std::copy_n(input->blockData() + col * input->blockSizeHeight(), input->blockSizeHeight(),
                outputLocation + col * ld);
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
void computeAndPrintAccuracyMatrixMultiplication(MatrixType *matrixCResult,
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


//    for (size_t k = 0; k < p; ++k) {
//      sumBTRow->at(k) = std::accumulate(bTrasnpose->begin() + k * m, bTrasnpose->begin() + (k + 1) * m, 0.);
//    }
//    for (size_t i = 0; i < n; ++i) {
//      sumA = std::accumulate(matrixA->begin() + i * m, matrixA->begin() + (i + 1) * m, 0.);
//
//      for (size_t k = 0; k < p; ++k) {
//        accuracyComputation->at(i * p + k) = 3 * epsilon + 2 * epsilon * (sumA + sumBTRow->at(k));
//      }
//    }
//    accurate = *diff <= *accuracyComputation;
//  }

  std::cout << "," << std::boolalpha << accurate;

  //  Deallocate matrices metrics
  delete diff;
//  delete bTrasnpose;
  delete accuracyComputation;
  delete sumBTRow;
}

int matrixMultiplicationWithUnifiedMemory(int argc, char **argv) {
  using MatrixType = float;

  // Mersenne Twister Random Generator
  uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
  std::mt19937_64 rng(ss);

  // Choose your distribution depending on the type of MatrixType
  std::uniform_real_distribution<float> unif(0, 10);
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
    TCLAP::ValueArg<size_t> nArg("n", "aheight", "Matrix A Height.", false, 10, &sc);
    cmd.add(nArg);
    TCLAP::ValueArg<size_t> mArg("m", "commondimension", "Common dimension.", false, 10, &sc);
    cmd.add(mArg);
    TCLAP::ValueArg<size_t> pArg("p", "bwidth", "Matrix B Width.", false, 10, &sc);
    cmd.add(pArg);
    TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 3, &sc);
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

    cmd.parse(argc, argv);

    n = nArg.getValue();
    m = mArg.getValue();
    p = pArg.getValue();
    numberThreadAddition = additionArg.getValue();
    numberThreadProduct = productArg.getValue();
    numRetry = retryArg.getValue();

    blockSize = blockArg.getValue();
    if (blockSize == 0) { blockSize = 1; }

    deviceIds = deviceIdsArg.getValue();
    if (deviceIds.empty()) {
      int numberGPUSAvailable = 0;
      hh::checkCudaErrors(cudaGetDeviceCount(&numberGPUSAvailable));
      deviceIds.assign(numberGPUSAvailable, 0);
      std::iota(deviceIds.begin(), deviceIds.end(), 0);
    }

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  for (auto device :deviceIds) {
    hh::checkCudaErrors(cudaSetDevice(device));
    hh::checkCudaErrors(cublasInit());
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


  // Allocate blocks and setup traversal
  for (size_t j = 0; j < mBlocks; ++j) {
    for (size_t i = 0; i < nBlocks; ++i) {
      aMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'a'>(i, j, blockSize, unif, rng));
    }
  }

  for (size_t i = 0; i < mBlocks; ++i) {
    for (size_t j = 0; j < pBlocks; ++j) {
      bMatrixData.push_back(allocateUnifiedMemory<MatrixType, 'b'>(i, j, blockSize, unif, rng));
    }
  }

  for (size_t i = 0; i < nBlocks; ++i) {
    for (size_t j = 0; j < pBlocks; ++j) {
      MatrixType *c = new MatrixType[blockSize * blockSize]();
      if constexpr(isTestResults) {
        std::for_each(c, c + (blockSize * blockSize), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
      }


      auto blkData = std::make_shared<MatrixBlockData<MatrixType, 'c', Order::Column>>(i, j, blockSize, blockSize,
                                                                                       blockSize, c, c);
      cMatrixData.push_back(blkData);
    }
  }

  std::cout << "experiment,numGPUs, numThreadsProduct, numThreadsAddition, n,m,p,blockSize,time(s),gflops" << std::endl;

  for (size_t retryNum = 0; retryNum < numRetry; ++retryNum) {
    for (auto mat: aMatrixData) {
      hh::checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), blockSize
                                                                 * blockSize * sizeof(MatrixType), cudaCpuDeviceId));
    }

    for (auto mat: bMatrixData) {
      hh::checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), blockSize
                                                                 * blockSize * sizeof(MatrixType), cudaCpuDeviceId));
    }

    for (auto device :deviceIds) {
      hh::checkCudaErrors(cudaSetDevice(device));
      hh::checkCudaErrors(cudaDeviceSynchronize());
    }
    if constexpr (isTestResults) {
//      std::cout << "MatA" << std::endl;
      for (auto mat : aMatrixData) {
//        printBlock<MatrixType, 'a'>(mat);
        copyBlock<MatrixType, 'a'>(mat, testA, n);
      }

//      std::cout << "MatB" << std::endl;
      for (auto mat : bMatrixData) {
//        printBlock<MatrixType, 'b'>(mat);
        copyBlock<MatrixType, 'b'>(mat, testB, m);
      }

//      std::cout << "MatC" << std::endl;
      for (auto mat : cMatrixData) {
//        printBlock<MatrixType, 'c'>(mat);
        copyBlock<MatrixType, 'c'>(mat, testResult, n);
      }
    }

//    std::cout << "testA" << std::endl;
//    printMatrix<MatrixType>(testA, n, m);
//    std::cout << "testB" << std::endl;
//    printMatrix<MatrixType>(testB, m, p);
//    std::cout << "testResult" << std::endl;
//    printMatrix<MatrixType>(testResult, n, p);

    // Graph
    // Global Graph
    auto matrixMultiplicationGraph =
        hh::Graph<MatrixBlockData<MatrixType, 'c', Order::Column>,
            UnifiedMatrixBlockData<MatrixType, 'a'>, UnifiedMatrixBlockData<MatrixType, 'b'>, MatrixBlockData<MatrixType, 'c', Order::Column>>
            ("Matrix Multiplication Graph");

    // GPU Graph
    auto cudaMatrixMultiplication = std::make_shared<UnifiedComputationGraph<MatrixType>>(n, m, p, blockSize,
                                                                                          numberThreadProduct);

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

    if constexpr (isTestResults) {
      while (auto res = matrixMultiplicationGraph.getBlockingResult()) {
        copyBlock<MatrixType, 'c'>(res, testHedgehog, n);
//        printBlock<MatrixType, 'c'>(res);
      }

//      printMatrix(testHedgehog, n, p);

//      std::cout << "=========================================================" << std::endl;

      cublasXtHandle_t handle;
      hh::checkCudaErrors(cublasXtCreate(&handle));

      hh::checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), deviceIds.data()));
      hh::checkCudaErrors(cublasXtSetBlockDim(handle, blockSize));

      MatrixType alpha = 1.0, beta = 1.0;

      if constexpr (std::is_same<MatrixType, double>::value) {
        hh::checkCudaErrors(cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, p, m, (double *) (&alpha),
                                          (double *) testA, n,
                                          (double *) testB, m,
                                          (double *) (&beta), (double *) testResult, n));
      } else {
        hh::checkCudaErrors(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, p, m, (float *) (&alpha),
                                          (float *) testA, n,
                                          (float *) testB, m,
                                          (float *) (&beta), (float *) testResult, n));
      }

      hh::checkCudaErrors(cudaDeviceSynchronize());
      hh::checkCudaErrors(cublasXtDestroy(handle));

//      printMatrix(testResult, n, p);

      computeAndPrintAccuracyMatrixMultiplication<MatrixType>(testHedgehog, testResult, n, p);


    }

    // Wait for the graph to terminate
    matrixMultiplicationGraph.waitForTermination();
    hh::checkCudaErrors(cudaGetLastError());
    for (auto device :deviceIds) {
      hh::checkCudaErrors(cudaSetDevice(device));
      hh::checkCudaErrors(cudaDeviceSynchronize());
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin).count();

    runtimes.push_back(duration);


    std::cout << "hedgehogGPUMatMul," << deviceIds.size() << "," << numberThreadProduct << "," << numberThreadAddition
              << ","
              << n << "," << m << "," << p << "," << blockSize << "," << duration << "," <<
              computeMatrixMultiplicationGFLOPS(n, m, p, duration) << std::endl;
//    matrixMultiplicationGraph
//        .createDotFile(std::to_string(retryNum) + "AdvancedTutorial1.dot", hh::ColorScheme::EXECUTION,
//                       hh::StructureOptions::ALL);

  }


//  double avgRuntime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
//
//  std::cout << "Avg: " << avgRuntime << ", " << computeMatrixMultiplicationGFLOPS(n, m, p, avgRuntime);


  for (auto mat: aMatrixData) {
    hh::checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), blockSize
                                                               * blockSize * sizeof(MatrixType), cudaCpuDeviceId));
  }

  for (auto mat: bMatrixData) {
    hh::checkCudaErrors(cudaMemPrefetchAsync(mat->blockData(), blockSize
                                                               * blockSize * sizeof(MatrixType), cudaCpuDeviceId));
  }

  for (auto device :deviceIds) {
    hh::checkCudaErrors(cudaSetDevice(device));
    hh::checkCudaErrors(cudaDeviceSynchronize());
  }

  for (auto partialData : cMatrixData) {
    delete[] partialData->blockData();
    partialData->blockData(nullptr);
  }

  for (auto partialData : aMatrixData) {
    hh::checkCudaErrors(cudaFree(partialData->blockData()));
    partialData->blockData(nullptr);
  }

  for (auto partialData : bMatrixData) {
    hh::checkCudaErrors(cudaFree(partialData->blockData()));
    partialData->blockData(nullptr);
  }

  for (auto device :deviceIds) {
    cudaSetDevice(device);
    hh::checkCudaErrors(cublasShutdown());
  }
  return 1;
}


int testUnifedMemory(int argc, char **argv) {
  using Type = float;
  size_t
      blockSize = 0,
      numRetry = 1,
      amountOfMemory = 0;

  try {
    TCLAP::CmdLine cmd("Test Unified memory parameters", ' ', "0.1");
    SizeConstraint sc;
    TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 3, &sc);
    cmd.add(blockArg);
    TCLAP::ValueArg<size_t> retryArg("r", "retry", "Number of retries.", false, 1, &sc);
    cmd.add(retryArg);
    TCLAP::ValueArg<size_t> amountArg("a", "amount", "Amount of unified memory.", false, 1, &sc);
    cmd.add(amountArg);

    cmd.parse(argc, argv);
    numRetry = retryArg.getValue();
    amountOfMemory = amountArg.getValue();

    if (numRetry == 0) { numRetry = 1; }

    blockSize = blockArg.getValue();
    if (blockSize == 0) { blockSize = 1; }

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  std::vector<Type *> pointers;

  for (size_t i = 0; i < numRetry; ++i) {

    cudaStream_t stream_;
    cublasHandle_t handle_;
    hh::checkCudaErrors(cudaSetDevice(0));
    hh::checkCudaErrors(cudaStreamCreate(&stream_));
    hh::checkCudaErrors(cublasCreate_v2(&handle_));
    hh::checkCudaErrors(cublasSetStream_v2(handle_, stream_));

    for (size_t j = 0; j < amountOfMemory; ++j) {
      Type *myBlockOfData;
      hh::checkCudaErrors(cudaMallocManaged((void **) &myBlockOfData, sizeof(Type) * blockSize * blockSize));
      pointers.push_back(myBlockOfData);
    }
    // [Do STUFF]

    for (auto ptr : pointers) {
      hh::checkCudaErrors(cudaMemPrefetchAsync(ptr, blockSize * blockSize * sizeof(Type), 0, stream_));
    }

    for (auto ptr : pointers) {
      Type alpha = 1.0;
      Type beta = 1.0;
      hh::checkCudaErrors(
          cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         blockSize, blockSize, blockSize, &alpha,
                         (float *) ptr, blockSize,
                         (float *) ptr, blockSize, &beta,
                         (float *) ptr, blockSize)
      );
    }

    for (auto ptr : pointers) {
      hh::checkCudaErrors(cudaMemPrefetchAsync(ptr, blockSize * blockSize * sizeof(Type), cudaCpuDeviceId, stream_));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    hh::checkCudaErrors(cudaStreamSynchronize(stream_));

    for (auto ptr : pointers) {
      for (size_t r = 0; r < blockSize*blockSize ; ++r) {
        ptr[r] = 9000.0f;
      }
//      std::memset(ptr, 9000, sizeof(Type) * blockSize * blockSize);
    }

    // [Do STUFF]
    for (auto ptr : pointers) {
      hh::checkCudaErrors(cudaFree(ptr));
    }

    pointers.clear();

    hh::checkCudaErrors(cublasDestroy_v2(handle_));
    hh::checkCudaErrors(cudaStreamDestroy(stream_));
  }

  std::cout << "Finished test unified memory" << std::endl;
  return 1;
}

int main(int argc, char *argv[]) {
//  return matrixMultiplicationWithUnifiedMemory(argc, argv);

  std::thread t1(testUnifedMemory, argc, argv);
  std::thread t2(testUnifedMemory, argc, argv);
  std::thread t3(testUnifedMemory, argc, argv);
  std::thread t4(testUnifedMemory, argc, argv);
//  return testUnifedMemory(argc, argv);

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  return 0;

}
