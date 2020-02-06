#include <memory>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cublas.h>
#include <cublasXt.h>
#include <random>
#include "../../utils/tclap/CmdLine.h"

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

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, [[maybe_unused]]const char *file, [[maybe_unused]]const int line) {
  if (cudaSuccess != err) {
    std::cerr << "checkCudaErrors() Cuda error = "
              << err
              << "\"" << cudaGetErrorString(err) << " \" from "
              << file << ":" << line << std::endl;
    cudaDeviceReset();
    exit(43);
  }
}
// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cublasStatus_t err, const char *file, const int line) {
  if (CUBLAS_STATUS_SUCCESS != err) {
    std::cerr << "checkCudaErrors() Status Error = "
              << err << " from "
              << file << ":" << line << std::endl;
    cudaDeviceReset();
    exit(44);
  }
}
#endif
#include "../../utils/tclap/CmdLine.h"
double computeMatrixMultiplicationGFLOPS(size_t n, size_t m, size_t p, double duration) {
  return ((double) n * (double) m * (double) p * 2. * 1.0e-9) / duration;
}

int main(int argc, char *argv[]) {
  using MatrixType = float;

  // Mersenne Twister Random Generator
  uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
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
      blockSize = 0,
      numRetry = 1;

  std::vector<int>
      deviceIds = {};

  std::vector<double> runtimes;

  // Allocate matrices
  // test matrices
  MatrixType *testA,
      *testB,
      *testResult;

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
    TCLAP::ValueArg<size_t> retryArg("r", "retry", "Number of retries.", false, 1, &sc);
    cmd.add(retryArg);
    TCLAP::MultiArg<int> deviceIdsArg(
        "d", "deviceids", "List of device Ids in which the computation will be decomposed.", false, "integer");
    cmd.add(deviceIdsArg);

    cmd.parse(argc, argv);

    n = nArg.getValue();
    m = mArg.getValue();
    p = pArg.getValue();
    numRetry = retryArg.getValue();

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

  for (auto device :deviceIds)
  {
    cudaSetDevice(device);
    checkCudaErrors(cublasInit());
  }

  testA = new MatrixType[m * n];
  testB = new MatrixType[m * p];
  testResult = new MatrixType[n * p];

  std::for_each(testA, testA + (m*n), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
  std::for_each(testB, testB + (m*p), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
  std::for_each(testResult, testResult + (n*p), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });


  std::cout << "n,m,p,blockSize,time(s),gflops" << std::endl;

  for (size_t retryNum = 0; retryNum < numRetry; ++retryNum) {
    checkCudaErrors(cudaDeviceSynchronize());

    auto begin = std::chrono::high_resolution_clock::now();

    cublasXtHandle_t handle;
    checkCudaErrors(cublasXtCreate(&handle));

    checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), deviceIds.data()));
    checkCudaErrors(cublasXtSetBlockDim(handle, blockSize));

    MatrixType
        alpha = 1.0,
        beta = 1.0;

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


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin).count();

    std::cout
        << n << "," << m << "," << p << "," << blockSize << "," << duration << "," <<
        computeMatrixMultiplicationGFLOPS(n, m, p, duration) << std::endl;
  }


//  double avgRuntime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();

//  std::cout << "Avg: " << avgRuntime << ", " << computeMatrixMultiplicationGFLOPS(n, m, p, avgRuntime);


  delete [] testA;
  delete [] testB;
  delete [] testResult;


  for (auto device :deviceIds)
  {
    cudaSetDevice(device);
    checkCudaErrors(cublasShutdown());
  }


}