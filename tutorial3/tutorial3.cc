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


#include <hedgehog/hedgehog.h>
#include <random>
#include "../utils/tclap/CmdLine.h"

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"

#include "task/addition_task.h"
#include "task/product_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "state/input_block_state.h"
#include "state/output_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"

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

int main(int argc, char **argv) {
  using MatrixType = float;
  constexpr Order Ord = Order::Column;

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
      numberThreadAddition = 0;

  // Allocate matrices
  MatrixType
      *dataA = nullptr,
      *dataB = nullptr,
      *dataC = nullptr;

  try {
    TCLAP::CmdLine cmd("Matrix Multiplication parameters", ' ', "0.1");
    SizeConstraint sc;
    TCLAP::ValueArg<size_t> nArg("n", "aheight", "Matrix A Height.", false, 10, &sc);
    cmd.add(nArg);
    TCLAP::ValueArg<size_t> mArg("m", "commondimension", "Common dimension.", false, 11, &sc);
    cmd.add(mArg);
    TCLAP::ValueArg<size_t> pArg("p", "bwidth", "Matrix B Width.", false, 12, &sc);
    cmd.add(pArg);
    TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 3, &sc);
    cmd.add(blockArg);
    TCLAP::ValueArg<size_t> productArg("x", "product", "Product task's number of threads.", false, 3, &sc);
    cmd.add(productArg);
    TCLAP::ValueArg<size_t> additionArg("a", "addition", "Addiction task's number of threads.", false, 3, &sc);
    cmd.add(additionArg);

    cmd.parse(argc, argv);

    n = nArg.getValue();
    m = mArg.getValue();
    p = pArg.getValue();
    numberThreadAddition = additionArg.getValue();
    numberThreadProduct = productArg.getValue();
    blockSize = blockArg.getValue();
    if (blockSize == 0) { blockSize = 1; }
  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

  // Allocate and fill the matrices' data randomly
  dataA = new MatrixType[n * m]();
  dataB = new MatrixType[m * p]();
  dataC = new MatrixType[n * p]();

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      dataA[i * m + j] = unif(rng);
    }
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      dataB[i * p + j] = unif(rng);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      dataC[i * p + j] = unif(rng);
    }
  }
  // Wrap them to convenient object representing the matrices
  auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(n, m, blockSize, dataA);
  auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(m, p, blockSize, dataB);
  auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(n, p, blockSize, dataC);

  // Print the matrices
  std::cout << *matrixA << std::endl;
  std::cout << *matrixB << std::endl;
  std::cout << *matrixC << std::endl;

  nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1),
  mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1),
  pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

  // Graph
  auto matrixMultiplicationGraph =
      hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>>
          ("Matrix Multiplication Graph");

  // Tasks
  auto taskTraversalA =
      std::make_shared<MatrixRowTraversalTask<MatrixType, 'a', Ord>>();
  auto taskTraversalB =
      std::make_shared<MatrixColumnTraversalTask<MatrixType, 'b', Ord>>();
  auto taskTraversalC =
      std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
  auto productTask =
      std::make_shared<ProductTask<MatrixType, Ord>>(numberThreadProduct, p);
  auto additionTask =
      std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

  // State
  auto stateInputBlock =
      std::make_shared<InputBlockState<MatrixType, Ord>>(nBlocks, mBlocks, pBlocks);
  auto statePartialComputation =
      std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks);
  auto stateOutput =
      std::make_shared<OutputState<MatrixType, Ord>>(nBlocks, pBlocks, mBlocks);

  // StateManager
  auto stateManagerInputBlock =
      std::make_shared<
          hh::StateManager<
              std::pair<
                  std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>,
                  std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>>, // Pair of block as output
              MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>> // Block as Input
      >("Input State Manager", stateInputBlock);
  auto stateManagerPartialComputation =
      std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

  auto stateManagerOutputBlock =
      std::make_shared<hh::StateManager<
          MatrixBlockData<MatrixType, 'c', Ord>,
          MatrixBlockData<MatrixType, 'c', Ord>>>("Output State Manager", stateOutput);

  // Build the graph
  matrixMultiplicationGraph.input(taskTraversalA);
  matrixMultiplicationGraph.input(taskTraversalB);
  matrixMultiplicationGraph.input(taskTraversalC);
  matrixMultiplicationGraph.addEdge(taskTraversalA, stateManagerInputBlock);
  matrixMultiplicationGraph.addEdge(taskTraversalB, stateManagerInputBlock);
  matrixMultiplicationGraph.addEdge(taskTraversalC, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdge(stateManagerInputBlock, productTask);
  matrixMultiplicationGraph.addEdge(productTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdge(stateManagerPartialComputation, additionTask);
  matrixMultiplicationGraph.addEdge(additionTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdge(additionTask, stateManagerOutputBlock);
  matrixMultiplicationGraph.output(stateManagerOutputBlock);

  // Execute the graph
  matrixMultiplicationGraph.executeGraph();

  // Push the matrices
  matrixMultiplicationGraph.pushData(matrixA);
  matrixMultiplicationGraph.pushData(matrixB);
  matrixMultiplicationGraph.pushData(matrixC);

  // Notify push done
  matrixMultiplicationGraph.finishPushingData();

  // Wait for the graph to terminate
  matrixMultiplicationGraph.waitForTermination();

  //Print the result matrix
  std::cout << *matrixC << std::endl;

  matrixMultiplicationGraph.createDotFile("Tutorial3MatrixMultiplicationCycle.dot", hh::ColorScheme::EXECUTION,
                                          hh::StructureOptions::NONE);

  // Deallocate the Matrices
  delete[] dataA;
  delete[] dataB;
  delete[] dataC;

  return 0;
}