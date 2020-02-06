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


#include <random>
#include <hedgehog/hedgehog.h>
#include "../utils/tclap/CmdLine.h"
#include "task/hadamard_product.h"
#include "data/triplet_matrix_block_data.h"
#include "data/matrix_data.h"

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
  using MatrixType = int;
  constexpr Order Ord = Order::Row;

  // Mersenne Twister Random Generator
  uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
  std::mt19937_64 rng(ss);

  // Choose your distribution depending on the type of MatrixType
//  std::uniform_real_distribution<MatrixType> unif(0, 10);
  std::uniform_int_distribution<MatrixType> unif(0, 10);

  // Thread count
  size_t
      numberThreadHadamard = 0,
      matrixHeight = 0,
      matrixWidth = 0,
      blockSize = 0,
      numberBlocksRows = 0,
      numberBlocksCols = 0;

  // Allocate matrices
  MatrixType
      *dataA = nullptr,
      *dataB = nullptr,
      *dataC = nullptr;

  // Parse the command line arguments
  try {
    TCLAP::CmdLine cmd("Hadamard product parameters", ' ', "0.1");
    SizeConstraint sc;
    TCLAP::ValueArg<size_t> nArg("n", "height", "Matrix Height.", false, 10, &sc);
    cmd.add(nArg);
    TCLAP::ValueArg<size_t> mArg("m", "width", "Matrix Width.", false, 10, &sc);
    cmd.add(mArg);
    TCLAP::ValueArg<size_t> bArg("b", "block", "Block Size.", false, 3, &sc);
    cmd.add(bArg);
    TCLAP::ValueArg<size_t> tArg("t", "threads", "Number of threads for the Hadamard Task.", false, 3, &sc);
    cmd.add(tArg);
    cmd.parse(argc, argv);
    matrixHeight = nArg.getValue();
    matrixWidth = mArg.getValue();
    numberThreadHadamard = tArg.getValue();
    blockSize = bArg.getValue();
  } catch (TCLAP::ArgException &e)  // catch any exceptions
  { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }


  // Allocate and fill the matrices' data randomly
  dataA = new MatrixType[matrixHeight * matrixWidth]();
  dataB = new MatrixType[matrixHeight * matrixWidth]();
  dataC = new MatrixType[matrixHeight * matrixWidth]();

  for(size_t i = 0; i < matrixHeight; ++i){
    for(size_t j = 0; j < matrixWidth; ++j) {
      dataA[i * matrixWidth + j] = unif(rng);
      dataB[i * matrixWidth + j] = unif(rng);
    }
  }

  // Wrap them to convenient object representing the matrices
  MatrixData<MatrixType, 'a', Ord> matrixA(matrixHeight, matrixWidth, blockSize, dataA);
  MatrixData<MatrixType, 'b', Ord> matrixB(matrixHeight, matrixWidth, blockSize, dataB);
  MatrixData<MatrixType, 'c', Ord> matrixC(matrixHeight, matrixWidth, blockSize, dataC);


  // Print the matrices
  std::cout << matrixA << std::endl;
  std::cout << matrixB << std::endl;

  // Declaring and instantiating the graph
  hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>, TripletMatrixBlockData<MatrixType, Ord>>
      graphHadamard("Tutorial 1 : Hadamard Product");

  // Declaring and instantiating the hadamard task
  auto hadamardProduct = std::make_shared<HadamardProduct<MatrixType, Ord>>("Hadamard Product", numberThreadHadamard);

  // Set The hadamard task as the task that will be connected to the graph inputs
  graphHadamard.input(hadamardProduct);

  // Set The hadamard task as the task that will be connected to the graph output
  graphHadamard.output(hadamardProduct);

  // Execute the graph
  graphHadamard.executeGraph();

  numberBlocksRows = matrixA.numBlocksRows();
  numberBlocksCols = matrixA.numBlocksCols();

  // Create the triplets of blocks and send them to the graph
  for (size_t iGrid = 0; iGrid < numberBlocksRows; ++iGrid) {
    for (size_t jGrid = 0; jGrid < numberBlocksCols; ++jGrid) {
      auto triplet = std::make_shared<TripletMatrixBlockData<MatrixType, Ord>>();
      triplet->a_ = std::make_shared<MatrixBlockData<MatrixType, 'a', Ord>>(iGrid, jGrid, matrixA);
      triplet->b_ = std::make_shared<MatrixBlockData<MatrixType, 'b', Ord>>(iGrid, jGrid, matrixB);
      triplet->c_ = std::make_shared<MatrixBlockData<MatrixType, 'c', Ord>>(iGrid, jGrid, matrixC);
      graphHadamard.pushData(triplet);
    }
  }

  // Notify the graph that no more data will be sent
  graphHadamard.finishPushingData();

  // Loop over the different resulting block of C here we are just interested by the final result
//  while (auto blockResults = graphHadamard.getBlockingResult()) {std::cout << *blockResult << std::endl;}

  // Wait for everything to be processed
  graphHadamard.waitForTermination();

  //Print the result matrix
  std::cout << matrixC << std::endl;

  graphHadamard.createDotFile(
      "Tutorial1HadamardProduct.dot",
      hh::ColorScheme::EXECUTION,
      hh::StructureOptions::ALL);

// Deallocate the Matrices
  delete[] dataA;
  delete[] dataB;
  delete[] dataC;
}