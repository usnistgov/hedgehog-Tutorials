#include <numeric>
#include <complex>
#include <memory>
#include "graph/parallel_reduction_graph.h"

double avgTimers(std::vector<long> const &timers) {
  return (double) std::accumulate(timers.cbegin(), timers.cend(), (long) 0) / (double) timers.size();
}

double stdevTimers(std::vector<long> const &timers) {
  double
      avg = avgTimers(timers),
      temp = 0;

  for (auto const &timer : timers) {
    temp += (((double) timer - avg) * ((double) timer - avg));
  }
  return std::sqrt(temp / (double) timers.size());
}

int main() {
  using DataType = int;
  auto addition = [](DataType const &lhs, DataType const &rhs) { return lhs + rhs; };

  std::cout << "VectorSize,BlockSize,AVGTimeNS,STDEVTimeNS,Exec/elemts\n";

  for (size_t blockSize = 256; blockSize <= 16777216; blockSize *= 2) {
    for (size_t vectorSize = blockSize; vectorSize <= 16777216; vectorSize *= 2) {
      auto v = std::make_shared<std::vector<DataType>>(vectorSize, 1);

      ParallelReductionGraph<DataType> graph(addition, blockSize, std::thread::hardware_concurrency());
      graph.executeGraph();

      std::vector<long> timers;
      for (size_t iteration = 0; iteration < 20; ++iteration) {
        std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now(), end;
        graph.pushData(v);
        *(std::get<std::shared_ptr<DataType>>(*graph.getBlockingResult()));
        graph.cleanGraph();
        end = std::chrono::system_clock::now();
        timers.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
      }
      std::cout << vectorSize << "," << blockSize << "," << avgTimers(timers) << "," << stdevTimers(timers) << ","
                << (avgTimers(timers) / static_cast<double>(vectorSize)) << "\n";
      graph.finishPushingData();
      graph.waitForTermination();
    }
  }
}