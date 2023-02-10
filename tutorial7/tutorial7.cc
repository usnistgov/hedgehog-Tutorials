#include <utility>

#include "graph/graph_c_int_c_int.h"
#include "graph/graph_int_int.h"
#include "task/task_c_int_c_int.h"
#include "task/task_int_int.h"
#include "task/task_int_int_with_can_terminate.h"

#if  (defined(__cpp_lib_constexpr_vector) && defined(__cpp_lib_constexpr_string))

template<class GraphType>
class TestCriticalPath : public hh_cx::AbstractTest<GraphType> {
 private:
  double_t
      maxPathValue_ = 0,
      currentPathValue_ = 0;

  hh_cx::PropertyMap<double>
      propertyMap_;

  hh_cx::Graph<GraphType> const
      *graph_ = nullptr;

  std::vector<hh_cx::behavior::AbstractNode const *>
      criticalVector_{},
      visited_{};

 public:
  constexpr explicit TestCriticalPath(hh_cx::PropertyMap<double> propertyMap)
      : hh_cx::AbstractTest<GraphType>("Critical Path"), propertyMap_(std::move(propertyMap)) {}

  constexpr ~TestCriticalPath() override = default;

  constexpr void test(hh_cx::Graph<GraphType> const *graph) override {
    graph_ = graph;

    auto const &inputNodeMap = graph->inputNodes();
    for (auto const &type : inputNodeMap.types()) {
      for (auto const &inputNode : inputNodeMap.nodes(type)) {
        this->visitNode(inputNode);
      }
    }

    if (criticalVector_.empty()) {
      this->graphValid(true);
    } else {
      this->graphValid(false);

      this->appendErrorMessage("The critical path is:\n\t");
      this->appendErrorMessage(criticalVector_.front()->name());

      for (size_t criticalNodeId = 1; criticalNodeId < criticalVector_.size(); ++criticalNodeId) {
        this->appendErrorMessage(" -> ");
        this->appendErrorMessage(criticalVector_.at(criticalNodeId)->name());
      }
    };
  }

 private:
  constexpr void visitNode(hh_cx::behavior::AbstractNode const *node) {
    if (std::find(visited_.cbegin(), visited_.cend(), node) == visited_.cend()) {
      currentPathValue_ += propertyMap_.property(node->name());
      visited_.push_back(node);

      if (std::find(visited_.cbegin(), visited_.cend(), node) != visited_.cend()) {
        if (currentPathValue_ > maxPathValue_) {
          maxPathValue_ = currentPathValue_;
          criticalVector_.clear();
          criticalVector_ = visited_;
        }
      }

      for (auto const &neighbor : graph_->adjacentNodes(node)) { visitNode(neighbor); }

      currentPathValue_ -= propertyMap_.property(node->name());
      visited_.pop_back();
    }
  }
};

constexpr auto constructGraphInt() {
  hh_cx::Node<TaskIntInt>
      node1("node1"),
      node2("node2"),
      node3("node3"),
      node4("node4"),
      node5("node5"),
      node6("node6"),
      node7("node7");

  hh_cx::Graph<GraphIntInt> graph{"Simple Graph"};

  graph.inputs(node1);
  graph.edges(node1, node2);
  graph.edges(node2, node3);
  graph.edges(node2, node5);
  graph.edges(node3, node4);
  graph.edges(node3, node5);
  graph.edges(node4, node1);
  graph.edges(node4, node7);
  graph.edges(node5, node6);
  graph.edges(node6, node2);
  graph.outputs(node7);

  auto dataRaceTest = hh_cx::DataRaceTest < GraphIntInt > {};
  auto cycleTest = hh_cx::CycleTest < GraphIntInt > {};
  graph.addTest(&dataRaceTest);
  graph.addTest(&cycleTest);
  return graph;
}

constexpr auto constructGraphWithCanTerminate() {
  hh_cx::Node<TaskIntIntWithCanTerminate>
      node1("node1"),
      node2("node2"),
      node3("node3"),
      node4("node4"),
      node5("node5"),
      node6("node6"),
      node7("node7");

  hh_cx::Graph<GraphIntInt> graph{"Simple Graph"};

  graph.inputs(node1);
  graph.edges(node1, node2);
  graph.edges(node2, node3);
  graph.edges(node2, node5);
  graph.edges(node3, node4);
  graph.edges(node3, node5);
  graph.edges(node4, node1);
  graph.edges(node4, node7);
  graph.edges(node5, node6);
  graph.edges(node6, node2);
  graph.outputs(node7);

  auto cycleTest = hh_cx::CycleTest < GraphIntInt > {};
  graph.addTest(&cycleTest);
  return graph;
}

constexpr auto constructGraphROInt() {
  hh_cx::Node<TaskIntInt, int>
      node1("node1"),
      node2("node2"),
      node3("node3"),
      node4("node4"),
      node5("node5"),
      node6("node6"),
      node7("node7");

  hh_cx::Graph<GraphIntInt> graph{"Graph with RO int"};

  graph.inputs(node1);
  graph.edges(node1, node2);
  graph.edges(node2, node3);
  graph.edges(node2, node5);
  graph.edges(node3, node4);
  graph.edges(node3, node5);
  graph.edges(node4, node1);
  graph.edges(node4, node7);
  graph.edges(node5, node6);
  graph.edges(node6, node2);
  graph.outputs(node7);

  auto dataRaceTest = hh_cx::DataRaceTest < GraphIntInt > {};
  graph.addTest(&dataRaceTest);
  return graph;
}

constexpr auto constructGraphConstInt() {
  hh_cx::Node<TaskCIntCInt>
      node1("node1"),
      node2("node2"),
      node3("node3"),
      node4("node4"),
      node5("node5"),
      node6("node6"),
      node7("node7");

  hh_cx::Graph<GraphCIntCInt> graph{"Graph with const int"};

  graph.inputs(node1);
  graph.edges(node1, node2);
  graph.edges(node2, node3);
  graph.edges(node2, node5);
  graph.edges(node3, node4);
  graph.edges(node3, node5);
  graph.edges(node4, node1);
  graph.edges(node4, node7);
  graph.edges(node5, node6);
  graph.edges(node6, node2);
  graph.outputs(node7);

  auto dataRaceTest = hh_cx::DataRaceTest < GraphCIntCInt > {};
  graph.addTest(&dataRaceTest);
  return graph;
}

constexpr auto constructGraphIntCustomTest() {
  hh_cx::Node<TaskIntInt>
      node1("node1"),
      node2("node2"),
      node3("node3"),
      node4("node4"),
      node5("node5"),
      node6("node6"),
      node7("node7");

  hh_cx::Graph<GraphIntInt> graph{"Custom test graph"};

  graph.inputs(node1);
  graph.edges(node1, node2);
  graph.edges(node2, node3);
  graph.edges(node2, node5);
  graph.edges(node3, node4);
  graph.edges(node3, node5);
  graph.edges(node4, node1);
  graph.edges(node4, node7);
  graph.edges(node5, node6);
  graph.edges(node6, node2);
  graph.outputs(node7);

  hh_cx::PropertyMap<double> propertyMap;

  propertyMap.insert("node1", 1);
  propertyMap.insert("node2", 2);
  propertyMap.insert("node3", 3);
  propertyMap.insert("node4", 4);
  propertyMap.insert("node5", 5);
  propertyMap.insert("node6", 6);
  propertyMap.insert("node7", 12);

  auto criticalPathTest = TestCriticalPath<GraphIntInt>(propertyMap);
  graph.addTest(&criticalPathTest);
  return graph;
}

int main() {
  constexpr auto defrosterInt = hh_cx::createDefroster<&constructGraphInt>();
  constexpr auto defrosterCanTerminate = hh_cx::createDefroster<&constructGraphWithCanTerminate>();
  constexpr auto defrosterROInt = hh_cx::createDefroster<&constructGraphROInt>();
  constexpr auto defrosterConstInt = hh_cx::createDefroster<&constructGraphConstInt>();
  constexpr auto defrosterCustomTest = hh_cx::createDefroster<&constructGraphIntCustomTest>();

  static_assert(!defrosterInt.isValid(), "The graph should not be valid");
  static_assert(defrosterCanTerminate.isValid(), "The graph should be valid");
  static_assert(defrosterROInt.isValid(), "The graph should be valid");
  static_assert(defrosterConstInt.isValid(), "The graph should be valid");
  static_assert(!defrosterCustomTest.isValid(), "The graph should not be valid");

  std::cout << defrosterInt.graphName() << " is Valid ? " << std::boolalpha << defrosterInt.isValid() << " -> "
            << defrosterInt.report() << "\n";
  std::cout << defrosterCanTerminate.graphName() << " is Valid ? " << std::boolalpha << defrosterCanTerminate.isValid()
            << "\n";
  std::cout << defrosterROInt.graphName() << " is Valid ? " << std::boolalpha << defrosterROInt.isValid() << "\n";
  std::cout << defrosterConstInt.graphName() << " is Valid ? " << std::boolalpha << defrosterConstInt.isValid() << "\n";

  std::cout << defrosterCustomTest.graphName() << " is Valid ? " << std::boolalpha << defrosterCustomTest.isValid()
            << " -> " << defrosterCustomTest.report() << "\n";

  if constexpr (defrosterCanTerminate.isValid()) {
    std::cout << "The graph with the canTerminate method defined is valid" << std::endl;
  } else {
    std::cout << "The graph with the canTerminate method defined is not valid" << std::endl;
  }

  if constexpr (!defrosterInt.isValid()) {
    std::cout << defrosterInt.report() << std::endl;
  } else {
    //[...]
  }

  if constexpr (!defrosterROInt.isValid()) {
    std::cout << defrosterROInt.report() << std::endl;
  } else {
    auto graph = defrosterROInt.map(
        "node1", std::make_shared<TaskIntInt>("Task1"),
        "node2", std::make_shared<TaskIntInt>("Task2"),
        "node3", std::make_shared<TaskIntInt>("Task3"),
        "node4", std::make_shared<TaskIntInt>("Task4"),
        "node5", std::make_shared<TaskIntInt>("Task5"),
        "node6", std::make_shared<TaskIntInt>("Task6"),
        "node7", std::make_shared<TaskIntInt>("Task7")
    );
    graph->createDotFile("GraphIntInt.dot");
  }

  return 0;
}

#else // (defined(__cpp_lib_constexpr_vector) && defined(__cpp_lib_constexpr_string))

int main () {
  std::cout << "You do not have a compiler with the features needed to run the compile-time analysis" << std::endl;
}

#endif // (defined(__cpp_lib_constexpr_vector) && defined(__cpp_lib_constexpr_string))