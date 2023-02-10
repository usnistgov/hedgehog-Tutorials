
#ifndef NEW_HEDGEHOG_TUTORIALS_GRAPH_C_INT_C_INT_H_
#define NEW_HEDGEHOG_TUTORIALS_GRAPH_C_INT_C_INT_H_

#include <hedgehog/hedgehog.h>

class GraphCIntCInt : public hh::Graph<1, int const, int const> {
 public:
  explicit GraphCIntCInt(std::string const &name) : hh::Graph<1, int const, int const>(name) {}
};

#endif //NEW_HEDGEHOG_TUTORIALS_GRAPH_C_INT_C_INT_H_
