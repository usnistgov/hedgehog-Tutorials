
#ifndef NEW_HEDGEHOG_TUTORIALS_GRAPH_INT_INT_H_
#define NEW_HEDGEHOG_TUTORIALS_GRAPH_INT_INT_H_

#include <hedgehog/hedgehog.h>

class GraphIntInt : public hh::Graph<1, int, int> {
 public:
  explicit GraphIntInt(std::string const &name) : hh::Graph<1, int, int>(name) {}
};

#endif //NEW_HEDGEHOG_TUTORIALS_GRAPH_INT_INT_H_
