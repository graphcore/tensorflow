/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_META_GRAPH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_META_GRAPH_H_

#include <functional>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "absl/types/optional.h"

namespace xla {
namespace poplarplugin {

template <typename T, typename Comparator = std::less<T>>
class MetaGraph {
 public:
  template <typename ValueT>
  using MetaGraphMap = std::map<T, ValueT, Comparator>;

  using MetaGraphSet = std::set<T, Comparator>;

 private:
  using Graph = std::map<T, MetaGraphSet, Comparator>;

  MetaGraph(){};

  template <typename Predicate>
  MetaGraphSet FindConsumers(T node, Predicate&& pred, bool inclusive,
                             MetaGraphSet& visited) const {
    MetaGraphSet consumers;

    const auto itr = graph_.find(node);
    if (itr != graph_.end()) {
      for (const auto& neighbour : itr->second) {
        if (inclusive) {
          consumers.insert(neighbour);
        }
        const bool already_visited = visited.count(neighbour);
        if (pred(neighbour) && !already_visited) {
          consumers.insert(neighbour);
          visited.insert(neighbour);
          MetaGraphSet neighbour_consumers =
              FindConsumers(neighbour, pred, inclusive, visited);
          consumers.insert(neighbour_consumers.begin(),
                           neighbour_consumers.end());
        }
      }
    }

    return consumers;
  }

  Graph graph_;

 public:
  template <typename NodeIt>
  MetaGraph(std::vector<T> root_nodes, NodeIt node_iterator_getter) {
    // DF traversal to create the initial graph.
    std::stack<T> to_visit;
    MetaGraphSet visited;
    for (T root_node : root_nodes) {
      to_visit.push(root_node);
    }

    while (!to_visit.empty()) {
      // Get the current node
      T current = to_visit.top();
      to_visit.pop();

      if (visited.count(current) != 0) {
        continue;
      }
      visited.insert(current);

      for (T operand : node_iterator_getter(current)) {
        graph_[operand].insert(current);
        to_visit.push(operand);
      }
    }
  };

  template <typename NodeIt>
  MetaGraph(T root_node, NodeIt node_iterator_getter)
      : MetaGraph(std::vector<T>({root_node}), node_iterator_getter) {}

  template <typename InputIt, typename NodeValueGetter>
  MetaGraph(InputIt input_it, NodeValueGetter node_value_getter) {
    for (T input : input_it) {
      graph_[input] = node_value_getter(input);
    }
  };

  MetaGraph Transpose() const {
    MetaGraph<T, Comparator> result;

    for (auto& edge : graph_) {
      for (auto v2 : edge.second) {
        result[v2].insert(edge.first);
      }
    }

    return result;
  }

  MetaGraphSet GetVertices() const {
    MetaGraphSet result;

    for (auto pair : graph_) {
      result.insert(pair.first);
      result.insert(pair.second.begin(), pair.second.end());
    }

    return result;
  }

  template <typename Predicate>
  MetaGraphSet FindConsumers(T node, Predicate&& pred,
                             bool inclusive = false) const {
    // FindConsumers is a depth first traversal - this is a wrapper for it where
    // we create a set of visited nodes to prevent getting stuck in cycles.
    MetaGraphSet visited;
    return FindConsumers(node, pred, inclusive, visited);
  }

  template <typename Predicate>
  MetaGraphSet FindVertices(Predicate pred) const {
    MetaGraphSet result;

    for (const auto& v : GetVertices()) {
      if (pred(v)) {
        result.insert(v);
      }
    }

    return result;
  }

  template <typename Predicate>
  static bool IsPathOk(const std::vector<T>& path, Predicate pred) {
    for (unsigned i = 0; i < path.size(); i++) {
      T node = path[i];
      if (!pred(node, i, path.size())) {
        return false;
      }
    }
    return true;
  };

  struct ShortestPaths {
    absl::optional<std::vector<T>> To(T dst) const {
      if (prev.count(dst) == 0) {
        return absl::nullopt;
      }

      std::vector<T> path = {dst};
      while (path.back() != src) {
        path.push_back(prev.at(path.back()));
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    T src;
    MetaGraphMap<T> prev;
  };

  ShortestPaths ShortestPathsFrom(T src) const {
    // Use a breadth-first search from the source to all targets. As this is an
    // unweighted graph, the path discovered first is the shortest one.

    MetaGraphMap<T> prev;
    std::queue<T> queue;
    queue.push(src);

    while (!queue.empty()) {
      const auto u = queue.front();
      queue.pop();

      const auto itr = graph_.find(u);
      if (itr != graph_.end()) {
        for (const auto& v : itr->second) {
          if (v != src && prev.count(v) == 0) {
            queue.push(v);
            prev[v] = u;
          }
        }
      }
    }

    return ShortestPaths{src, std::move(prev)};
  }

  absl::optional<std::vector<T>> ShortestPath(T src, T dst) const {
    return ShortestPathsFrom(src).To(dst);
  }

  MetaGraphSet& operator[](T& key) { return graph_[key]; }

  const MetaGraphSet& operator[](const T key) const { return graph_.at(key); }

  bool contains(T key) const { return graph_.find(key) != graph_.end(); }

  typename Graph::const_iterator begin() const { return graph_.begin(); }

  typename Graph::const_iterator end() const { return graph_.end(); }

  typename Graph::const_iterator find(T key) const { return graph_.find(key); }
};

}  // namespace poplarplugin
}  // namespace xla
#endif
