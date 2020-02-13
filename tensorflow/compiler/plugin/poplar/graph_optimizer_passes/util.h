/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_GRAPH_OPTIMIZER_PASSES_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_GRAPH_OPTIMIZER_PASSES_UTIL_H_

#include <functional>
#include <string>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using stream_executor::port::StatusOr;

namespace tensorflow {
class FunctionLibraryDefinition;

Status CopyXLAAttributesIfPresent(Node* from, Node* to);

// Traverse the given graph for any while loops, and call the `fn` on the body
// of the loop. Changes the body definition of any while loop if `fn` returns
// true. Returns true if the definition of any while loop has changed.
StatusOr<bool> CallFunctionForWhileLoopBodies(
    Graph* graph, FunctionLibraryDefinition* flib_def,
    std::function<StatusOr<bool>(Graph*, FunctionLibraryDefinition*)> fn);

// Functions used for testing.
// Count how many nodes there are with a given op name, including traversing
// any While ops.
Status CountNodesWithOpName(const string& name, Graph* graph,
                            FunctionLibraryDefinition* flib_def, uint64* count);

// Call `fn` for each node with a given op name, including traversing any While
// ops.
Status CallFunctionForEachNodeWithOpName(const string& name, Graph* graph,
                                         FunctionLibraryDefinition* flib_def,
                                         std::function<Status(Node*)> fn);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_GRAPH_OPTIMIZER_PASSES_UTIL_H_
