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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/reorder_gradient_accumulation_pass.h"

#include <list>
#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/util.h"

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {
constexpr char kIpuCrossReplicaSum[] = "IpuCrossReplicaSum";
constexpr char kIpuReplicationNormalise[] = "IpuReplicationNormalise";
constexpr char kIpuStatefulGradientAccumulate[] =
    "IpuStatefulGradientAccumulate";

// Reorder the `node` if its input is a gradient accumulation node.
// Returns true if the input have been swapped.
StatusOr<bool> SwapIfInputGradientAccumulation(Graph* graph, Node* node) {
  const Edge* accumulator_edge;
  TF_RETURN_IF_ERROR(node->input_edge(0, &accumulator_edge));
  Node* accumulator = accumulator_edge->src();
  if (accumulator->def().op() != kIpuStatefulGradientAccumulate) {
    return false;
  }

  // We do not allow for gradient accumulation to have multiple users.
  uint64 num_edges = 0;
  for (const Edge* edge : accumulator->out_edges()) {
    if (!edge->IsControlEdge()) {
      num_edges++;
    } else if (edge->dst() == node) {
      // Remove any control edges.
      graph->RemoveEdge(edge);
    }
  }
  if (num_edges > 1) {
    return errors::FailedPrecondition(
        "Detected a %s node with multiple users, which is not supported.",
        kIpuStatefulGradientAccumulate);
  }

  VLOG(1) << "Reordering " << node->name() << " and " << accumulator->name();

  const Edge* accumulator_input_edge;
  TF_RETURN_IF_ERROR(accumulator->input_edge(0, &accumulator_input_edge));
  Node* accumulator_input = accumulator_input_edge->src();
  const int accumulator_input_idx = accumulator_input_edge->src_output();

  // Remove the old edges.
  graph->RemoveEdge(accumulator_input_edge);
  graph->RemoveEdge(accumulator_edge);

  // Move any non-control edges from the node to accumulator.
  for (const Edge* edge : node->out_edges()) {
    if (edge->IsControlEdge()) {
      continue;
    }
    CHECK_EQ(edge->src_output(), 0);
    graph->AddEdge(accumulator, 0, edge->dst(), edge->dst_input());
    graph->RemoveEdge(edge);
  }

  // Add new edges.
  graph->AddEdge(accumulator_input, accumulator_input_idx, node, 0);
  graph->AddEdge(node, 0, accumulator, 0);

  return true;
}

// Returns true if the graph, including any nested graphs, has changed.
StatusOr<bool> OptimizeGraph(Graph* graph,
                             FunctionLibraryDefinition* flib_def) {
  // Call this function for any nested loops - replace the definitions so
  // that the inner loop is optimized.
  TF_ASSIGN_OR_RETURN(bool changed, CallFunctionForWhileLoopBodies(
                                        graph, flib_def, OptimizeGraph));

  // First reorder:
  // IpuCrossReplicaSum(IpuStatefulGradientAccumulate) =>
  // IpuStatefulGradientAccumulate(IpuCrossReplicaSum)
  std::list<Node*> cross_replica_sum_nodes;
  // Find all the cross_replica_sum ops.
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kIpuCrossReplicaSum) {
      cross_replica_sum_nodes.push_back(node);
    }
  }

  for (Node* node : cross_replica_sum_nodes) {
    TF_ASSIGN_OR_RETURN(bool swapped,
                        SwapIfInputGradientAccumulation(graph, node));
    changed |= swapped;
  }

  // Second reorder:
  // IpuReplicationNormalise(IpuStatefulGradientAccumulate) =>
  // IpuStatefulGradientAccumulate(IpuReplicationNormalise)
  std::list<Node*> replica_normalise_nodes;
  // Find all the replica_normalise ops.
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kIpuReplicationNormalise) {
      replica_normalise_nodes.push_back(node);
    }
  }

  for (Node* node : replica_normalise_nodes) {
    TF_ASSIGN_OR_RETURN(bool swapped,
                        SwapIfInputGradientAccumulation(graph, node));
    changed |= swapped;
  }

  return changed;
}
}  // namespace

Status ReorderGradientAccumulationPass::Run(
    const GraphOptimizationPassOptions& options) {
  FunctionLibraryDefinition* flib_def = options.flib_def;
  TF_RET_CHECK(flib_def != nullptr);
  Graph* graph = options.graph->get();
  TF_RETURN_IF_ERROR(OptimizeGraph(graph, flib_def).status());
  return Status::OK();
}
}  // namespace tensorflow
