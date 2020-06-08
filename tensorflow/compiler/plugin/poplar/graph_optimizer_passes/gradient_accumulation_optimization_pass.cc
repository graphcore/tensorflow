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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/gradient_accumulation_optimization_pass.h"

#include <list>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

constexpr char kIpuStatefulGradientAccumulate[] =
    "IpuStatefulGradientAccumulate";
constexpr char kIpuStatefulGradientAccumulateWithMomentum[] =
    "IpuStatefulGradientAccumulateWithMomentum";
constexpr char kResourceApplyMomentum[] = "ResourceApplyMomentum";
constexpr char kResourceApplyGradientDescent[] = "ResourceApplyGradientDescent";
// Momentum operands.
constexpr int kMomentumVariableOperand = 0;
constexpr int kMomentumAccumOperand = 1;
constexpr int kMomentumLROperand = 2;
constexpr int kMomentumGradOperand = 3;
constexpr int kMomentumMomentumOperand = 4;
// GD operands.
constexpr int kGDVariableOperand = 0;
constexpr int kGDLROperand = 1;
constexpr int kGDGradOperand = 2;

// Returns true if gradient accumulation was converted to a momentum specific
// op.
StatusOr<bool> ProcessMomentumOp(Graph* graph, Node* node) {
  // Check whether the gradient is a gradient accumulation op.
  const Edge* accumulator_edge;
  TF_RETURN_IF_ERROR(node->input_edge(kMomentumGradOperand, &accumulator_edge));
  Node* accumulator = accumulator_edge->src();
  if (accumulator->def().op() != kIpuStatefulGradientAccumulate) {
    return false;
  }

  bool use_nesterov;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "use_nesterov", &use_nesterov));
  if (use_nesterov) {
    return errors::FailedPrecondition(
        "Nesterov is not supported with gradient accumulation.");
  }

  // If the accumulator node has more than one non control edge, then we abort.
  uint64 num_edges = 0;
  for (const Edge* edge : accumulator->out_edges()) {
    if (!edge->IsControlEdge()) {
      num_edges++;
    }
    if (edge->dst() == node && edge->IsControlEdge()) {
      // Remove any control edges.
      graph->RemoveEdge(edge);
    }
  }

  if (num_edges > 1) {
    return errors::FailedPrecondition(
        "Detected a %s node with multiple users, which is not supported. This "
        "usually occurs when the `GradientAccumulationOptimizer` is used with "
        "another optimizer which is not supported. Please note that "
        "`GradientAccumulationOptimizer` is currently only supported with "
        "`GradientDescentOptimizer` and `MomentumOptimizer` optimizers. For "
        "any other optimizers use `GradientAccumulationOptimizerV2`.",
        kIpuStatefulGradientAccumulate);
  }

  VLOG(1) << "Found a momentum node with a gradient accumulation input.";

  // Get all the inputs.
  const Edge* var_edge;
  TF_RETURN_IF_ERROR(node->input_edge(kMomentumVariableOperand, &var_edge));
  Node* var = var_edge->src();
  int var_output_idx = var_edge->src_output();

  const Edge* accum_edge;
  TF_RETURN_IF_ERROR(node->input_edge(kMomentumAccumOperand, &accum_edge));
  Node* accum = accum_edge->src();
  int accum_output_idx = accum_edge->src_output();

  const Edge* lr_edge;
  TF_RETURN_IF_ERROR(node->input_edge(kMomentumLROperand, &lr_edge));
  Node* lr = lr_edge->src();
  int lr_output_idx = lr_edge->src_output();

  const Edge* momentum_edge;
  TF_RETURN_IF_ERROR(
      node->input_edge(kMomentumMomentumOperand, &momentum_edge));
  Node* momentum = momentum_edge->src();
  int momentum_output_idx = momentum_edge->src_output();

  // Get the input to the accumulator - the actual gradient.
  const Edge* gradient_edge;
  TF_RETURN_IF_ERROR(accumulator->input_edge(0, &gradient_edge));
  Node* gradient = gradient_edge->src();
  int gradient_output_idx = gradient_edge->src_output();

  // Remove the old accumulator op and replace it with a momentum version of it.
  graph->RemoveEdge(gradient_edge);
  graph->RemoveEdge(accumulator_edge);

  // Create the new accumulator op.
  int num_mini_batches;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(accumulator->attrs(), "num_mini_batches", &num_mini_batches));
  DataType accumulator_dtype;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(accumulator->attrs(), "dtype", &accumulator_dtype));
  Node* new_accumulator;
  TF_RETURN_IF_ERROR(NodeBuilder(accumulator->name(),
                                 kIpuStatefulGradientAccumulateWithMomentum)
                         .Input(accum, accum_output_idx)
                         .Input(gradient, gradient_output_idx)
                         .Input(momentum, momentum_output_idx)
                         .Attr("dtype", accumulator_dtype)
                         .Attr("num_mini_batches", num_mini_batches)
                         .Finalize(graph, &new_accumulator));

  TF_RETURN_IF_ERROR(CopyXLAAttributesIfPresent(accumulator, new_accumulator));
  // Move all the control edges from the old accumulator to the new accumulator
  // node.
  for (const Edge* edge : accumulator->out_edges()) {
    graph->AddEdge(new_accumulator, edge->src_output(), edge->dst(),
                   edge->dst_input());
    graph->RemoveEdge(edge);
  }
  graph->RemoveNode(accumulator);

  // Record original momentum op output edges and remove them first. This is
  // to avoid multiple producers for dst nodes' input.
  std::vector<OutEdgeInfo> out_edge_info;
  std::vector<const Edge*> out_edges;
  for (const Edge* edge : node->out_edges()) {
    // Resource apply have no outputs, but they can have control edges.
    CHECK(edge->IsControlEdge());
    out_edges.push_back(edge);
    out_edge_info.push_back(
        {edge->dst(), edge->src_output(), edge->dst_input()});
  }

  for (const Edge* edge : out_edges) {
    graph->RemoveEdge(edge);
  }

  // Create a gradient descent node instead of momentum.
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
  bool use_locking;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "use_locking", &use_locking));
  Node* gd;
  TF_RETURN_IF_ERROR(NodeBuilder(node->name(), kResourceApplyGradientDescent)
                         .Input(var, var_output_idx)
                         .Input(lr, lr_output_idx)
                         .Input(new_accumulator, 0)
                         .Attr("T", dtype)
                         .Attr("use_locking", use_locking)
                         .Finalize(graph, &gd));

  // Add output edges.
  for (const OutEdgeInfo& out_edge : out_edge_info) {
    graph->AddEdge(gd, out_edge.src_output, out_edge.dst, out_edge.dst_input);
  }

  TF_RETURN_IF_ERROR(CopyXLAAttributesIfPresent(node, gd));
  // Remove the original node.
  graph->RemoveNode(node);

  return true;
}

// Returns true if the graph, including any nested graphs, has changed.
StatusOr<bool> OptimizeGraph(Graph* graph,
                             FunctionLibraryDefinition* flib_def) {
  // Call this function for any nested loops - replace the definitions so
  // that the inner loop is optimized.
  TF_ASSIGN_OR_RETURN(bool changed, CallFunctionForWhileLoopBodies(
                                        graph, flib_def, OptimizeGraph));

  std::list<Node*> momentum_nodes;
  // Find all the momentum ops.
  for (Node* node : graph->nodes()) {
    if (node->def().op() == kResourceApplyMomentum) {
      momentum_nodes.push_back(node);
    }
  }

  // Try and replace them.
  for (Node* node : momentum_nodes) {
    TF_ASSIGN_OR_RETURN(bool optimized, ProcessMomentumOp(graph, node));
    changed |= optimized;
  }
  return changed;
}
}  // namespace

Status GradientAccumulationOptimizationPass::Run(
    const GraphOptimizationPassOptions& options) {
  FunctionLibraryDefinition* flib_def = options.flib_def;
  TF_RET_CHECK(flib_def != nullptr);
  Graph* graph = options.graph->get();
  TF_RETURN_IF_ERROR(OptimizeGraph(graph, flib_def).status());
  return Status::OK();
}

}  // namespace tensorflow
