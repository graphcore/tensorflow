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

#include <memory>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/util.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

constexpr char kIpuCrossReplicaSum[] = "IpuCrossReplicaSum";
constexpr char kIpuReplicationNormalise[] = "IpuReplicationNormalise";
constexpr char kIpuStatefulGradientAccumulate[] =
    "IpuStatefulGradientAccumulate";
constexpr char kRetval[] = "_Retval";

Status BuildGraph(Graph* g) {
  Scope scope = Scope::NewRootScope();
  Output grad = ops::Const(scope.WithOpName("grad"), 10.0f, {1});
  return scope.ToGraph(g);
}

Status BuildGraphForWhileLoop(Graph* main_graph, FunctionDefLibrary* fdl,
                              std::function<Status(Graph*)> while_body_fn) {
  {
    // Function for While's "body".
    // Create the single argument.
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("grad"), DT_FLOAT, 0);
    // auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg0, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));

    // Call the function which will add ops to the graph.
    TF_CHECK_OK(while_body_fn(g.get()));

    FunctionDef* fdef = fdl->add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "body", fdef));
  }
  {
    // Function for While's "cond".
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("grad"), DT_FLOAT, 0);
    Output c = ops::Const(s.WithOpName("c"), true, {});
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), c, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef* fdef = fdl->add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cond", fdef));
  }

  // Build the while loop.
  Scope scope = Scope::NewRootScope();
  Output grad = ops::Const(scope.WithOpName("grad"), 10.0f, {1});
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op =
      ops::While(scope.WithOpName("while"), std::initializer_list<Input>{grad},
                 cond_fn, body_fn);
  return scope.ToGraph(main_graph);
}

Status RunPass(std::unique_ptr<Graph>* graph,
               FunctionLibraryDefinition* flib_def) {
  // Assign all nodes to the IPU device.
  static const char* kIPUDevice = "/job:localhost/replica:0/task:0/IPU:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kIPUDevice);
    }
  }

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.flib_def = flib_def;
  SessionOptions session_options;
  session_options.env = Env::Default();
  opt_options.session_options = &session_options;
  ReorderGradientAccumulationPass pass;
  TF_RETURN_IF_ERROR(pass.Run(opt_options));
  return Status::OK();
}

TEST(ReorderGradientAccumulationPass, CrossReplicaAndGradientAccumulation) {
  FunctionDefLibrary fdl;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(BuildGraph(g.get()));

  {
    auto node_index = g->BuildNodeNameIndex();
    Node* grad_accum;
    TF_CHECK_OK(NodeBuilder(kIpuStatefulGradientAccumulate,
                            kIpuStatefulGradientAccumulate)
                    .Input(node_index["grad"], 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Attr("num_mini_batches", 5)
                    .Finalize(g.get(), &grad_accum));
    Node* cross_replica_sum;
    TF_CHECK_OK(NodeBuilder(kIpuCrossReplicaSum, kIpuCrossReplicaSum)
                    .Input(grad_accum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(g.get(), &cross_replica_sum));
  }

  uint64 count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    return input_node->def().op() == kIpuCrossReplicaSum
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

TEST(ReorderGradientAccumulationPass, ReplicaNormaliseAndGradientAccumulation) {
  FunctionDefLibrary fdl;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(BuildGraph(g.get()));

  {
    auto node_index = g->BuildNodeNameIndex();
    Node* grad_accum;
    TF_CHECK_OK(NodeBuilder(kIpuStatefulGradientAccumulate,
                            kIpuStatefulGradientAccumulate)
                    .Input(node_index["grad"], 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Attr("num_mini_batches", 5)
                    .Finalize(g.get(), &grad_accum));
    Node* replica_normalise;
    TF_CHECK_OK(NodeBuilder(kIpuReplicationNormalise, kIpuReplicationNormalise)
                    .Input(grad_accum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(g.get(), &replica_normalise));
  }

  uint64 count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    return input_node->def().op() == kIpuReplicationNormalise
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

TEST(ReorderGradientAccumulationPass,
     CrossReplicaAndReplicaNormaliseAndGradientAccumulation) {
  FunctionDefLibrary fdl;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(BuildGraph(g.get()));

  {
    auto node_index = g->BuildNodeNameIndex();
    Node* grad_accum;
    TF_CHECK_OK(NodeBuilder(kIpuStatefulGradientAccumulate,
                            kIpuStatefulGradientAccumulate)
                    .Input(node_index["grad"], 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Attr("num_mini_batches", 5)
                    .Finalize(g.get(), &grad_accum));
    Node* cross_replica_sum;
    TF_CHECK_OK(NodeBuilder(kIpuCrossReplicaSum, kIpuCrossReplicaSum)
                    .Input(grad_accum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(g.get(), &cross_replica_sum));
    Node* replica_normalise;
    TF_CHECK_OK(NodeBuilder(kIpuReplicationNormalise, kIpuReplicationNormalise)
                    .Input(cross_replica_sum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(g.get(), &replica_normalise));
  }

  uint64 count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    if (input_node->def().op() != kIpuReplicationNormalise) {
      return errors::FailedPrecondition("Not a match.");
    }
    const Node* input_input_node;
    TF_RETURN_IF_ERROR(input_node->input_node(0, &input_input_node));
    return input_input_node->def().op() == kIpuCrossReplicaSum
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

TEST(ReorderGradientAccumulationPass,
     CrossReplicaAndGradientAccumulationInLoop) {
  FunctionDefLibrary fdl;
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));

  auto body_fn = [](Graph* graph) -> Status {
    auto node_index = graph->BuildNodeNameIndex();
    Node* grad_accum;
    TF_RETURN_IF_ERROR(NodeBuilder(kIpuStatefulGradientAccumulate,
                                   kIpuStatefulGradientAccumulate)
                           .Input(node_index["grad"], 0)
                           .Attr("dtype", DataType::DT_FLOAT)
                           .Attr("num_mini_batches", 5)
                           .Finalize(graph, &grad_accum));
    Node* cross_replica_sum;
    TF_RETURN_IF_ERROR(NodeBuilder(kIpuCrossReplicaSum, kIpuCrossReplicaSum)
                           .Input(grad_accum, 0)
                           .Attr("dtype", DataType::DT_FLOAT)
                           .Finalize(graph, &cross_replica_sum));
    Node* ret_val;
    TF_RETURN_IF_ERROR(NodeBuilder(kRetval, kRetval)
                           .Input(cross_replica_sum, 0)
                           .Attr("T", DataType::DT_FLOAT)
                           .Attr("index", 0)
                           .Finalize(graph, &ret_val));
    return Status::OK();
  };

  TF_CHECK_OK(BuildGraphForWhileLoop(g.get(), &fdl, std::move(body_fn)));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);

  uint64 count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    return input_node->def().op() == kIpuCrossReplicaSum
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

TEST(ReorderGradientAccumulationPass,
     ReplicaNormaliseAndGradientAccumulationInLoop) {
  FunctionDefLibrary fdl;
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));

  auto body_fn = [](Graph* graph) -> Status {
    auto node_index = graph->BuildNodeNameIndex();
    Node* grad_accum;
    TF_RETURN_IF_ERROR(NodeBuilder(kIpuStatefulGradientAccumulate,
                                   kIpuStatefulGradientAccumulate)
                           .Input(node_index["grad"], 0)
                           .Attr("dtype", DataType::DT_FLOAT)
                           .Attr("num_mini_batches", 5)
                           .Finalize(graph, &grad_accum));
    Node* replica_normalise;
    TF_CHECK_OK(NodeBuilder(kIpuReplicationNormalise, kIpuReplicationNormalise)
                    .Input(grad_accum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(graph, &replica_normalise));
    Node* ret_val;
    TF_RETURN_IF_ERROR(NodeBuilder(kRetval, kRetval)
                           .Input(replica_normalise, 0)
                           .Attr("T", DataType::DT_FLOAT)
                           .Attr("index", 0)
                           .Finalize(graph, &ret_val));
    return Status::OK();
  };

  TF_CHECK_OK(BuildGraphForWhileLoop(g.get(), &fdl, std::move(body_fn)));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);

  uint64 count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    return input_node->def().op() == kIpuReplicationNormalise
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

TEST(ReorderGradientAccumulationPass,
     CrossReplicaAndReplicaNormaliseAndGradientAccumulationInLoop) {
  FunctionDefLibrary fdl;
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));

  auto body_fn = [](Graph* graph) -> Status {
    auto node_index = graph->BuildNodeNameIndex();
    Node* grad_accum;
    TF_CHECK_OK(NodeBuilder(kIpuStatefulGradientAccumulate,
                            kIpuStatefulGradientAccumulate)
                    .Input(node_index["grad"], 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Attr("num_mini_batches", 5)
                    .Finalize(graph, &grad_accum));
    Node* cross_replica_sum;
    TF_CHECK_OK(NodeBuilder(kIpuCrossReplicaSum, kIpuCrossReplicaSum)
                    .Input(grad_accum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(graph, &cross_replica_sum));
    Node* replica_normalise;
    TF_CHECK_OK(NodeBuilder(kIpuReplicationNormalise, kIpuReplicationNormalise)
                    .Input(cross_replica_sum, 0)
                    .Attr("dtype", DataType::DT_FLOAT)
                    .Finalize(graph, &replica_normalise));
    Node* ret_val;
    TF_RETURN_IF_ERROR(NodeBuilder(kRetval, kRetval)
                           .Input(replica_normalise, 0)
                           .Attr("T", DataType::DT_FLOAT)
                           .Attr("index", 0)
                           .Finalize(graph, &ret_val));
    return Status::OK();
  };

  TF_CHECK_OK(BuildGraphForWhileLoop(g.get(), &fdl, std::move(body_fn)));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdl);

  uint64 count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  TF_CHECK_OK(RunPass(&g, &flib_def));

  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuReplicationNormalise, g.get(), &flib_def,
                                   &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(
      CountNodesWithOpName(kIpuCrossReplicaSum, g.get(), &flib_def, &count));
  EXPECT_EQ(count, 1);
  count = 0;
  TF_CHECK_OK(CountNodesWithOpName(kIpuStatefulGradientAccumulate, g.get(),
                                   &flib_def, &count));
  EXPECT_EQ(count, 1);

  // Check that each gradient accumulate has cross replica sum as input.
  auto fn = [](Node* node) -> Status {
    const Node* input_node;
    TF_RETURN_IF_ERROR(node->input_node(0, &input_node));
    if (input_node->def().op() != kIpuReplicationNormalise) {
      return errors::FailedPrecondition("Not a match.");
    }
    const Node* input_input_node;
    TF_RETURN_IF_ERROR(input_node->input_node(0, &input_input_node));
    return input_input_node->def().op() == kIpuCrossReplicaSum
               ? Status::OK()
               : errors::FailedPrecondition("Not a match.");
  };
  TF_CHECK_OK(CallFunctionForEachNodeWithOpName(
      kIpuStatefulGradientAccumulate, g.get(), &flib_def, std::move(fn)));
}

}  // namespace
}  // namespace tensorflow
