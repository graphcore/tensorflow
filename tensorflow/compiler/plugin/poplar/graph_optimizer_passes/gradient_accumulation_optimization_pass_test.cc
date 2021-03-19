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

#include <memory>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

constexpr char kIpuStatefulGradientAccumulate[] =
    "IpuStatefulGradientAccumulate";
constexpr char kIpuStatefulGradientAccumulateWithMomentum[] =
    "IpuStatefulGradientAccumulateWithMomentum";
constexpr char kResourceApplyMomentum[] = "ResourceApplyMomentum";
constexpr char kResourceApplyGradientDescent[] = "ResourceApplyGradientDescent";

Status BuildGraph(Graph* g) {
  Scope scope = Scope::NewRootScope();
  {
    Output momentum = ops::Const(scope.WithOpName("momentum"), 10.0f, {});
    Output lr = ops::Const(scope.WithOpName("lr"), 10.0f, {});
    Output grad = ops::Const(scope.WithOpName("grad"), 10.0f, {1});

    Output var =
        ops::VarHandleOp(scope.WithOpName("var"), DataType::DT_FLOAT, {1});
    Output accum =
        ops::VarHandleOp(scope.WithOpName("accum"), DataType::DT_FLOAT, {1});
  }
  return scope.ToGraph(g);
}

Status RunPass(std::unique_ptr<Graph>* graph, bool* has_momentum_node,
               bool* has_gd_node, bool* has_grad_accum_node,
               bool* has_grad_accum_with_momentum_node) {
  // Assign all nodes to the IPU device.
  static const char* kIPUDevice = "/job:localhost/replica:0/task:0/IPU:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kIPUDevice);
    }
  }

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  opt_options.flib_def = &flib_def;
  SessionOptions session_options;
  session_options.env = Env::Default();
  opt_options.session_options = &session_options;
  GradientAccumulationOptimizationPass pass;
  TF_RETURN_IF_ERROR(pass.Run(opt_options));

  for (Node* node : (*graph)->nodes()) {
    if (node->def().op() == kIpuStatefulGradientAccumulate) {
      *has_grad_accum_node = true;
    } else if (node->def().op() == kIpuStatefulGradientAccumulateWithMomentum) {
      *has_grad_accum_with_momentum_node = true;
    } else if (node->def().op() == kResourceApplyMomentum) {
      *has_momentum_node = true;
    } else if (node->def().op() == kResourceApplyGradientDescent) {
      *has_gd_node = true;
    }
  }

  return Status::OK();
}

TEST(GradientAccumulationOptimizationPass, NoIPUGradAccumulator) {
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(BuildGraph(g.get()));

  {
    auto node_index = g->BuildNodeNameIndex();
    Node* momentum;
    TF_CHECK_OK(NodeBuilder(kResourceApplyMomentum, kResourceApplyMomentum)
                    .Input(node_index["var"], 0)
                    .Input(node_index["accum"], 0)
                    .Input(node_index["lr"], 0)
                    .Input(node_index["grad"], 0)
                    .Input(node_index["momentum"], 0)
                    .Attr("T", DataType::DT_FLOAT)
                    .Finalize(g.get(), &momentum));
  }

  bool has_momentum_node = false;
  bool has_gd_node = false;
  bool has_grad_accum_node = false;
  bool has_grad_accum_with_momentum_node = false;
  TF_CHECK_OK(RunPass(&g, &has_momentum_node, &has_gd_node,
                      &has_grad_accum_node,
                      &has_grad_accum_with_momentum_node));

  EXPECT_TRUE(has_momentum_node);
  EXPECT_FALSE(has_gd_node);
  EXPECT_FALSE(has_grad_accum_node);
  EXPECT_FALSE(has_grad_accum_with_momentum_node);
}

TEST(GradientAccumulationOptimizationPass, WithIPUGradAccumulator) {
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
    Node* momentum;
    TF_CHECK_OK(NodeBuilder(kResourceApplyMomentum, kResourceApplyMomentum)
                    .Input(node_index["var"], 0)
                    .Input(node_index["accum"], 0)
                    .Input(node_index["lr"], 0)
                    .Input(grad_accum, 0)
                    .Input(node_index["momentum"], 0)
                    .Attr("T", DataType::DT_FLOAT)
                    .Finalize(g.get(), &momentum));
  }

  bool has_momentum_node = false;
  bool has_gd_node = false;
  bool has_grad_accum_node = false;
  bool has_grad_accum_with_momentum_node = false;
  TF_CHECK_OK(RunPass(&g, &has_momentum_node, &has_gd_node,
                      &has_grad_accum_node,
                      &has_grad_accum_with_momentum_node));

  EXPECT_FALSE(has_momentum_node);
  EXPECT_TRUE(has_gd_node);
  EXPECT_FALSE(has_grad_accum_node);
  EXPECT_TRUE(has_grad_accum_with_momentum_node);
  {
    // Check the gradient accumulation counter.
    auto node_index = g->BuildNodeNameIndex();
    int num_mini_batches;
    TF_CHECK_OK(GetNodeAttr(node_index[kIpuStatefulGradientAccumulate]->attrs(),
                            "num_mini_batches", &num_mini_batches));
    EXPECT_EQ(num_mini_batches, 5);
  }
}
}  // namespace
}  // namespace tensorflow
