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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/extract_outside_compilation_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

class ExtractOutsideCompilationPassTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        session_options_, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
    for (Device* d : device_mgr_->ListDevices()) {
      device_set_.AddDevice(d);
    }
  }

  Status RunPass(FunctionLibraryDefinition* flib_def,
                 std::unique_ptr<Graph>* graph) {
    GraphOptimizationPassOptions opts;
    opts.flib_def = flib_def;
    opts.graph = graph;
    opts.device_set = &device_set_;
    opts.session_options = &session_options_;
    return ExtractOutsideCompilationPass().Run(opts);
  }

  Status AddLaunchNode(Graph* graph, const std::string& func_name) {
    NodeDef launch_def;
    launch_def.set_op("XlaLaunch");
    launch_def.set_name("launch");
    AddNodeAttr("Tconstants", DataTypeVector{}, &launch_def);
    AddNodeAttr("Targs", DataTypeVector{}, &launch_def);
    AddNodeAttr("Nresources", 0, &launch_def);
    AddNodeAttr("Tresults", 0, &launch_def);
    NameAttrList function;
    function.set_name(func_name);
    AddNodeAttr("function", function, &launch_def);

    Status s;
    graph->AddNode(launch_def, &s);
    return s;
  }

 private:
  SessionOptions session_options_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  DeviceSet device_set_;
};

TEST_F(ExtractOutsideCompilationPassTest, TwoInputsTwoOutputs) {
  // First make a function that looks like this, where the "outside"
  // nodes should be extracted:
  //
  //              outside
  //            -----------
  // const0 -> | identity0 | -> identity2
  // const1 -> | identity1 | -> identity3
  //            -----------

  FunctionDefLibrary fdef_lib;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output const0 = ops::Const(s.WithOpName("const0"), 1.0f, {});
    Output const1 = ops::Const(s.WithOpName("const1"), 2.0f, {});

    Output identity0 = ops::Identity(s.WithOpName("identity0"), const0);
    Output identity1 = ops::Identity(s.WithOpName("identity1"), const1);

    Output identity2 = ops::Identity(s.WithOpName("identity2"), identity0);
    Output identity3 = ops::Identity(s.WithOpName("identity3"), identity1);

    identity0.node()->AddAttr(kXlaOutsideCompilationAttrName, "outside");
    identity0.node()->AddAttr(kXlaInferredShapesAttrName,
                              std::vector<PartialTensorShape>{{}});

    identity1.node()->AddAttr(kXlaOutsideCompilationAttrName, "outside");
    identity1.node()->AddAttr(kXlaInferredShapesAttrName,
                              std::vector<PartialTensorShape>{{}});

    Graph g(OpRegistry::Global());
    TF_CHECK_OK(s.ToGraph(&g));
    TF_CHECK_OK(GraphToFunctionDef(g, "cluster", fdef_lib.add_function()));
  }

  auto flib_def = FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib);
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());

  // Add an XlaLaunch node that launches the function in the outside graph.
  TF_CHECK_OK(AddLaunchNode(graph.get(), "cluster"));

  // Run the pass to perform the extraction.
  TF_CHECK_OK(RunPass(&flib_def, &graph));

  // Check that we find the expected nodes in the outside graph.
  auto graph_nodes = graph->BuildNodeNameIndex();

  // The two first identity nodes should have been extracted.
  Node* identity0 = graph_nodes["identity0"];
  ASSERT_NE(identity0, nullptr);
  Node* identity1 = graph_nodes["identity1"];
  ASSERT_NE(identity1, nullptr);

  // There should be two XlaRecvAtHost nodes with the identity nodes as outputs.
  const auto num_recv_at_host = absl::c_count_if(graph->nodes(), [](Node* n) {
    return n->type_string() == "_XlaRecvAtHost";
  });
  EXPECT_EQ(num_recv_at_host, 2);

  Node* recv0 =
      graph_nodes["outside_compilation_cluster_cluster_outside_recv0"];
  ASSERT_NE(recv0, nullptr);
  EXPECT_EQ(recv0->type_string(), "_XlaRecvAtHost");
  EXPECT_EQ(recv0->attrs().Find("key")->s(),
            "host_compute_channel_cluster_cluster_outside:0");
  EXPECT_EQ(recv0->num_outputs(), 1);
  EXPECT_EQ(recv0->out_nodes().begin()->name(), identity0->name());

  Node* recv1 =
      graph_nodes["outside_compilation_cluster_cluster_outside_recv1"];
  ASSERT_NE(recv1, nullptr);
  EXPECT_EQ(recv1->type_string(), "_XlaRecvAtHost");
  EXPECT_EQ(recv1->attrs().Find("key")->s(),
            "host_compute_channel_cluster_cluster_outside:1");
  EXPECT_EQ(recv1->num_outputs(), 1);
  EXPECT_EQ(recv1->out_nodes().begin()->name(), identity1->name());

  // There should be two XlaSendFromHost nodes with the identity nodes as
  // inputs.
  const auto num_send_from_host = absl::c_count_if(graph->nodes(), [](Node* n) {
    return n->type_string() == "_XlaSendFromHost";
  });
  EXPECT_EQ(num_send_from_host, 2);

  Node* send0 =
      graph_nodes["outside_compilation_cluster_cluster_outside_send0"];
  ASSERT_NE(send0, nullptr);
  EXPECT_EQ(send0->type_string(), "_XlaSendFromHost");
  EXPECT_EQ(send0->attrs().Find("key")->s(),
            "host_compute_channel_cluster_cluster_outside:0");
  EXPECT_EQ(send0->num_inputs(), 2);
  EXPECT_EQ(identity0->out_nodes().begin()->name(), send0->name());

  Node* send1 =
      graph_nodes["outside_compilation_cluster_cluster_outside_send1"];
  ASSERT_NE(send1, nullptr);
  EXPECT_EQ(send1->type_string(), "_XlaSendFromHost");
  EXPECT_EQ(send1->attrs().Find("key")->s(),
            "host_compute_channel_cluster_cluster_outside:1");
  EXPECT_EQ(send1->num_inputs(), 2);
  EXPECT_EQ(identity1->out_nodes().begin()->name(), send1->name());

  // Check that we find the expected nodes in the rewritten function.
  std::unique_ptr<FunctionBody> xla_fbody;
  TF_CHECK_OK(FunctionDefToBodyHelper(*flib_def.Find("cluster"), AttrSlice(),
                                      &flib_def, &xla_fbody));

  auto function_nodes = xla_fbody->graph->BuildNodeNameIndex();

  // The const nodes should still be there.
  Node* const0 = function_nodes["const0"];
  ASSERT_NE(const0, nullptr);
  Node* const1 = function_nodes["const1"];
  ASSERT_NE(const1, nullptr);

  // The first two identity nodes should not be there anymore, as they are
  // extracted.
  EXPECT_EQ(function_nodes.count("identity0"), 0);
  EXPECT_EQ(function_nodes.count("identity1"), 0);

  // The last two identity nodes should still be there.
  Node* identity2 = function_nodes["identity2"];
  ASSERT_NE(identity2, nullptr);
  Node* identity3 = function_nodes["identity3"];
  ASSERT_NE(identity3, nullptr);

  // The extracted nodes should be replaced by an XlaHostCompute node.
  Node* host_compute =
      function_nodes["outside_compilation_outside_host_compute"];
  ASSERT_NE(host_compute, nullptr);
  EXPECT_EQ(host_compute->type_string(), "XlaHostCompute");
  EXPECT_EQ(host_compute->num_inputs(), 2);
  EXPECT_EQ(host_compute->out_nodes().begin()->name(), identity2->name());
  EXPECT_EQ(std::next(host_compute->out_nodes().begin())->name(),
            identity3->name());
  EXPECT_EQ(host_compute->in_nodes().begin()->name(), const0->name());
  EXPECT_EQ(std::next(host_compute->in_nodes().begin())->name(),
            const1->name());
}

TEST_F(ExtractOutsideCompilationPassTest, PipelineRepeatCount) {
  FunctionDefLibrary fdef_lib;

  // Create a "pipeline" function that contains an "outside" scope:
  //
  //              pipeline
  //  ----------------------------------
  // |            outside               |
  // |           ---------              |
  // | const -> | sigmoid | -> identity |
  // |           ---------              |
  //  ----------------------------------
  const auto pipeline_function_name = "pipeline";
  {
    auto s = tensorflow::Scope::NewRootScope();
    auto const0 = ops::Const(s.WithOpName("const"), 1.0f, {});
    auto sigmoid = ops::Sigmoid(s.WithOpName("sigmoid"), const0);
    auto identity = ops::Identity(s.WithOpName("identity"), sigmoid);

    sigmoid.node()->AddAttr(kXlaOutsideCompilationAttrName, "outside");
    sigmoid.node()->AddAttr(kXlaInferredShapesAttrName,
                            std::vector<PartialTensorShape>{{}});

    Graph g(OpRegistry::Global());
    TF_CHECK_OK(s.ToGraph(&g));
    TF_CHECK_OK(
        GraphToFunctionDef(g, pipeline_function_name, fdef_lib.add_function()));
  }

  // Create a "cluster" function that contains a Pipeline op invoking the
  // "pipeline" function with a repeat_count (i.e. a loop).
  const int32 pipeline_repeat_count = 42;
  {
    Graph g(OpRegistry::Global());

    NameAttrList to_apply;
    to_apply.set_name(pipeline_function_name);

    Node* constant_node;
    TF_CHECK_OK(NodeBuilder("Const", "Const")
                    .Attr("dtype", DT_INT32)
                    .Attr("value", Tensor(1))
                    .Finalize(&g, &constant_node));

    Node* pipeline_node;
    TF_CHECK_OK(
        NodeBuilder("pipeline", "Pipeline")
            .Input(std::vector<NodeBuilder::NodeOut>{})
            .Input(constant_node)
            .Attr("repeat_count", pipeline_repeat_count)
            .Attr("output_shapes", std::vector<TensorShape>{})
            .Attr("pipeline_poplar_config", "")
            .Attr("Tout", std::vector<TensorShape>{})
            .Attr("batch_serialization_iterations", 1)
            .Attr("schedule", 0)
            .Attr("offload_activations", "THREESTATE_FALSE")
            .Attr("offload_gradient_accumulation_buffers", "THREESTATE_FALSE")
            .Attr("replicated_weight_sharding", "THREESTATE_FALSE")
            .Attr("offload_weights", "THREESTATE_FALSE")
            .Attr("recomputation_mode", "Auto")
            .Attr("to_apply", to_apply)
            .Finalize(&g, &pipeline_node));

    TF_CHECK_OK(GraphToFunctionDef(g, "cluster", fdef_lib.add_function()));
  }

  auto flib_def = FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib);
  auto graph = absl::make_unique<Graph>(&flib_def);

  // Add an XlaLaunch node that launches the function in the outside graph.
  TF_CHECK_OK(AddLaunchNode(graph.get(), "cluster"));

  // Run the pass to perform the extraction.
  TF_CHECK_OK(RunPass(&flib_def, &graph));

  // This is roughly what we expect:
  //
  // The pipeline function has got the sigmoid op extracted and replaced
  // by an XlaHostCompute op (that will be lowered to SendToHost/RecvFromHost):
  //
  // pipeline() -> () {
  //  const = Const()
  //  host_compute = XlaHostCompute(const)
  //  identity = Identity(host_compute)
  // }
  //
  // On the host side we expect a while loop with a body that communicates with
  // the pipeline function using _XlaRecvAtHost/_XlaSendFromHost:
  //
  // body(input:int32) -> (output:int32) {
  //   recv = _XlaRecvAtHost()
  //   sigmoid = Sigmoid(recv)
  //   send = _XlaSendFromHost(sigmoid)
  //   one = Const[value=1]()
  //   add = Add(input, one)
  //   return output = add
  // }

  // Check that we have a loop with the expected nodes in the outside graph.
  int num_loop_conds = 0;
  int num_next_iterations = 0;
  for (const Node* n : graph->nodes()) {
    if (n->IsLoopCond()) {
      ++num_loop_conds;

      // Chceck that the loop condition contains the expected nodes.
      Node* cond_node;
      TF_CHECK_OK(n->input_node(0, &cond_node));

      const auto cond_function_name = cond_node->type_string();
      const auto* cond_fdef = flib_def.Find(cond_function_name);
      CHECK_NOTNULL(cond_fdef);

      std::unique_ptr<FunctionBody> cond_fbody;
      TF_CHECK_OK(FunctionDefToBodyHelper(*cond_fdef, AttrSlice(), &flib_def,
                                          &cond_fbody));

      // The condition function should have type `int32 -> bool` and have an
      // upper bound comparison with the value `pipeline_repeat_count`.
      ASSERT_EQ(cond_fdef->signature().input_arg_size(), 1);
      ASSERT_EQ(cond_fdef->signature().output_arg_size(), 1);
      ASSERT_EQ(cond_fdef->signature().input_arg(0).type(), DT_INT32);
      ASSERT_EQ(cond_fdef->signature().output_arg(0).type(), DT_BOOL);
      ASSERT_EQ(cond_fdef->node_def_size(), 2);
      ASSERT_EQ(cond_fdef->node_def(0).op(), "Const");
      ASSERT_EQ(cond_fdef->node_def(0).attr().at("value").tensor().int_val(0),
                pipeline_repeat_count);
      ASSERT_EQ(cond_fdef->node_def(1).op(), "Less");
    } else if (n->IsNextIteration()) {
      ++num_next_iterations;

      // Check that the loop body contains the expected nodes.
      Node* body_node;
      TF_CHECK_OK(n->input_node(0, &body_node));

      const auto body_function_name = body_node->type_string();
      const auto* body_fdef = flib_def.Find(body_function_name);
      CHECK_NOTNULL(body_fdef);

      // The body function should have type `int32 -> int32` for the loop
      // counter.
      ASSERT_EQ(body_fdef->signature().input_arg_size(), 1);
      ASSERT_EQ(body_fdef->signature().output_arg_size(), 1);
      ASSERT_EQ(body_fdef->signature().input_arg(0).type(), DT_INT32);
      ASSERT_EQ(body_fdef->signature().output_arg(0).type(), DT_INT32);

      std::unique_ptr<FunctionBody> body_fbody;
      TF_CHECK_OK(FunctionDefToBodyHelper(*body_fdef, AttrSlice(), &flib_def,
                                          &body_fbody));

      // Check that the expected nodes are found in the while body function.
      auto body_nodes = body_fbody->graph->BuildNodeNameIndex();

      const Node* recv =
          body_nodes["outside_compilation_pipeline_pipeline_outside_recv0"];
      ASSERT_NE(recv, nullptr);

      const Node* sigmoid = body_nodes["sigmoid"];
      ASSERT_NE(sigmoid, nullptr);
      ASSERT_EQ(sigmoid->num_inputs(), 1);
      ASSERT_EQ(sigmoid->in_nodes().begin()->name(), recv->name());

      const Node* send =
          body_nodes["outside_compilation_pipeline_pipeline_outside_send0"];
      ASSERT_NE(send, nullptr);
      ASSERT_EQ(send->num_inputs(), 2);
      ASSERT_EQ(send->in_nodes().begin()->name(), sigmoid->name());

      const Node* one = body_nodes["pipeline_oc_host_graph_launch_one"];
      ASSERT_NE(one, nullptr);
      ASSERT_TRUE(one->IsConstant());

      const Node* add = body_nodes["pipeline_oc_host_graph_launch_add"];
      ASSERT_NE(add, nullptr);
      ASSERT_EQ(add->num_inputs(), 2);
      ASSERT_TRUE(add->in_nodes().begin()->IsArg());
      ASSERT_EQ(std::next(add->in_nodes().begin())->name(), one->name());
      ASSERT_TRUE(add->out_nodes().begin()->IsRetval());
    }
  }

  CHECK_EQ(num_loop_conds, 1);
  CHECK_EQ(num_next_iterations, 1);
}

TEST_F(ExtractOutsideCompilationPassTest, NoOutsideScope) {
  FunctionDefLibrary fdef_lib;

  // Create a "cluster" function without any outside scopes.
  {
    auto s = tensorflow::Scope::NewRootScope();
    auto const0 = ops::Const(s.WithOpName("const"), 1.0f, {});
    auto sigmoid = ops::Sigmoid(s.WithOpName("sigmoid"), const0);
    auto identity = ops::Identity(s.WithOpName("identity"), sigmoid);

    Graph g(OpRegistry::Global());
    TF_CHECK_OK(s.ToGraph(&g));
    TF_CHECK_OK(GraphToFunctionDef(g, "cluster", fdef_lib.add_function()));
  }

  auto flib_def = FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib);
  auto graph = absl::make_unique<Graph>(&flib_def);

  // Add an XlaLaunch node that launches the function in the outside graph.
  TF_CHECK_OK(AddLaunchNode(graph.get(), "cluster"));

  GraphDef before_graph_def;
  graph->ToGraphDef(&before_graph_def);
  const FunctionDefLibrary before_fdef_lib = flib_def.ToProto();

  // Run the pass that should do nothing.
  TF_CHECK_OK(RunPass(&flib_def, &graph));

  GraphDef after_graph_def;
  graph->ToGraphDef(&after_graph_def);
  const FunctionDefLibrary after_fdef_lib = flib_def.ToProto();

  // Check that the graph is unchanged.
  EqualGraphDefOptions equal_options;
  equal_options.ignore_internal_attrs = false;
  std::string diff_graph_def;
  const bool equal_graphs = EqualGraphDef(after_graph_def, before_graph_def,
                                          &diff_graph_def, equal_options);
  ASSERT_TRUE(equal_graphs) << diff_graph_def;

  // Check that the functions are unchanged.
  ASSERT_EQ(before_fdef_lib.function_size(), after_fdef_lib.function_size());
  for (int i = 0; i < before_fdef_lib.function_size(); ++i) {
    const auto& before_function = before_fdef_lib.function(i);
    const auto& after_function = after_fdef_lib.function(i);
    ASSERT_TRUE(FunctionDefsEqual(before_function, after_function))
        << DebugString(before_function)
        << " != " << DebugString(after_function);
  }
}

}  // namespace
}  // namespace tensorflow
