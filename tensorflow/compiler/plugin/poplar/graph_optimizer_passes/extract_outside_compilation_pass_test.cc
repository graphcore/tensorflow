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

 private:
  SessionOptions session_options_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  DeviceSet device_set_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
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

    auto g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_CHECK_OK(s.ToGraph(g.get()));

    auto function_nodes = g->BuildNodeNameIndex();

    function_nodes["identity0"]->AddAttr(kXlaOutsideCompilationAttrName,
                                         "outside");
    function_nodes["identity1"]->AddAttr(kXlaOutsideCompilationAttrName,
                                         "outside");

    function_nodes["identity0"]->AddAttr(kXlaInferredShapesAttrName,
                                         std::vector<PartialTensorShape>{{}});
    function_nodes["identity1"]->AddAttr(kXlaInferredShapesAttrName,
                                         std::vector<PartialTensorShape>{{}});

    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", fdef_lib.add_function()));
  }

  auto flib_def = FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib);
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());

  // Add an XlaLaunch node that launches the function in the outside graph.
  NodeDef launch_def;
  launch_def.set_op("XlaLaunch");
  launch_def.set_name("launch");
  AddNodeAttr("Tconstants", DataTypeVector{}, &launch_def);
  AddNodeAttr("Targs", DataTypeVector{}, &launch_def);
  AddNodeAttr("Nresources", 0, &launch_def);
  AddNodeAttr("Tresults", 0, &launch_def);
  NameAttrList function;
  function.set_name("cluster");
  AddNodeAttr("function", function, &launch_def);
  Status s;
  graph->AddNode(launch_def, &s);
  TF_CHECK_OK(s);

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

}  // namespace
}  // namespace tensorflow
