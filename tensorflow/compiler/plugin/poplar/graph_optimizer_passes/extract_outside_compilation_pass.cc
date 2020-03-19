/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/jit/extract_outside_compilation_pass.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

constexpr char kFunctionToApplyAttrName[] = "to_apply";
constexpr char kInputTypesAttrName[] = "Tinputs";
constexpr char kKeyAttrName[] = "key";
constexpr char kOutputTypesAttrName[] = "Toutputs";
constexpr char kXlaClusterAttrName[] = "_XlaCluster";

bool IsKeyPlaceholderNode(const Node& n) {
  return n.type_string() == "Placeholder" &&
         absl::EndsWith(n.name(), "_key_placeholder");
}

bool IsSequencerNode(const Node& n) {
  return n.type_string() == "NoOp" &&
         HasNodeAttr(n.def(), "_xla_host_transfer_sequencer");
}

bool IsXlaLaunchNode(const Node& n) { return n.type_string() == "XlaLaunch"; }

bool IsXlaRecvAtHostNode(const Node& n) {
  return n.type_string() == "_XlaRecvAtHost";
}

bool IsXlaSendFromHostNode(const Node& n) {
  return n.type_string() == "_XlaSendFromHost";
}

Status ReplaceKeyPlaceholdersWithConstants(Graph* g) {
  for (Node* n : g->nodes()) {
    if (IsKeyPlaceholderNode(*n)) {
      NodeDef const_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(n->name(), "Const")
                             .Attr("dtype", DT_STRING)
                             .Attr("value", Tensor(DT_STRING))
                             .Finalize(&const_def));
      TF_ASSIGN_OR_RETURN(n, ReplaceNode(g, n, const_def));
    }
  }

  return Status::OK();
}

void RemoveSequencerNodes(Graph* g) {
  for (Node* n : g->nodes()) {
    if (IsSequencerNode(*n)) {
      g->RemoveNode(n);
    }
  }
}

Status CheckForXlaSendToHostNodes(const FunctionDef* function_def) {
  for (const NodeDef& n : function_def->node_def()) {
    if (n.op() == "XlaSendToHost") {
      return errors::Unimplemented(
          "`outside_compilation_scope` enclosed in control flow "
          "(loop or cond) is not supported");
    }
  }

  return Status::OK();
}

void AddFunctions(const string& func, Node* xla_launch_node,
                  FunctionLibraryDefinition* flib_def,
                  std::unordered_map<string, XlaClusterInfo>* clusters) {
  // Add the current function.
  NameAttrList func_name_attrs;
  func_name_attrs.set_name(func);
  clusters->emplace(func, XlaClusterInfo{func, func_name_attrs, xla_launch_node,
                                         std::map<string, int>{}});

  // Do a recursive depth first search through the functions found
  // in the kFunctionToApplyAttrName. This is done for extracting
  // outside compilations from within pipeline functions.
  const FunctionDef* function_def = flib_def->Find(func);
  CHECK(function_def) << "not found: " << func;
  for (const NodeDef& n : function_def->node_def()) {
    const auto found = n.attr().find(kFunctionToApplyAttrName);
    if (found != n.attr().end()) {
      const AttrValue& to_apply = found->second;
      if (to_apply.has_func()) {
        const string& child_func = to_apply.func().name();
        AddFunctions(child_func, xla_launch_node, flib_def, clusters);
      }
    }
  }
}

std::unordered_map<string, XlaClusterInfo> FindClusters(
    Graph* g, FunctionLibraryDefinition* flib_def) {
  std::unordered_map<string, XlaClusterInfo> clusters;

  for (Node* n : g->op_nodes()) {
    if (IsXlaLaunchNode(*n)) {
      const AttrValue* f = n->attrs().Find("function");
      CHECK_NOTNULL(f);
      CHECK(f->has_func());
      const string& func = f->func().name();
      AddFunctions(func, n, flib_def, &clusters);
    }
  }

  return clusters;
}

// Splits a node with N outputs into N nodes with 1 output each, all
// getting the same inputs as the original node.
Status SplitXlaRecvAtHostNode(Graph* g, Node* n) {
  // Create a template based on the original node and clear attributes that must
  // be unique for the split nodes.
  NodeDef node_def_template(n->def());
  node_def_template.clear_name();
  node_def_template.mutable_attr()->erase(kKeyAttrName);
  node_def_template.mutable_attr()->erase(kOutputTypesAttrName);

  // Grab attributes needed to set unique attributes for the split nodes.
  string key;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kKeyAttrName, &key));
  std::vector<DataType> output_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kOutputTypesAttrName, &output_types));
  TF_RET_CHECK(static_cast<int32>(output_types.size()) == n->num_outputs());

  // Record the original node's output edges and remove them first. This is to
  // avoid multiple producers for dst nodes' input.
  std::vector<OutEdgeInfo> out_edge_info;
  std::vector<const Edge*> out_edges;
  for (const Edge* edge : n->out_edges()) {
    out_edges.push_back(edge);
    out_edge_info.push_back(
        {edge->dst(), edge->src_output(), edge->dst_input()});
  }
  for (const Edge* edge : out_edges) {
    g->RemoveEdge(edge);
  }

  // Add the new split nodes based on the template.
  for (int32 i = 0; i < n->num_outputs(); ++i) {
    NodeDef new_node_def(node_def_template);
    new_node_def.set_name(strings::StrCat(n->name(), i));

    // Refer to CreateSendRendezvousKey in host_compute_kernels.cc.
    const string new_key = strings::StrCat(key, ":", i);
    AddNodeAttr(kKeyAttrName, new_key, &new_node_def);
    AddNodeAttr(kOutputTypesAttrName, {output_types[i]}, &new_node_def);

    Status s;
    Node* new_node = g->AddNode(new_node_def, &s);
    if (!s.ok()) {
      return s;
    }

    // Add all the original node's input edges to the new node.
    for (const Edge* in_edge : n->in_edges()) {
      g->AddEdge(in_edge->src(), in_edge->src_output(), new_node,
                 in_edge->dst_input());
    }

    // Add the output edges belonging to this new node, i.e. the output
    // edges connected to output slot i of the original node.
    for (const OutEdgeInfo& out_edge : out_edge_info) {
      if (out_edge.src_output == i) {
        g->AddEdge(new_node, /*src_output=*/0, out_edge.dst,
                   out_edge.dst_input);
      }
    }
  }

  // Remove the original node.
  g->RemoveNode(n);

  return Status::OK();
}

Status SplitXlaRecvAtHostNodes(Graph* g) {
  for (Node* n : g->op_nodes()) {
    if (IsXlaRecvAtHostNode(*n)) {
      TF_RETURN_IF_ERROR(SplitXlaRecvAtHostNode(g, n));
    }
  }

  return Status::OK();
}

// Splits a node with N inputs into N - 1 nodes with 2 inputs each (the last
// original input is connected to all the new nodes).
Status SplitXlaSendFromHostNode(Graph* g, Node* n) {
  // Create a template based on the original node and clear attributes that must
  // be unique for the split nodes.
  NodeDef node_def_template(n->def());
  node_def_template.clear_name();
  node_def_template.mutable_attr()->erase(kKeyAttrName);
  node_def_template.mutable_attr()->erase(kInputTypesAttrName);

  CHECK_EQ(n->num_outputs(), 0);
  CHECK_GT(n->num_inputs(), 1);

  // Grab attributes needed to set unique attributes for the split nodes.
  string key;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kKeyAttrName, &key));
  std::vector<DataType> input_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kInputTypesAttrName, &input_types));
  TF_RET_CHECK(static_cast<int32>(input_types.size()) == n->num_inputs() - 1);

  const Edge* last_edge;
  TF_RETURN_IF_ERROR(n->input_edge(n->num_inputs() - 1, &last_edge));

  // Add the new split nodes based on the template.
  for (int32 i = 0; i < n->num_inputs() - 1; ++i) {
    NodeDef new_node_def(node_def_template);
    new_node_def.set_name(strings::StrCat(n->name(), i));

    // Refer to CreateRecvRendezvousKey in host_compute_kernels.cc.
    const string new_key = strings::StrCat(key, ":", i);
    AddNodeAttr(kKeyAttrName, new_key, &new_node_def);
    AddNodeAttr(kInputTypesAttrName, {input_types[i]}, &new_node_def);

    Status s;
    Node* new_node = g->AddNode(new_node_def, &s);
    if (!s.ok()) {
      return s;
    }

    const Edge* in_edge;
    TF_RETURN_IF_ERROR(n->input_edge(i, &in_edge));
    g->AddEdge(in_edge->src(), in_edge->src_output(), new_node,
               /*dst_input=*/0);
    g->AddEdge(last_edge->src(), last_edge->src_output(), new_node,
               /*dst_input=*/1);
  }

  // Remove the original node.
  g->RemoveNode(n);

  return Status::OK();
}

Status SplitXlaSendFromHostNodes(Graph* g) {
  for (Node* n : g->op_nodes()) {
    if (IsXlaSendFromHostNode(*n)) {
      TF_RETURN_IF_ERROR(SplitXlaSendFromHostNode(g, n));
    }
  }

  return Status::OK();
}

}  // namespace

Status ExtractOutsideCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  FunctionLibraryDefinition* flib_def = options.flib_def;
  TF_RET_CHECK(flib_def != nullptr);
  TF_RET_CHECK(options.session_options != nullptr);

  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, TF_GRAPH_DEF_VERSION, flib_def,
      OptimizerOptions());

  Graph* graph = options.graph->get();

  const auto clusters = FindClusters(graph, flib_def);

  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  TF_RET_CHECK(flr != nullptr);

  bool modified = false;

  // Rewrites XLA computation in `clusters` to replace outside compilation nodes
  // with XlaHostCompute, and moves those outside compilations into `graph`.
  TF_RETURN_IF_ERROR(ExtractOutsideCompilation(
      kXlaClusterAttrName, kXlaOutsideCompilationAttrName, clusters, graph, flr,
      flib_def, &modified));

  if (!modified) {
    return Status::OK();
  }

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_fixup_before", *graph,
                    flib_def);
  }

  // According to the docs in tpu_host_compute_ops.cc, the key placeholder is
  // supposed to be sent at runtime by the compile node to identify which
  // execution the transfer corresponds to. We should have control over this
  // by the Send/Recv registration in poplar_compiler.cc and poplar_executor.cc.
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholdersWithConstants(graph));

  // The sequencer node has control input edges from the
  // _XlaRecvAtHost/_XlaSendFromHost ops and a control output edge to the
  // XlaLaunch op, so it requires the Send/Recv to complete before engine
  // compilation, which does not match our architecture, resulting in deadlock.
  // To be honest, not really sure what it's supposed to do for TPUs.
  RemoveSequencerNodes(graph);

  // XlaSendToHost nodes are inserted when the outside compilation scope
  // is enclosed in control flow. This is not currently supported, so
  // report a nice error message if we find any.
  for (const auto& func : flib_def->ListFunctionNames()) {
    const FunctionDef* function_def = flib_def->Find(func);
    TF_RET_CHECK(function_def != nullptr);
    TF_RETURN_IF_ERROR(CheckForXlaSendToHostNodes(function_def));
  }

  // Split XlaRecvAtHost nodes with multiple outputs such that the new nodes
  // have only one output each. This rewrite avoids the synchronization point
  // of a single node that waits for all the necessary data to satisfy all of
  // its outputs. The split nodes can complete when only their data is ready.
  TF_RETURN_IF_ERROR(SplitXlaRecvAtHostNodes(graph));

  // Do the same for XlaSendFromHost nodes such that we can start sending
  // data as soon as it is ready instead of waiting for all of it.
  TF_RETURN_IF_ERROR(SplitXlaSendFromHostNodes(graph));

  // Run the placer again to assign devices to the nodes added by this pass.
  // Make sure the default local device is used when in a distributed context.
  Device* default_local_device = options.device_set->client_device();
  Placer placer(graph, "", options.device_set, default_local_device);

  TF_RETURN_IF_ERROR(placer.Run());

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_fixup_after", *graph,
                    flib_def);
  }

  return Status::OK();
}

}  // namespace tensorflow
