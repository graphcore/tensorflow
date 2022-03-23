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
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

constexpr char kFunctionToApplyAttrName[] = "to_apply";
constexpr char kFunctionRepeatCountAttrName[] = "repeat_count";
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

struct XlaClusterRepeatInfo : public XlaClusterInfo {
  XlaClusterRepeatInfo(const string& cluster_name,
                       const NameAttrList& func_name_attrs, Node* node,
                       int32 repeat_count)
      : XlaClusterInfo(cluster_name, func_name_attrs, node, {}),
        repeat_count(repeat_count) {}
  const int32 repeat_count;
};

// Control return mapping function for outside compilation host graphs.
// All nodes with kXlaHasHostTransfer attribute are control outputs.
absl::optional<string> HostGraphControlRetMapping(const Node* n) {
  if (HasNodeAttr(n->def(), kXlaHasHostTransferAttrName)) {
    return n->name();
  }
  return absl::nullopt;
}

void AddFunctions(const string& func, Node* xla_launch_node,
                  FunctionLibraryDefinition* flib_def,
                  std::unordered_map<string, XlaClusterRepeatInfo>* clusters,
                  int32 repeat_count = 1) {
  // Add the current function.
  NameAttrList func_name_attrs;
  func_name_attrs.set_name(func);
  clusters->emplace(func, XlaClusterRepeatInfo{func, func_name_attrs,
                                               xla_launch_node, repeat_count});

  // Do a recursive depth first search through the functions found
  // in the kFunctionToApplyAttrName. This is done for extracting
  // outside compilations from within pipeline functions.
  const FunctionDef* function_def = flib_def->Find(func);
  CHECK(function_def) << "not found: " << func;
  for (const NodeDef& n : function_def->node_def()) {
    const auto to_apply_attr = n.attr().find(kFunctionToApplyAttrName);
    if (to_apply_attr != n.attr().end()) {
      const AttrValue& to_apply = to_apply_attr->second;
      if (to_apply.has_func()) {
        const auto repeat_count_attr =
            n.attr().find(kFunctionRepeatCountAttrName);
        if (repeat_count_attr != n.attr().end()) {
          const int32 repeat_count_value = repeat_count_attr->second.i();
          CHECK_GT(repeat_count_value, 0);
          repeat_count *= repeat_count_value;
        }

        const string& child_func = to_apply.func().name();
        AddFunctions(child_func, xla_launch_node, flib_def, clusters,
                     repeat_count);
      }
    }
  }
}

std::unordered_map<string, XlaClusterRepeatInfo> FindClusters(
    Graph* g, FunctionLibraryDefinition* flib_def) {
  std::unordered_map<string, XlaClusterRepeatInfo> clusters;

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

// Note: Copied from tensorflow/compiler/jit/extract_outside_compilation_pass.cc
// as it is in an unnamed namespace there.
// Expand XLA computation's outside compilation host side graph into main graph.
// Add a control edge between sequencer node and the XLA computation node.
Status ExpandHostGraphIntoMainGraph(Graph* main_graph,
                                    FunctionLibraryDefinition* fld,
                                    const string& host_graph_func_name,
                                    Node* xla_computation_node,
                                    Node* pivot_node) {
  // Temporarily use "0" as "_device_ordinal". It will be rewritten with the
  // correct value in a later pass. We cannot just use placeholder value here
  // because FunctionDef instantiation does not allow placeholder value for
  // attributes.
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  const FunctionDef* host_graph_func = fld->Find(host_graph_func_name);
  TF_RET_CHECK(host_graph_func);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*host_graph_func,
                                             AttrSlice(&attrs), fld, &fbody));
  Graph* host_graph = fbody->graph;

  // We use ReverseDFS() to copy nodes. Make sure all nodes are reverse
  // reachable from sink node so all nodes will be copied.
  // TODO(b/77601805): consolidate copy graph functions.
  FixupSourceAndSinkEdges(host_graph);

  // Copy all nodes.
  std::map<const Node*, Node*> node_map;
  if (pivot_node) {
    node_map[host_graph->source_node()] = pivot_node;
  } else {
    node_map[host_graph->source_node()] = main_graph->source_node();
  }
  node_map[host_graph->sink_node()] = main_graph->sink_node();
  Status s = Status::OK();
  auto copy_node_fn = [&](const Node* n) {
    if (!s.ok()) {
      return;
    }

    Node* copy;
    if (node_map.find(n) != node_map.end()) {
      // Already copied this node.
      copy = node_map.at(n);
    } else {
      // Copy the node.
      NodeDef copy_def = n->def();
      copy = main_graph->AddNode(copy_def, &s);
      if (!s.ok()) {
        return;
      }
      node_map[n] = copy;
    }

    // Only handle input edges. Output edges will be added later as its output
    // nodes' input edges.
    for (auto e : n->in_edges()) {
      if (node_map.find(e->src()) == node_map.end()) {
        s = errors::Internal("Cannot find node image for ",
                             e->src()->DebugString());
        return;
      }
      main_graph->AddEdge(node_map[e->src()], e->src_output(), copy,
                          e->dst_input());
    }

    // Add control edge from sequencer to XLA computation node.
    if (copy->type_string() == "NoOp" &&
        HasNodeAttr(copy->def(), "_xla_host_transfer_sequencer")) {
      main_graph->AddControlEdge(copy, xla_computation_node);
    }
  };
  ReverseDFS(*host_graph, /*enter=*/nullptr, copy_node_fn, NodeComparatorID());
  return s;
}

Status FixupGraph(Graph* graph) {
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

  // Split XlaRecvAtHost nodes with multiple outputs such that the new nodes
  // have only one output each. This rewrite avoids the synchronization point
  // of a single node that waits for all the necessary data to satisfy all of
  // its outputs. The split nodes can complete when only their data is ready.
  TF_RETURN_IF_ERROR(SplitXlaRecvAtHostNodes(graph));

  // Do the same for XlaSendFromHost nodes such that we can start sending
  // data as soon as it is ready instead of waiting for all of it.
  TF_RETURN_IF_ERROR(SplitXlaSendFromHostNodes(graph));

  return Status::OK();
}

FunctionDef CreateLessThanFunction(const std::string& name,
                                   int32 repeat_count) {
  /*
   * Create the following function:
   *
   * func(in:int32) -> (out:bool) {
   *   repeat_count = Const[value=...]()
   *   less = Less(in, repeat_count)
   *   return out = less
   * }
   */
  return FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{"in: int32"},
      /*out_def=*/{"out: bool"},
      /*attr_def=*/{},
      /*node_def=*/
      {
          FunctionDefHelper::Const("repeat_count", repeat_count),
          FunctionDefHelper::Node{/*ret=*/{"less"},
                                  /*op=*/"Less",
                                  /*arg=*/{"in", "repeat_count:output:0"},
                                  /*attr=*/{{"T", DT_INT32}}},
      },
      /*ret_def=*/{{"out", "less:z:0"}});
}

Status AddWhileLoop(Graph* graph, FunctionLibraryDefinition* fld,
                    int32 repeat_count, const string& xla_cluster_name,
                    const string& loop_body_func_name) {
  Node* start_node;
  TF_RETURN_IF_ERROR(NodeBuilder(loop_body_func_name + "_start", "Const")
                         .Attr("value", Tensor(int32{0}))
                         .Attr("dtype", DT_INT32)
                         .Finalize(graph, &start_node));

  const string loop_cond_func_name = loop_body_func_name + "_cond";
  const FunctionDef loop_cond_func =
      CreateLessThanFunction(loop_cond_func_name, repeat_count);

  TF_RETURN_IF_ERROR(fld->AddFunctionDef(loop_cond_func));

  NameAttrList loop_cond_func_attr;
  loop_cond_func_attr.set_name(loop_cond_func_name);

  NameAttrList loop_body_func_attr;
  loop_body_func_attr.set_name(loop_body_func_name);

  Node* while_node;
  TF_RETURN_IF_ERROR(NodeBuilder(loop_body_func_name + "_while", "While")
                         .Attr("cond", loop_cond_func_attr)
                         .Attr("body", loop_body_func_attr)
                         .Attr(kXlaHasHostTransferAttrName, true)
                         .Attr(kXlaClusterAttrName, xla_cluster_name)
                         .Attr("parallel_iterations", 1)
                         .Input(std::vector<NodeBuilder::NodeOut>{{start_node}})
                         .Finalize(graph, &while_node));

  // Convert the While op into its lowered form, as this is the only form
  // that seems to work with the collective executor (when the loop body
  // contains collective ops).
  TF_RETURN_IF_ERROR(
      RewriteWhileNode(while_node, graph, fld, /*keep_node_fetchable=*/false));

  return Status::OK();
}

Status AddLoopBodyNodes(Graph* graph, const string& prefix) {
  Node* input_node;
  TF_RETURN_IF_ERROR(NodeBuilder(prefix + "_input", "_Arg")
                         .Attr("T", DT_INT32)
                         .Attr("index", 0)
                         .Finalize(graph, &input_node));

  Node* one_node;
  TF_RETURN_IF_ERROR(NodeBuilder(prefix + "_one", "Const")
                         .Attr("value", Tensor(int32{1}))
                         .Attr("dtype", DT_INT32)
                         .Finalize(graph, &one_node));

  Node* add_node;
  TF_RETURN_IF_ERROR(NodeBuilder(prefix + "_add", "Add")
                         .Input(input_node)
                         .Input(one_node)
                         .Finalize(graph, &add_node));

  Node* output_node;
  TF_RETURN_IF_ERROR(NodeBuilder(prefix + "_output", "_Retval")
                         .Attr("index", 0)
                         .Input(add_node)
                         .Finalize(graph, &output_node));

  return Status::OK();
}

Status RewriteAsLoopBodyFunction(FunctionLibraryDefinition* fld,
                                 const string& body_func_name) {
  // First find and instantiate the function as a graph.
  const FunctionDef* body_func = fld->Find(body_func_name);
  CHECK_NOTNULL(body_func);

  // The function should not have any inputs/outputs yet.
  CHECK_EQ(body_func->signature().input_arg_size(), 0);
  CHECK_EQ(body_func->signature().output_arg_size(), 0);

  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_temp_value;

  std::unique_ptr<FunctionBody> body_fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*body_func, AttrSlice(&attrs), fld, &body_fbody));

  // Fix up the graph after ExtractOutsideCompilationForFunction.
  TF_RETURN_IF_ERROR(FixupGraph(body_fbody->graph));

  // Add the iteration counter input, increment and output ops.
  TF_RETURN_IF_ERROR(AddLoopBodyNodes(body_fbody->graph, body_func_name));

  // Then replace the function with the rewritten one.
  FunctionDef rewritten_body_func;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*body_fbody->graph, body_func_name,
                                        HostGraphControlRetMapping,
                                        &rewritten_body_func));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(body_func_name, rewritten_body_func));

  return Status::OK();
}

xla::StatusOr<bool> CustomExtractOutsideCompilation(
    const std::unordered_map<string, XlaClusterRepeatInfo>& clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld) {
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_before", *g, fld);
  }

  bool modified = false;
  auto node_name_index = g->BuildNodeNameIndex();
  for (const auto& iter : clusters) {
    const string xla_cluster_name = iter.first;
    Node* xla_computation_node = iter.second.node;
    const auto& func_name_attrs = iter.second.func_name_attrs;
    const auto& host_compute_core = iter.second.host_compute_core;
    const int32 repeat_count = iter.second.repeat_count;
    CHECK_GT(repeat_count, 0);

    const string host_graph_func_name = absl::StrCat(
        xla_cluster_name, "_oc_host_graph_", xla_computation_node->name());

    bool has_outside_compilation;
    std::vector<string> shape_inference_graphs;
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        kXlaClusterAttrName, kXlaOutsideCompilationAttrName, xla_cluster_name,
        func_name_attrs, func_name_attrs.name(), host_graph_func_name,
        host_compute_core, flr, fld, &shape_inference_graphs,
        &has_outside_compilation));

    modified |= has_outside_compilation;

    if (has_outside_compilation) {
      if (repeat_count > 1) {
        // We must call the extracted host function repeatedly the same number
        // of times as the compiled loop for the communication to match up, so
        // wrap it in a while loop.

        // Rewrite the extracted host graph function to be callable from the
        // While op (e.g. accept an iteration counter, increment and return it).
        TF_RETURN_IF_ERROR(
            RewriteAsLoopBodyFunction(fld, host_graph_func_name));

        // Add the while loop calling the function from the outer host-side
        // graph.
        TF_RETURN_IF_ERROR(AddWhileLoop(g, fld, repeat_count, xla_cluster_name,
                                        host_graph_func_name));
      } else {
        // The outside function is not repeated; expand it into the host graph
        // and remove the function.
        string pivot_name = absl::StrCat(xla_cluster_name, "/pivot");
        Node* pivot_node = node_name_index[pivot_name];
        TF_RETURN_IF_ERROR(ExpandHostGraphIntoMainGraph(
            g, fld, host_graph_func_name, xla_computation_node, pivot_node));

        TF_RETURN_IF_ERROR(fld->RemoveFunction(host_graph_func_name));
      }
    }
  }

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_after", *g, fld);
  }

  return modified;
}

}  // namespace

Status ExtractOutsideCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  FunctionLibraryDefinition* flib_def = options.flib_def;
  TF_RET_CHECK(flib_def != nullptr);
  TF_RET_CHECK(options.session_options != nullptr);

  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, options.session_options->env, nullptr, TF_GRAPH_DEF_VERSION,
      flib_def, OptimizerOptions());

  Graph* graph = options.graph->get();

  const auto clusters = FindClusters(graph, flib_def);

  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  TF_RET_CHECK(flr != nullptr);

  // Rewrites XLA computation in `clusters` to replace outside compilation nodes
  // with XlaHostCompute, and moves those outside compilations into `graph`.
  TF_ASSIGN_OR_RETURN(bool modified, CustomExtractOutsideCompilation(
                                         clusters, graph, flr, flib_def));

  if (!modified) {
    return Status::OK();
  }

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_fixup_before", *graph,
                    flib_def);
  }

  // XlaSendToHost nodes are inserted when the outside compilation scope
  // is enclosed in control flow. This is not currently supported (except
  // for the special case handling of repeat_count above), so report a
  // nice error message if we find any.
  for (const auto& func : flib_def->ListFunctionNames()) {
    const FunctionDef* function_def = flib_def->Find(func);
    TF_RET_CHECK(function_def != nullptr);
    TF_RETURN_IF_ERROR(CheckForXlaSendToHostNodes(function_def));
  }

  TF_RETURN_IF_ERROR(FixupGraph(graph));

  // Run the placer again to assign devices to the nodes added by this pass.
  // Make sure the default local device is used when in a distributed context.
  Device* default_local_device = options.device_set->client_device();
  Placer placer(graph, "", flib_def, options.device_set, default_local_device);

  TF_RETURN_IF_ERROR(placer.Run());

  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_fixup_after", *graph,
                    flib_def);
  }

  return Status::OK();
}

}  // namespace tensorflow
