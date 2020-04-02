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

#include "tensorflow/compiler/plugin/poplar/kernels/functional/rearrange_function_arguments.h"

#include <string>

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

#include <algorithm>

namespace tensorflow {

namespace {

// Given original input types and argument index mapping, return the new input
// types.
std::vector<DataType> ShuffleInputDataTypeAttribute(
    const std::vector<DataType>& in_types,
    const std::vector<int>& index_mapping) {
  std::vector<DataType> result(index_mapping.size());
  for (size_t i = 0; i < in_types.size(); i++) {
    result[index_mapping.at(i)] = in_types[i];
  }
  return result;
}

// Given original input types, check if we need to rewrite the function (by
// checking if all DT_RESOURCE inputs are in the end). If the function needs to
// be rewritten, `resource_input_count` will be set to number of DT_RESOURCE
// inputs, and `index_mapping` will hold a mapping for original input index to
// rearranged input index.
Status InputTypesNeedsRearrange(const std::vector<DataType>& in_types,
                                bool* need_rewrite, int* resource_input_count,
                                std::vector<int>* index_mapping) {
  int first_resource_index = -1;
  for (size_t i = 0; i < in_types.size(); i++) {
    DataType type = in_types[i];
    if (type == DT_RESOURCE) {
      first_resource_index = i;
      break;
    }
  }
  if (first_resource_index == -1) {
    // No resource input. No need to rewrite.
    *need_rewrite = false;
    return Status::OK();
  }

  *need_rewrite = false;
  for (size_t i = first_resource_index + 1; i < in_types.size(); i++) {
    if (in_types[i] != DT_RESOURCE) {
      *need_rewrite = true;
      break;
    }
  }
  if (!*need_rewrite) {
    return Status::OK();
  }

  *resource_input_count = 0;
  for (size_t i = 0; i < in_types.size(); i++) {
    DataType type = in_types[i];
    if (type == DT_RESOURCE) {
      ++(*resource_input_count);
    }
  }
  int non_resource_index = 0,
      resource_index = in_types.size() - *resource_input_count;
  index_mapping->resize(in_types.size());
  for (size_t i = 0; i < in_types.size(); i++) {
    if (in_types[i] != DT_RESOURCE) {
      (*index_mapping)[i] = non_resource_index;
      non_resource_index++;
    } else {
      (*index_mapping)[i] = resource_index;
      resource_index++;
    }
  }

  return Status::OK();
}

// Given mapping between original input index and rearranged input index, change
// "index" attribute for _Arg nodes.
void RearrangeArgNodes(
    const gtl::InlinedVector<Node*, 4>* arg_nodes,  // non-absl ok
    const std::vector<int>& index_mapping) {
  for (size_t i = 0; i < arg_nodes->size(); i++) {
    Node* n = (*arg_nodes)[i];
    int new_index = index_mapping.at(i);
    n->ClearAttr("index");
    n->AddAttr("index", new_index);
  }
}

// Given all _Retval nodes in the function, return if we need to rewrite the
// function (by checking if we have DT_RESOURCE return values). If we need to
// rewrite the function, `retval_index_mapping` will hold the mapping from
// original _Retval to rearranged _Retval, and `resource_retval_to_arg` will
// hold mapping from DT_RESOURCE _Retval index to its input _Arg index. Here we
// assume that all DT_RESOURCE _Retval nodes come from _Arg nodes directly.
Status CalculateRetvalRearrange(
    const gtl::InlinedVector<Node*, 4>& ret_nodes,  // non-absl ok
    std::map<int, int>* retval_index_mapping,
    std::map<int, int>* resource_retval_to_arg) {
  for (size_t i = 0; i < ret_nodes.size(); i++) {
    Node* n = ret_nodes[i];
    DataType t;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "T", &t));
    if (t != DT_RESOURCE) {
      int new_retval_index = retval_index_mapping->size();
      retval_index_mapping->insert(std::make_pair(i, new_retval_index));
      continue;
    }

    const Edge* e;
    TF_RETURN_IF_ERROR(n->input_edge(0, &e));
    if (!e->src()->IsArg()) {
      return errors::Unimplemented(
          "Resource _Retval node's input does not come from _Arg "
          "directly: ",
          e->DebugString());
    }
    Node* arg = e->src();
    int src_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(arg->def(), "index", &src_index));
    resource_retval_to_arg->insert(std::make_pair(i, src_index));
  }
  return Status::OK();
}

// Given original output types and return value index mapping, return the new
// output types. Notice that DT_RESOURCE will be removed.
std::vector<DataType> ShuffleOutputDataTypeAttribute(
    const std::vector<DataType>& out_types,
    const std::map<int, int>& index_mapping) {
  std::vector<DataType> result(index_mapping.size());
  for (size_t i = 0; i < out_types.size(); i++) {
    auto iter = index_mapping.find(i);
    if (iter != index_mapping.end()) {
      result[iter->second] = out_types[i];
    }
  }
  return result;
}

// Given mapping between original output index and rearranged output index,
// change "index" attribute for _Retval nodes. Notice that DT_RESOURCE _Retval
// nodes will be removed.
void RearrangeRetvalNodes(
    const gtl::InlinedVector<Node*, 4>& ret_nodes,  // non-absl ok
    Graph* g, const std::map<int, int>& retval_index_mapping) {
  for (size_t i = 0; i < ret_nodes.size(); i++) {
    Node* n = ret_nodes[i];
    auto iter = retval_index_mapping.find(i);
    if (iter == retval_index_mapping.end()) {
      g->RemoveNode(n);
    } else {
      n->ClearAttr("index");
      n->AddAttr("index", iter->second);
    }
  }
}

// Given mapping between original input index and rearranged input index,
// reorder input edges for the node.
Status ReorderInputEdges(Graph* g, Node* n,
                         const std::vector<int>& index_mapping) {
  std::vector<const Edge*> input_edges;
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    input_edges.push_back(e);
  }
  for (const Edge* e : input_edges) {
    Node* src = e->src();
    int src_output = e->src_output();
    int dst_input = e->dst_input();
    int new_dst_input = index_mapping.at(dst_input);
    g->RemoveEdge(e);
    g->AddEdge(src, src_output, n, new_dst_input)->DebugString();
  }
  return Status::OK();
}

Status MaybeRewritePiplineStageNode(
    std::function<Status(const NameAttrList&, const FunctionBody**)>
        get_function_body_fn,
    Graph* g, Node* n, FunctionLibraryDefinition* fld, bool* node_rewritten) {
  // This node needs rewrite when either of these is true:
  // 1) Tin has DT_RESOURCE which requires rearrange;
  // 2) Tout has DT_RESOURCE.
  std::vector<DataType> in_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "Tin", &in_types));
  bool input_need_rearrange;
  int resource_input_count;
  std::vector<int> index_mapping;
  TF_RETURN_IF_ERROR(InputTypesNeedsRearrange(
      in_types, &input_need_rearrange, &resource_input_count, &index_mapping));
  std::vector<DataType> out_types;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "Tout", &out_types));
  bool has_resource_output = std::find(out_types.begin(), out_types.end(),
                                       DT_RESOURCE) != out_types.end();
  if (!input_need_rearrange && !has_resource_output) {
    *node_rewritten = false;
    return Status::OK();
  }

  *node_rewritten = true;

  if (input_need_rearrange) {
    // Reorder input edges.
    TF_RETURN_IF_ERROR(ReorderInputEdges(g, n, index_mapping));

    // Change Tin attribute.
    std::vector<DataType> new_in_types =
        ShuffleInputDataTypeAttribute(in_types, index_mapping);
    n->ClearAttr("Tin");
    n->AddAttr("Tin", new_in_types);
  }

  std::map<int, int> resource_retval_to_arg, retval_index_mapping;
  // Rewrite the to_apply function.
  NameAttrList f;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "to_apply", &f));
  const FunctionBody* fbody;
  TF_RETURN_IF_ERROR(get_function_body_fn(f, &fbody));

  if (input_need_rearrange) {
    // Change _Arg node index.
    RearrangeArgNodes(&fbody->arg_nodes, index_mapping);
  }

  if (has_resource_output) {
    // Resource _Retval must come from resource _Arg directly, or we do
    // not support it.
    TF_RETURN_IF_ERROR(CalculateRetvalRearrange(
        fbody->ret_nodes, &retval_index_mapping, &resource_retval_to_arg));

    // Change index for _Retval nodes.
    RearrangeRetvalNodes(fbody->ret_nodes, fbody->graph, retval_index_mapping);
  }

  // Save the new FunctionDef.
  FunctionDef new_fdef;
  string new_name =
      fld->UniqueFunctionName(absl::StrCat(f.name(), "_rearrange_"));
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*fbody->graph, new_name, &new_fdef));
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(new_fdef));

  // Change node to use rewritten function.
  f.set_name(new_name);
  n->ClearAttr("to_apply");
  n->AddAttr("to_apply", f);

  if (has_resource_output) {
    // Rearrange output edges.
    std::vector<const Edge*> out_edges;
    for (const Edge* e : n->out_edges()) {
      if (!e->IsControlEdge()) {
        out_edges.push_back(e);
      }
    }
    for (const Edge* e : out_edges) {
      Node* dst = e->dst();
      int dst_input = e->dst_input();
      int src_output = e->src_output();
      auto iter = retval_index_mapping.find(src_output);
      if (iter == retval_index_mapping.end()) {
        TF_RET_CHECK(resource_retval_to_arg.find(src_output) !=
                     resource_retval_to_arg.end());
        g->RemoveEdge(e);
        const Edge* input_edge;
        TF_RETURN_IF_ERROR(
            n->input_edge(resource_retval_to_arg.at(src_output), &input_edge));
        g->AddEdge(input_edge->src(), input_edge->src_output(), dst, dst_input);
      } else {
        g->RemoveEdge(e);
        g->AddEdge(n, iter->second, dst, dst_input);
      }
    }

    // Change Tout attribute for the node.
    std::vector<DataType> new_out_types =
        ShuffleOutputDataTypeAttribute(out_types, retval_index_mapping);
    n->ClearAttr("Tout");
    n->AddAttr("Tout", new_out_types);
  }
  return Status::OK();
}

}  // namespace

Status RearrangeFunctionArguments(
    std::function<Status(const NameAttrList&, const FunctionBody**)>
        get_function_body_fn,
    NameAttrList& new_func, const NameAttrList& old_func,
    FunctionLibraryDefinition* fld) {
  if (old_func.attr().size()) {
    return errors::FailedPrecondition(
        "Expected NameAttrList for the `to_apply` attribute to not have any "
        "attributes.");
  }

  const FunctionBody* func_body;
  TF_RETURN_IF_ERROR(get_function_body_fn(old_func, &func_body));

  // Create a new graph representing the Pipeline function.
  std::unique_ptr<Graph> graph(new Graph(fld));
  CopyGraph(*func_body->graph, graph.get());

  // Rewrite PiplineStage nodes.
  bool rewritten = false;
  for (Node* n : graph->nodes()) {
    // Rewrite both pipeline stages and the backward ones.
    if (absl::StartsWith(n->op_def().name(), "PipelineStage")) {
      bool node_rewritten = false;
      TF_RETURN_IF_ERROR(MaybeRewritePiplineStageNode(
          get_function_body_fn, graph.get(), n, fld, &node_rewritten));
      rewritten |= node_rewritten;
    }
  }
  if (rewritten) {
    VLOG(2) << "The Pipeline function was rewritten.";
  }

  // Save the new FunctionDef and use that for the Pipeline op.
  FunctionDef new_fdef;
  string new_name =
      fld->UniqueFunctionName(absl::StrCat(old_func.name(), "_rewritten_"));
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph.get(), new_name, &new_fdef));
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(new_fdef));
  new_func.set_name(new_name);

  return Status::OK();
}

}  // namespace tensorflow
