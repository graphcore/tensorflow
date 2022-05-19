/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/kernels/application_runtime/resource_handle_pruner.h"

#include <map>
#include <memory>

#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"

namespace tensorflow {

ResourceHandlePruner::ResourceHandlePruner(FunctionLibraryRuntime* flib)
    : flib_(flib),
      flib_def_(const_cast<FunctionLibraryDefinition*>(
          flib_->GetFunctionLibraryDefinition())) {}

Status ResourceHandlePruner::Run(const NameAttrList& function) {
  return Run(function.name(), AttrSlice(&function.attr()));
}

Status ResourceHandlePruner::Run(const std::string& function_name,
                                 const AttrSlice& attrs) {
  if (IsVisited(function_name)) return Status::OK();

  visited_.insert(function_name);

  const FunctionDef* fdef = CHECK_NOTNULL(flib_def_)->Find(function_name);
  std::unique_ptr<FunctionBody> fbody = nullptr;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, attrs, flib_def_, &fbody));
  const IndicesVector resource_arg_indices =
      GetResourceIndices(fbody->arg_types);

  if (resource_arg_indices.empty()) return Status::OK();

  const std::set<Node*> resource_dst_nodes =
      GetResourceDstNodes(*fbody, resource_arg_indices);
  const std::unordered_set<std::string> resource_names =
      GetResourceNames(*fbody, resource_arg_indices);

  bool ret_value_from_resource = false;

  for (Node* node : resource_dst_nodes) {
    if (node->IsRetval()) {
      ret_value_from_resource = true;
      continue;
    }

    std::vector<NameAttrList> functions;
    TF_RETURN_IF_ERROR(GetFunctionNamesAndAttrs(node, &functions));

    for (const auto& function : functions) {
      if (!function.name().empty()) {
        TF_RETURN_IF_ERROR(Run(function));
      }
    }

    UpdateNodeInputTypes(node);
    RemoveNodeDefInputs(node->properties()->node_def, resource_names);
  }

  RemoveResourceNodes(*fbody, resource_arg_indices);
  UpdateNonResourceArgsIndices(*fbody);
  UpdateNonResourceArgsEdges(*fbody, resource_dst_nodes);

  if (ret_value_from_resource) {
    RemoveRetvalFromResource(*fbody);
  }

  FunctionDef fdef_optimized;
  TF_RETURN_IF_ERROR(
      ConvertGraphToFunctionDef(*fbody, *fdef, function_name, &fdef_optimized));

  fbody.release();
  flib_def_->ReplaceFunction(function_name, fdef_optimized);
  return Status::OK();
}

Status ResourceHandlePruner::ConvertGraphToFunctionDef(
    const FunctionBody& fbody, const FunctionDef& fdef_old,
    const std::string& function_name, FunctionDef* fdef) const {
  std::unordered_set<std::string> control_ret_names;
  for (const auto& control_ret : fdef_old.control_ret()) {
    control_ret_names.insert(control_ret.first);
    control_ret_names.insert(control_ret.second);
  }

  if (!control_ret_names.empty()) {
    std::function<absl::optional<string>(const Node*)> control_ret =
        [&](const Node* node) -> absl::optional<string> {
      const auto it = control_ret_names.find(node->name());
      if (it != control_ret_names.cend()) return *it;

      return absl::optional<string>();
    };

    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*fbody.graph, function_name, control_ret, fdef));
  } else {
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*fbody.graph, function_name, fdef));
  }

  return Status::OK();
}

void ResourceHandlePruner::RemoveResourceNodes(
    const FunctionBody& fbody,
    const IndicesVector& resource_arg_indices) const {
  for (const auto index : resource_arg_indices) {
    fbody.graph->RemoveNode(fbody.arg_nodes[index]);
  }
}

void ResourceHandlePruner::UpdateNonResourceArgsIndices(
    const FunctionBody& fbody) const {
  static constexpr const char* const kIndexAttr = "index";

  int64_t index = 0;
  for (std::size_t i = 0; i < fbody.arg_types.size(); ++i) {
    if (fbody.arg_types[i] != DT_RESOURCE) {
      Node* node = fbody.arg_nodes[i];
      if (node->attrs().Find(kIndexAttr)) {
        node->AddAttr(kIndexAttr, index);
        ++index;
      }
    }
  }
}

void ResourceHandlePruner::UpdateNonResourceArgsEdges(
    const FunctionBody& fbody, const std::set<Node*> resource_dst_nodes) const {
  for (const Node* const node : resource_dst_nodes) {
    std::map<int, const Edge*> dst_input_to_edge;
    for (const Edge* edge : node->in_edges()) {
      dst_input_to_edge.emplace(edge->dst_input(), edge);
    }
    int index = 0;

    for (const auto& dst_id_with_edge : dst_input_to_edge) {
      const auto& dst_id = dst_id_with_edge.first;
      if (dst_id <= Graph::kControlSlot) continue;
      if (dst_id != index) {
        const Edge* const edge = dst_id_with_edge.second;
        Node* src_node = edge->src();
        Node* dst_node = edge->dst();
        const int src_output = edge->src_output();
        fbody.graph->RemoveEdge(edge);
        fbody.graph->AddEdge(src_node, src_output, dst_node, index);
      }

      ++index;
    }
  }
}

DataTypeVector ResourceHandlePruner::FilterResourceDataTypes(
    const DataTypeVector& src) const {
  DataTypeVector result;
  result.reserve(src.size());

  std::copy_if(src.cbegin(), src.cend(), std::back_inserter(result),
               [](const auto& data_type) { return data_type != DT_RESOURCE; });
  return result;
}

void ResourceHandlePruner::UpdateNodeInputTypes(Node* node) {
  static constexpr const char* const kInputTypesInputAttr = "Tin";
  static constexpr const char* const kInputTypesAttr = "T";

  for (const auto& typeAttr : {kInputTypesInputAttr, kInputTypesAttr}) {
    const auto& attrs = node->attrs();
    if (attrs.Find(typeAttr)) {
      DataTypeVector types;
      GetNodeAttr(attrs, typeAttr, &types);
      node->AddAttr(typeAttr, FilterResourceDataTypes(types));
    }
  }
}

void ResourceHandlePruner::RemoveNodeDefInputs(
    NodeDef& node_def,
    const std::unordered_set<std::string>& resource_names) const {
  const auto inputs_old = node_def.input();
  node_def.clear_input();
  for (const std::string& input : inputs_old) {
    if (resource_names.find(input) == resource_names.cend()) {
      node_def.add_input(input);
    }
  }
}

void ResourceHandlePruner::RemoveRetvalFromResource(
    const FunctionBody& fbody) const {
  static constexpr const char* const kIndexAttr = "index";

  int64_t index = 0;
  for (Node* node : fbody.ret_nodes) {
    if (node->in_edges().empty()) {
      fbody.graph->RemoveNode(node);
    } else if (node->attrs().Find(kIndexAttr)) {
      node->AddAttr(kIndexAttr, index);
      ++index;
    }
  }
}

std::set<Node*> ResourceHandlePruner::GetResourceDstNodes(
    const FunctionBody& fbody, const IndicesVector& resource_indices) const {
  std::set<Node*> resource_dst_nodes;

  for (const auto index : resource_indices) {
    const Node* resource_node = fbody.arg_nodes[index];
    for (const Edge* edge : resource_node->out_edges()) {
      Node* dstNode = edge->dst();
      if (!dstNode->IsSink()) {
        resource_dst_nodes.insert(dstNode);
      }
    }
  }

  return resource_dst_nodes;
}

std::unordered_set<std::string> ResourceHandlePruner::GetResourceNames(
    const FunctionBody& fbody, const IndicesVector& resource_indices) const {
  std::unordered_set<std::string> resource_names;
  for (const auto index : resource_indices) {
    resource_names.insert(fbody.arg_nodes[index]->name());
  }

  return resource_names;
}

Status ResourceHandlePruner::GetFunctionNamesAndAttrs(
    const Node* node, std::vector<NameAttrList>* func) const {
  if (node->IsFunctionCall()) {
    TF_RETURN_IF_ERROR(
        GetFunctionCallNameAndAttr(*node, &(*func->emplace(func->cend()))));
  } else {
    for (const auto& attr : node->attrs()) {
      const AttrValue& value = attr.second;
      if (value.has_func()) {
        func->emplace_back(value.func());
      }
    }
  }

  return Status::OK();
}

Status ResourceHandlePruner::GetFunctionNameAndAttr(const Node& node,
                                                    const char* func_field_name,
                                                    NameAttrList* func) const {
  const AttrValue* attr_value;
  TF_RETURN_IF_ERROR(node.attrs().Find(func_field_name, &attr_value));
  if (!attr_value->has_func()) {
    return errors::InvalidArgument(
        "The attribute value for attribute '", func_field_name, "' in node ",
        node.DebugString(), " does not have 'func' field set");
  }
  *func = attr_value->func();
  return Status::OK();
}

Status ResourceHandlePruner::GetFunctionCallNameAndAttr(
    const Node& node, NameAttrList* func) const {
  if (node.IsPartitionedCall()) {
    TF_RETURN_IF_ERROR(GetFunctionNameAndAttr(
        node, FunctionLibraryDefinition::kFuncAttr, func));
    return Status::OK();
  }

  if (flib_->GetFunctionLibraryDefinition()->Find(node.def().op())) {
    func->set_name(node.type_string());
  } else {
    func->set_name(FunctionLibraryDefinition::kGradientOp);
  }
  *func->mutable_attr() = node.def().attr();
  return Status::OK();
}

bool ResourceHandlePruner::IsVisited(const std::string& function_name) const {
  return visited_.find(function_name) != visited_.cend();
}
}  // namespace tensorflow
