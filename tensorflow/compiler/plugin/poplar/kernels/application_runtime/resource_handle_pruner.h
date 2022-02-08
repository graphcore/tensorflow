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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_APPLICATION_RUNTIME_RESOURCE_HANDLE_PRUNER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_APPLICATION_RUNTIME_RESOURCE_HANDLE_PRUNER_H_

#include <cstddef>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class AttrSlice;
class FunctionBody;
class FunctionDef;
class FunctionLibraryDefinition;
class FunctionLibraryRuntime;
class NameAttrList;
class Node;
class NodeDef;

/*
 * Pass solves issue with caputring resource EagerTensors in
 * ApplicationCompileOp, when freeze_variables option is set. The source of the
 * problem is the way the FuncGraph works. With capture_by_value option set,
 * FuncGraph creates Placeholders for captured resources immediately on capture
 * without determining if they're ultimately used inside the FuncGraph. In the
 * next step resource is captured by value, but dangling placeholder remains. As
 * the result of this behavior, despite setting freeze_variables user is
 * obligated to pass inputs in embedded_runtime_start.
 *
 * Pass goes recursively through FuncDefs and removes input resource handlers.
 */
class ResourceHandlePruner {
 public:
  explicit ResourceHandlePruner(FunctionLibraryRuntime* flib);
  ~ResourceHandlePruner() = default;
  TF_DISALLOW_COPY_AND_ASSIGN(ResourceHandlePruner);
  ResourceHandlePruner(ResourceHandlePruner&&) = delete;
  ResourceHandlePruner& operator=(ResourceHandlePruner&&) = delete;

  Status Run(const NameAttrList& root_function);

 private:
  using IndicesVector = std::vector<std::size_t>;

  Status Run(const std::string& function_name, const AttrSlice& attrs);

  Status ConvertGraphToFunctionDef(const FunctionBody& fbody,
                                   const FunctionDef& fdef_old,
                                   const std::string& function_name,
                                   FunctionDef* fdef) const;

  std::set<Node*> GetResourceDstNodes(
      const FunctionBody& fbody, const IndicesVector& resource_indices) const;
  std::unordered_set<std::string> GetResourceNames(
      const FunctionBody& fbody, const IndicesVector& resource_indices) const;
  Status GetFunctionNamesAndAttrs(const Node* node,
                                  std::vector<NameAttrList>* func) const;
  Status GetFunctionNameAndAttr(const Node& node, const char* func_field_name,
                                NameAttrList* func) const;
  Status GetFunctionCallNameAndAttr(const Node& node, NameAttrList* func) const;
  template <typename DataTypeContainer>
  IndicesVector GetResourceIndices(
      const DataTypeContainer& dt_container) const {
    IndicesVector resource_arg_indices;
    const auto dt_container_size = dt_container.size();
    resource_arg_indices.reserve(dt_container_size);

    for (std::size_t i = 0; i < dt_container_size; ++i) {
      if (dt_container[i] == DT_RESOURCE) resource_arg_indices.push_back(i);
    }

    return resource_arg_indices;
  }

  void UpdateNonResourceArgsIndices(const FunctionBody& fbody) const;
  void UpdateNonResourceArgsEdges(
      const FunctionBody& fbody,
      const std::set<Node*> resource_dst_nodes) const;
  void UpdateNodeInputTypes(Node* node);

  void RemoveNodeDefInputs(
      NodeDef& node_def,
      const std::unordered_set<std::string>& resource_names) const;
  void RemoveRetvalFromResource(const FunctionBody& fbody) const;
  void RemoveResourceNodes(const FunctionBody& fbody,
                           const IndicesVector& resource_arg_indices) const;

  DataTypeVector FilterResourceDataTypes(const DataTypeVector& src) const;
  bool IsVisited(const std::string& function_name) const;

  FunctionLibraryRuntime* flib_;
  FunctionLibraryDefinition* flib_def_;
  std::unordered_set<std::string> visited_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_APPLICATION_RUNTIME_RESOURCE_HANDLE_PRUNER_H_
