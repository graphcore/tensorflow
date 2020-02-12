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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/util.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace {
constexpr char kXlaCompileId[] = "_xla_compile_id";
constexpr char kXlaSeparateCompiledGradients[] =
    "_XlaSeparateCompiledGradients";
constexpr char kXlaCompile[] = "_XlaCompile";
constexpr char kXlaScope[] = "_XlaScope";

constexpr char kWhile[] = "While";
constexpr char kWhileBodyArg[] = "body";
}  // namespace

Status CopyXLAAttributesIfPresent(Node* from, Node* to) {
  for (auto& attr :
       {kXlaCompileId, kXlaSeparateCompiledGradients, kXlaCompile, kXlaScope}) {
    if (HasNodeAttr(from->def(), attr)) {
      to->AddAttr(attr, *from->attrs().Find(attr));
    }
  }
  return Status::OK();
}

Status CallForGraphFromFunctionDef(
    const NameAttrList& func, FunctionLibraryDefinition* flib_def,
    std::function<Status(Graph*, FunctionLibraryDefinition*)> fn,
    bool replace_function) {
  // Create a graph for the function def.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *flib_def->Find(func.name()), AttrSlice(&func.attr()), flib_def, &fbody));
  Graph* graph = fbody->graph;

  // Call the user function.
  TF_RETURN_IF_ERROR(fn(graph, flib_def));

  if (replace_function) {
    // Replace original function.
    FunctionDef replace_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph, func.name(), &replace_fdef));
    TF_RETURN_IF_ERROR(flib_def->ReplaceFunction(func.name(), replace_fdef));
  }
  return Status::OK();
}

Status CountNodesWithOpName(const string& name, Graph* graph,
                            FunctionLibraryDefinition* flib_def,
                            uint64* count) {
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kWhile) {
      NameAttrList body;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kWhileBodyArg, &body));
      // Call this function for any nested loops.
      auto fn = [name, count](Graph* graph,
                              FunctionLibraryDefinition* flib_def) -> Status {
        return CountNodesWithOpName(name, graph, flib_def, count);
      };

      TF_RETURN_IF_ERROR(
          CallForGraphFromFunctionDef(body, flib_def, std::move(fn)));
    }
    if (node->def().op() == name) {
      (*count)++;
    }
  }
  return Status::OK();
}

Status CallFunctionForEachNodeWithOpName(const string& name, Graph* graph,
                                         FunctionLibraryDefinition* flib_def,
                                         std::function<Status(Node*)> fn) {
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kWhile) {
      NameAttrList body;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kWhileBodyArg, &body));
      // Call this function for any nested loops.
      auto nested_fn = [name, fn](
                           Graph* graph,
                           FunctionLibraryDefinition* flib_def) -> Status {
        return CallFunctionForEachNodeWithOpName(name, graph, flib_def, fn);
      };

      TF_RETURN_IF_ERROR(
          CallForGraphFromFunctionDef(body, flib_def, std::move(nested_fn)));
    }
    if (node->def().op() == name) {
      TF_RETURN_IF_ERROR(fn(node));
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
