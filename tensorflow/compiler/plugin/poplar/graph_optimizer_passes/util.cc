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

#include <list>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
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

constexpr char kStatelessWhile[] = "StatelessWhile";
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

StatusOr<bool> CallFunctionForWhileLoopBodies(
    Graph* graph, FunctionLibraryDefinition* flib_def,
    std::function<StatusOr<bool>(Graph*, FunctionLibraryDefinition*)> fn) {
  std::list<Node*> while_loop_nodes;
  // Optimize any while loops as well.
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kStatelessWhile || node->def().op() == kWhile) {
      while_loop_nodes.push_back(node);
    }
  }

  bool changed = false;
  for (Node* while_loop : while_loop_nodes) {
    // Get the body of the loop.
    NameAttrList body;
    TF_RETURN_IF_ERROR(GetNodeAttr(while_loop->attrs(), kWhileBodyArg, &body));

    // Create a graph for the function def.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*flib_def->Find(body.name()),
                                               AttrSlice(&body.attr()),
                                               flib_def, &fbody));
    Graph* body_graph = fbody->graph;
    // Call the body function.
    TF_ASSIGN_OR_RETURN(bool loop_changed, fn(body_graph, flib_def));

    if (loop_changed) {
      changed = true;
      // Add a new function - do not modify any other call sites.
      string new_name = flib_def->UniqueFunctionName(body.name());
      body.set_name(new_name);
      FunctionDef new_fdef;
      TF_RETURN_IF_ERROR(
          GraphToFunctionDef(*body_graph, body.name(), &new_fdef));
      TF_RETURN_IF_ERROR(flib_def->AddFunctionDef(new_fdef));
      // Make sure the while loop now uses the new function.
      while_loop->ClearAttr(kWhileBodyArg);
      while_loop->AddAttr(kWhileBodyArg, body);
    }
  }

  return changed;
}

Status CountNodesWithOpName(const string& name, Graph* graph,
                            FunctionLibraryDefinition* flib_def,
                            uint64* count) {
  // Call this function for any nested loops.
  auto fn = [name, count](
                Graph* graph,
                FunctionLibraryDefinition* flib_def) -> StatusOr<bool> {
    TF_RETURN_IF_ERROR(CountNodesWithOpName(name, graph, flib_def, count));
    return false;
  };
  TF_RETURN_IF_ERROR(
      CallFunctionForWhileLoopBodies(graph, flib_def, std::move(fn)).status());

  // Count nodes in this graph.
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == name) {
      (*count)++;
    }
  }
  return Status::OK();
}

Status CallFunctionForEachNodeWithOpName(const string& name, Graph* graph,
                                         FunctionLibraryDefinition* flib_def,
                                         std::function<Status(Node*)> fn) {
  // Call this function for any nested loops.
  auto nested_fn = [name, fn](
                       Graph* graph,
                       FunctionLibraryDefinition* flib_def) -> StatusOr<bool> {
    TF_RETURN_IF_ERROR(
        CallFunctionForEachNodeWithOpName(name, graph, flib_def, fn));
    return false;
  };

  TF_RETURN_IF_ERROR(
      CallFunctionForWhileLoopBodies(graph, flib_def, std::move(nested_fn))
          .status());

  // Evaluate nodes in this graph.
  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == name) {
      TF_RETURN_IF_ERROR(fn(node));
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
