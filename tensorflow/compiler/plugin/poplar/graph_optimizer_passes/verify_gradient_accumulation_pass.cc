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

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/verify_gradient_accumulation_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/graph_optimizer_passes/util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {
constexpr char kIpuStatefulGradientAccumulate[] =
    "IpuStatefulGradientAccumulate";
constexpr char kResourceApplyGradientDescent[] = "ResourceApplyGradientDescent";
constexpr char kResourceApplyMomentum[] = "ResourceApplyMomentum";
constexpr char kVerifyUsage[] = "verify_usage";

StatusOr<bool> VerifyGraph(Graph* graph, FunctionLibraryDefinition* flib_def) {
  // Call this function for any nested loops.
  TF_RETURN_IF_ERROR(
      CallFunctionForWhileLoopBodies(graph, flib_def, VerifyGraph).status());

  for (Node* node : graph->op_nodes()) {
    if (node->def().op() == kIpuStatefulGradientAccumulate) {
      // Go through all the output edges - we only allow up to one data edge.
      Node* user = nullptr;
      uint64 num_users = 0;
      for (const Edge* edge : node->out_edges()) {
        // Ignore control edges.
        if (edge->IsControlEdge()) {
          continue;
        }
        user = edge->dst();
        num_users++;
      }

      if (num_users == 0) {
        // If it has no users then it is fine.
        continue;
      } else if (num_users > 1) {
        const std::string error_msg = absl::StrCat(
            "The ", node->name(), " op (", node->def().op(), " optype) has ",
            num_users,
            " users which is not supported. This usually occurs when the "
            "`GradientAccumulationOptimizer` is used with another optimizer "
            "which is not supported. Please note that "
            "`GradientAccumulationOptimizer` is currently only supported with "
            "`GradientDescentOptimizer` and `MomentumOptimizer` optimizers. "
            "For any other optimizers use `GradientAccumulationOptimizerV2`.");
        return errors::FailedPrecondition(error_msg);
      }

      // Check that the user is a supported op type.
      bool user_supported = false;
      for (auto& supported_op :
           {kResourceApplyGradientDescent, kResourceApplyMomentum}) {
        if (user->def().op() == supported_op) {
          user_supported = true;
          break;
        }
      }

      if (!user_supported) {
        bool verify_usage;
        TF_CHECK_OK(GetNodeAttr(node->attrs(), kVerifyUsage, &verify_usage));
        if (verify_usage) {
          const std::string error_msg = absl::StrCat(
              "The ", node->name(), " op (", node->def().op(),
              " optype) has user op ", user->name(), " (", user->def().op(),
              " optype) which is not supported.\n"
              "Please use `GradientAccumulationOptimizerV2`.");
          return errors::FailedPrecondition(error_msg);
        } else {
          VLOG(1) << "Detected unsafe usage of "
                  << kIpuStatefulGradientAccumulate;
        }
      }
    }
  }

  return false;
}
}  // namespace

Status VerifyGradientAccumulationPass::Run(
    const GraphOptimizationPassOptions& options) {
  FunctionLibraryDefinition* flib_def = options.flib_def;
  TF_RET_CHECK(flib_def != nullptr);
  Graph* graph = options.graph->get();
  TF_RETURN_IF_ERROR(VerifyGraph(graph, flib_def).status());
  return Status::OK();
}
}  // namespace tensorflow
