/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_control_dependency_inserter.h"

#include <list>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
std::vector<HloInstruction*> GetMultiDeviceStages(
    const PipelineStages& pipeline_stages) {
  auto is_par_stage = [](HloInstruction* stage) -> bool {
    if (stage->has_sharding()) {
      return stage->sharding().GetUniqueDevice() == Devices::All;
    }

    return false;
  };

  std::vector<HloInstruction*> stages;
  absl::c_copy_if(pipeline_stages.forward, std::back_inserter(stages),
                  is_par_stage);
  absl::c_copy_if(pipeline_stages.backward, std::back_inserter(stages),
                  is_par_stage);

  return stages;
}

std::vector<HloInstruction*> SelectPipelineStages(
    const std::vector<HloInstruction*>& insts) {
  std::vector<HloInstruction*> result;
  absl::c_copy_if(insts, std::back_inserter(result),
                  IsPipelineStageOrBackwardOp);

  return result;
}

std::vector<HloInstruction*> SelectInterTilesetCopies(
    const std::vector<HloInstruction*>& insts) {
  std::vector<HloInstruction*> result;
  absl::c_copy_if(insts, std::back_inserter(result),
                  IsPoplarInstruction(PoplarOp::IpuInterCopy));

  return result;
}

/**
 * Check that if there is a path between instructions in A and B, that they are
 * only reachable from A to B.
 */
template <typename IIter>
Status CheckAllReachable(const HloReachabilityMap& reachability_map,
                         IIter a_begin, IIter a_end, IIter b_begin,
                         IIter b_end) {
  for (IIter a_itr = a_begin; a_itr != a_end; ++a_itr) {
    for (IIter b_itr = b_begin; b_itr != b_end; ++b_itr) {
      if (reachability_map.IsConnected(*a_itr, *b_itr) &&
          !reachability_map.IsReachable(*a_itr, *b_itr)) {
        return InvalidArgument("Instruction '%s' is not reachable from '%s'.",
                               (*b_itr)->name(), (*a_itr)->name());
      }
    }
  }

  return Status::OK();
}
}  // namespace

StatusOr<bool> PipelineControlDependencyInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);

  // Collect all the stages.
  TF_ASSIGN_OR_RETURN(
      PipelineStages pipeline_stages,
      GetPipelineStages(pipeline_ops.back()->to_apply(), false));

  // Only select the stages that span multiple IPUs.
  std::vector<HloInstruction*> stages = GetMultiDeviceStages(pipeline_stages);

  bool changed = false;
  for (auto stage : stages) {
    // Get the stage instructions in post-order.
    auto insts = stage->to_apply()->MakeInstructionPostOrder();

    // Select the sub-stages
    auto sub_stages = SelectPipelineStages(insts);

    // Select the inter-ipu-copies
    auto inter_ipu_copies = SelectInterTilesetCopies(insts);

    // Partition the inter ipu copies based on whether they are before the
    // substages or after.
    auto has_ipu_sharding = [](const HloInstruction* inter_ipu_copy) -> bool {
      return inter_ipu_copy->sharding().GetUniqueDevice() != Devices::All;
    };
    auto itr = absl::c_stable_partition(inter_ipu_copies, has_ipu_sharding);

    // Build a reachability map within the pipeline stage.
    auto reachability = HloReachabilityMap::Build(stage->to_apply());

    // Check that all the substages are reachable from all the copies in.
    TF_RETURN_IF_ERROR(CheckAllReachable(*reachability,
                                         inter_ipu_copies.begin(), itr,
                                         sub_stages.begin(), sub_stages.end()));

    // Check that all the copies out are reachable from all the substages.
    TF_RETURN_IF_ERROR(CheckAllReachable(*reachability, sub_stages.begin(),
                                         sub_stages.end(), itr,
                                         inter_ipu_copies.end()));

    // For each inter-ipu copy, add control dependencies that gaurantee that it
    // will only be before or after the computation stages. Never between the
    // stages.
    for (auto inter_ipu_copy : inter_ipu_copies) {
      if (has_ipu_sharding(inter_ipu_copy)) {
        for (auto sub_stage : sub_stages) {
          TF_RETURN_IF_ERROR(inter_ipu_copy->AddControlDependencyTo(sub_stage));
          changed = true;
        }
      } else {
        for (auto sub_stage : sub_stages) {
          TF_RETURN_IF_ERROR(sub_stage->AddControlDependencyTo(inter_ipu_copy));
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
