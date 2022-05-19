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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_schedule_optimizer.h"

#include <map>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

StatusOr<bool> AddDependenciesOnPreviousShard(
    const std::vector<HloInstruction*>& curr_shard,
    const std::vector<HloInstruction*>& prev_shard,
    const HloReachabilityMap& reachability_map) {
  bool added = false;

  for (HloInstruction* curr : curr_shard) {
    for (HloInstruction* prev : prev_shard) {
      if (reachability_map.IsReachable(curr, prev)) {
        return FailedPrecondition(
            "Unexpected dependency would cause cycle: %s -> %s",
            curr->ToString().c_str(), prev->ToString().c_str());
      }
      TF_RETURN_IF_ERROR(prev->AddControlDependencyTo(curr));
      added = true;
    }
  }

  return added;
}

StatusOr<bool> ScheduleByShard(std::vector<HloInstruction*> insts,
                               const HloReachabilityMap& reachability_map) {
  bool scheduled = false;

  std::map<int64_t, std::vector<HloInstruction*>> shard_to_insts;
  for (HloInstruction* inst : insts) {
    const auto shard = inst->sharding_unique_device();
    if (shard.has_value()) {
      shard_to_insts[shard.value()].push_back(inst);
    }
  }

  const std::vector<HloInstruction*>* prev_shard = nullptr;
  for (const auto& shard : shard_to_insts) {
    if (prev_shard != nullptr) {
      TF_ASSIGN_OR_RETURN(const bool added,
                          AddDependenciesOnPreviousShard(
                              shard.second, *prev_shard, reachability_map));
      scheduled |= added;
    }
    prev_shard = &shard.second;
  }

  return scheduled;
}

StatusOr<bool> ScheduleSendRecvs(HloComputation* comp,
                                 const OpSendRecvs& op_send_recvs) {
  bool scheduled = false;

  const auto reachability_map = HloReachabilityMap::Build(comp);

  for (const auto& op_send_recv : op_send_recvs) {
    const auto& op_name = op_send_recv.first;
    TF_ASSIGN_OR_RETURN(
        const bool send_scheduled,
        ScheduleByShard(op_send_recv.second.sends, *reachability_map));
    TF_ASSIGN_OR_RETURN(
        const bool recv_scheduled,
        ScheduleByShard(op_send_recv.second.recvs, *reachability_map));
    scheduled |= send_scheduled;
    scheduled |= recv_scheduled;
  }

  return scheduled;
}

}  // namespace

StatusOr<bool> HostComputeScheduleOptimizer::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      // Optimization: Popops fusion computations cannot have Send/Recv ops.
      continue;
    }

    const OpSendRecvs op_send_recvs = GroupSendRecvsByHostComputeOp(comp);
    TF_ASSIGN_OR_RETURN(const bool scheduled,
                        ScheduleSendRecvs(comp, op_send_recvs));
    changed |= scheduled;
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
