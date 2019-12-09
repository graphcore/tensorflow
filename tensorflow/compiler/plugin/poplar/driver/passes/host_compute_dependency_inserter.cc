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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_dependency_inserter.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

static constexpr char kHostComputeOp[] = "XlaHostCompute";

struct SendRecvs {
  std::vector<HloSendDoneInstruction*> sends;
  std::vector<HloRecvDoneInstruction*> recvs;
};

using OpSendRecvs = std::unordered_map<string, SendRecvs>;

StatusOr<OpSendRecvs> GroupSendRecvsByHostComputeOp(const HloModule* module) {
  OpSendRecvs result;

  for (HloComputation* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      // Optimization: Popops fusion computations cannot have Send/Recv ops.
      continue;
    }

    for (HloInstruction* inst : comp->instructions()) {
      if (inst->metadata().op_type() == kHostComputeOp) {
        const auto& op_name = inst->metadata().op_name();
        if (inst->opcode() == HloOpcode::kSendDone) {
          result[op_name].sends.push_back(Cast<HloSendDoneInstruction>(inst));
        } else if (inst->opcode() == HloOpcode::kRecvDone) {
          result[op_name].recvs.push_back(Cast<HloRecvDoneInstruction>(inst));
        }
      }
    }
  }

  return result;
}

Status AddControlDependencies(const OpSendRecvs& op_send_recvs) {
  for (const auto& op_send_recv : op_send_recvs) {
    const auto& op_name = op_send_recv.first;
    const auto& sends = op_send_recv.second.sends;
    const auto& recvs = op_send_recv.second.recvs;

    VLOG(2) << "Found " << sends.size() << " sends and " << recvs.size()
            << " recvs from the op " << op_name;

    for (auto* send : sends) {
      for (auto* recv : recvs) {
        TF_RETURN_IF_ERROR(send->AddControlDependencyTo(recv));
      }
    }
  }

  return Status::OK();
}

}  // namespace

StatusOr<bool> HostComputeDependencyInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(const auto send_recvs,
                      GroupSendRecvsByHostComputeOp(module));
  if (send_recvs.empty()) {
    return false;
  }
  TF_RETURN_IF_ERROR(AddControlDependencies(send_recvs));
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
