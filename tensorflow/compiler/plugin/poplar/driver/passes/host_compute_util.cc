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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_recv_barrier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {

namespace {

static constexpr char kHostComputeOp[] = "XlaHostCompute";

}  // namespace

OpSendRecvs GroupSendRecvsByHostComputeOp(const HloComputation* comp) {
  OpSendRecvs result;

  auto is_send = IsPoplarInstruction(SendToHost);
  auto is_recv = IsPoplarInstruction(RecvFromHost);

  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (inst->metadata().op_type() == kHostComputeOp) {
      const auto& op_name = inst->metadata().op_name();
      if (is_send(inst)) {
        result[op_name].sends.push_back(inst);
      } else if (is_recv(inst)) {
        result[op_name].recvs.push_back(inst);
      }
    }
  }

  return result;
}

}  // namespace poplarplugin
}  // namespace xla
