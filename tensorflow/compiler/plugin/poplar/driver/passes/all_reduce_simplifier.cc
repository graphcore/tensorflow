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

#include "tensorflow/compiler/plugin/poplar/driver/passes/all_reduce_simplifier.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

// Convert add(x, replica-normalise()) to add(x, mul(y, 1 / replication-factor))

namespace xla {
namespace poplarplugin {

namespace {

namespace m = match;

StatusOr<bool> HandleAllReduce(HloInstruction* all_reduce,
                               uint32 replication_factor) {
  auto* all_reduce_computation = all_reduce->to_apply();

  // Match `add(x, replica-normalise(y))` and replace it with
  // `add(x, mul(y, 1 / replication-factor))`
  if (Match(
          all_reduce_computation->root_instruction(),
          m::Add(m::Parameter(0), m::CustomCall(m::Parameter(1))
                                      .WithCustomCallTarget(PoplarOp_Name(
                                          PoplarOp::ReplicationNormalise))))) {
    auto* accumulator =
        all_reduce_computation->root_instruction()->mutable_operand(0);
    auto* replica_normalise =
        all_reduce_computation->root_instruction()->mutable_operand(1);
    auto* to_normalise =
        const_cast<HloInstruction*>(replica_normalise->mutable_operand(0));

    CHECK_EQ(accumulator->shape(), to_normalise->shape());

    TF_ASSIGN_OR_RETURN(Literal literal,
                        LiteralUtil::CreateR0(replication_factor)
                            .Convert(to_normalise->shape().element_type()));

    HloInstruction* scalec = all_reduce_computation->AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));

    auto* normalised =
        all_reduce_computation->AddInstruction(HloInstruction::CreateBinary(
            to_normalise->shape(), HloOpcode::kDivide, to_normalise, scalec));

    TF_RETURN_IF_ERROR(all_reduce_computation->ReplaceInstruction(
        replica_normalise, normalised));
    return true;
  }

  return false;
}

};  // namespace

AllReduceSimplifier::AllReduceSimplifier(uint32 replication_factor)
    : replication_factor_(replication_factor) {}

StatusOr<bool> AllReduceSimplifier::Run(HloModule* module) {
  bool changed = false;
  for (auto* computation : module->MakeComputationPostOrder()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        TF_ASSIGN_OR_RETURN(bool inst_changed,
                            HandleAllReduce(instruction, replication_factor_));
        changed |= inst_changed;
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
