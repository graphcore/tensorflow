/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_ipu_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<HloInstruction*> InsertInterIpuCopy(
    HloInstruction* inst, const HloSharding& output_sharding) {
  HloComputation* comp = inst->parent();

  std::vector<int64> vec;

  vec = GetShardingDeviceIdVector(output_sharding);
  std::set<int64> output_devices(vec.begin(), vec.end());

  vec = GetShardingDeviceIdVector(inst->sharding());
  std::set<int64> input_devices(vec.begin(), vec.end());

  if (input_devices.size() > 1 || output_devices.size() > 1) {
    std::vector<HloInstruction*> instructions;
    int64 tuple_count = ShapeUtil::TupleElementCount(inst->shape());
    auto& input_sharding = inst->sharding();
    for (int64 i = 0; i < tuple_count; i++) {
      auto output_sub_sharding =
          output_sharding.GetSubSharding(inst->shape(), ShapeIndex({i}));
      auto input_sub_sharding =
          input_sharding.GetSubSharding(inst->shape(), ShapeIndex({i}));
      const auto& element_shape =
          ShapeUtil::GetTupleElementShape(inst->shape(), i);
      // Add GTE, set its sharding
      auto* gte = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, inst, i));
      gte->set_sharding(input_sub_sharding);
      if (input_sub_sharding != output_sub_sharding) {
        TF_ASSIGN_OR_RETURN(HloInstruction * copy,
                            InsertInterIpuCopy(gte, output_sub_sharding));
        instructions.push_back(copy);
      } else {
        instructions.push_back(gte);
      }
    }
    auto* tuple =
        comp->AddInstruction(HloInstruction::CreateTuple(instructions));
    tuple->set_sharding(output_sharding);
    return tuple;
  }

  HloInstruction* new_inst;
  if (inst->opcode() == HloOpcode::kConstant || IsWideConstant(inst) ||
      IsPoplarInstruction(PoplarOp::ExecutionCounter)(inst)) {
    new_inst = comp->AddInstruction(inst->Clone());
  } else {
    new_inst = comp->AddInstruction(CreateInterIpuCopy({inst}));
  }

  new_inst->set_sharding(output_sharding);
  return new_inst;
}

}  // namespace

using UserAndParam = std::pair<HloInstruction*, int>;

StatusOr<bool> InterIpuCopyInserter::Run(HloModule* module) {
  if (!HaveSharding(module)) {
    return false;
  }

  bool added = false;

  auto is_ineligible_op = [](const HloInstruction* inst) {
    // These ops are expected to have their input(s) on a different device to
    // their output(s).
    // Dont perform an InterIpuCopy for WithinReplica operands since they
    // will do their own copies.
    return inst->opcode() == HloOpcode::kAfterAll ||
           IsPoplarInstruction(PoplarOp::InterIpuCopy)(inst) ||
           IsGCLWithinReplicaOp(inst);
  };

  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Add InterIpuCopy instructions between nodes which are on different
    // devices
    auto original_insts = comp->MakeInstructionPostOrder();
    for (auto* inst : original_insts) {
      if (!inst->has_sharding() || is_ineligible_op(inst)) {
        continue;
      }

      const auto& src_sharding = GetShardingOfOutputTensor(inst);

      // Construct a map from the sharding of the tensor users' operand inputs
      // (represented as a vector of int64 values, to all users and operand
      // indices.
      std::multimap<std::vector<int64>, UserAndParam> dst_sharding_map;
      std::set<std::vector<int64>> dst_shardings;
      std::map<std::vector<int64>, HloSharding> sharding_map;
      for (const auto& user : inst->users()) {
        if (is_ineligible_op(user)) {
          continue;
        }

        if (user->opcode() == HloOpcode::kGetTupleElement) {
          // GTEs should always have the same sharding as the tuple
          const auto& s = inst->sharding();
          const auto& tuple_sub_sharding =
              s.IsTuple()
                  ? s.GetSubSharding(inst->shape(), {user->tuple_index()})
                  : s;
          if (tuple_sub_sharding != user->sharding()) {
            return InternalError(
                "Different sharding on Tuple and GTE: %s != %s",
                inst->ToString(), user->ToString());
          }
          continue;
        }

        for (int operand = 0; operand < user->operand_count(); operand++) {
          if (user->operand(operand) == inst) {
            const auto& dst_sharding = GetShardingForOperand(user, operand);

            std::vector<int64> sharding_vector =
                GetShardingDeviceIdVector(dst_sharding);

            if (src_sharding != dst_sharding) {
              auto u = std::make_pair(user, operand);
              dst_sharding_map.insert(std::make_pair(sharding_vector, u));
              dst_shardings.insert(sharding_vector);
              sharding_map.insert(
                  std::make_pair(sharding_vector, dst_sharding));
            }
          }
        }
      }

      // For each unique destination sharding that is not the same as the
      // sharding of the source of the tensors, add an inter-ipu copy to move
      // the tensors to the other devices.
      for (auto s : dst_shardings) {
        added = true;

        auto sharding = sharding_map.at(s);

        TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                            InsertInterIpuCopy(inst, sharding));

        auto range = dst_sharding_map.equal_range(s);
        for (auto user = range.first; user != range.second; ++user) {
          auto* u = user->second.first;
          auto o = user->second.second;
          TF_RETURN_IF_ERROR(u->ReplaceOperandWith(o, new_inst));
        }
      }
    }
  }

  return added;
}

InterIpuCopyInserter::InterIpuCopyInserter() {}

}  // namespace poplarplugin
}  // namespace xla
