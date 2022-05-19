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
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
namespace {
constexpr char kReplicatedParameterLoadFusionName[] =
    "_pop_replicated_parameter_load_fusion";
constexpr char kReplicatedParameterStoreFusionName[] =
    "_pop_replicated_parameter_store_fusion";
}  // namespace

std::string GetReplicatedParameterLoadFusionName() {
  return kReplicatedParameterLoadFusionName;
}

std::string GetReplicatedParameterStoreFusionName() {
  return kReplicatedParameterStoreFusionName;
}

bool IsReplicatedParameterLoadFusion(const HloInstruction* inst) {
  return IsFusion(inst, kReplicatedParameterLoadFusionName);
}

bool IsReplicatedParameterStoreFusion(const HloInstruction* inst) {
  return IsFusion(inst, kReplicatedParameterStoreFusionName);
}

bool IsReplicatedParameterLoad(const HloInstruction* inst) {
  return IsReplicatedParameterLoadFusion(inst) &&
         IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst->operand(0));
}

bool IsReplicatedParameterStore(const HloInstruction* inst) {
  return IsReplicatedParameterStoreFusion(inst) && inst->user_count() == 1 &&
         IsPoplarInstruction(PoplarOp::RemoteParameterStore)(inst->users()[0]);
}

bool IsGradientAccumulatorSink(const HloInstruction* inst) {
  return IsPoplarInstruction(GradientAccumulatorSink, inst);
}

bool IsReshape(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kReshape;
}

bool IsRemoteBufferStore(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterStore, inst) ||
         IsPoplarInstruction(BufferStoreSlice, inst);
}

bool IsRemoteBufferPassthrough(const HloInstruction* inst) {
  return IsRemoteBufferStore(inst) || IsGradientAccumulatorSink(inst) ||
         IsReshape(inst);
}

const Shape GetReplicatedParameterLoadFusionAllGatherShape(
    const HloInstruction* inst) {
  CHECK(IsReplicatedParameterLoadFusion(inst));
  const HloComputation* comp = inst->fused_instructions_computation();
  const HloInstruction* parameter = comp->parameter_instruction(0);
  CHECK_EQ(parameter->user_count(), 1);
  const HloInstruction* user = parameter->users()[0];
  CHECK(IsPoplarInstruction(PoplarOp::AllGather)(user));
  return user->shape();
}

Status GetRemoteLoadStoreUsers(HloInstruction* inst, HloInstruction** load,
                               HloInstruction** store) {
  if (inst->user_count() != 2) {
    return FailedPrecondition(
        "Expected an offloaded instruction %s to have two users.",
        inst->name().c_str());
  }

  const int64_t load_user_idx =
      IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst->users()[0]) ? 0
                                                                           : 1;

  *load = inst->users()[load_user_idx];
  *store = inst->users()[1 - load_user_idx];

  if (!IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(*load)) {
    return FailedPrecondition("Could not find a load instruction.");
  }
  if (!IsPoplarInstruction(PoplarOp::RemoteParameterStore)(*store)) {
    return FailedPrecondition("Could not find a store instruction.");
  }
  return Status::OK();
}

int64_t PartitionedElementCountPerReplica(
    int64_t element_count, int64_t partition_replication_factor) {
  CHECK_GE(element_count, 0);
  CHECK_GT(partition_replication_factor, 0);
  return tensorflow::MathUtil::CeilOfRatio(element_count,
                                           partition_replication_factor);
}

std::size_t PartitionedByteCountPerReplica(
    std::size_t byte_count, PrimitiveType element_type,
    int64_t partition_replication_factor) {
  const std::size_t bytes_per_element =
      ShapeUtil::ByteSizeOfPrimitiveType(element_type);

  CHECK_NE(bytes_per_element, 0);
  CHECK_EQ(byte_count % bytes_per_element, 0);
  const std::size_t element_count = byte_count / bytes_per_element;

  const std::size_t partitioned_element_count =
      PartitionedElementCountPerReplica(element_count,
                                        partition_replication_factor);

  return partitioned_element_count * bytes_per_element;
}

StatusOr<int64_t> GetRemoteBufferEntryParameterNumber(
    const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto dfa,
                      HloDataflowAnalysis::Run(*inst->parent()->parent()));

  return GetRemoteBufferEntryParameterNumber(*dfa, inst);
}

StatusOr<int64_t> GetRemoteBufferEntryParameterNumber(
    const HloDataflowAnalysis& dfa, const HloInstruction* inst) {
  VLOG(2) << "GetRemoteBufferParameterNumber " << inst->ToString();

  const HloModule* module = inst->GetModule();
  const HloInstruction* source = inst;
  do {
    if (IsReplicatedParameterStore(source)) {
      VLOG(2) << "remote store instruction: " << source->ToString();
      source = source->users()[0]->operand(0);
    } else if (IsReplicatedParameterLoad(source)) {
      VLOG(2) << "remote load instruction: " << source->ToString();
      source = source->operand(0)->operand(0);
    } else if (IsRemoteBufferPassthrough(source) ||
               IsPoplarInstruction(RemoteParameterLoad, source)) {
      VLOG(2) << "remote parameter instruction: " << source->ToString();
      source = source->operand(0);
    }
    const auto value_set = dfa.GetFlattenedValueSet(source);
    if (value_set.values().size() != 1) {
      return FailedPrecondition(
          "Failed to determine remote buffer source. Is complex control "
          "flow used to pass remote buffers? Found %d sources for: %s.",
          value_set.values().size(), source->ToString());
    }
    const auto* next = value_set.values()[0]->instruction();
    if (next == source) {
      VLOG(2) << "stopped at instruction " << next->ToString();
      break;
    }
    source = next;
  } while (source->opcode() != HloOpcode::kParameter);

  if (source->parent() == module->entry_computation() &&
      source->opcode() == HloOpcode::kParameter) {
    return source->parameter_number();
  } else {
    return InternalErrorStrCat("Can't find an entry parameter index for ",
                               inst->name());
  }
}

}  // namespace poplarplugin
}  // namespace xla
