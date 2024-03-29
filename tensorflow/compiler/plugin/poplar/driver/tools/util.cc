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
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_ipu_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/within_replicas.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace poplarplugin {

bool operator==(const int64_t lhs, const Devices rhs) {
  return lhs == static_cast<int64_t>(rhs);
}

bool operator==(const Devices lhs, const int64_t rhs) { return (rhs == lhs); }
bool operator!=(const int64_t lhs, const Devices rhs) { return !(lhs == rhs); }
bool operator!=(const Devices lhs, const int64_t rhs) { return !(lhs == rhs); }

void StripAllInstructionLayouts(const HloModule* module) {
  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->instructions()) {
      if (!LayoutUtil::HasLayout(inst->shape()) ||
          LayoutUtil::IsDense(inst->shape().layout())) {
        LayoutUtil::SetToDefaultLayout(inst->mutable_shape());
      }
    }
  }
}

bool IsSupportedSharding(const HloSharding& sharding) {
  // We support unique single device sharding, representing an op/tensor which
  // is on an IPU, or single device sharding in a tuple/tree, repesenting a
  // tuple/tree of tensors on multiple devices.
  if (sharding.IsTuple()) {
    for (const auto& s : sharding.tuple_elements()) {
      if (!s.HasUniqueDevice()) {
        return false;
      }
    }
    return true;
  } else {
    return sharding.HasUniqueDevice();
  }
}

// Get the sharding for a particular input operand of an instruction
HloSharding GetShardingForOperand(const HloInstruction* inst, int operand) {
  auto get_sub_sharding = [=]() {
    auto s = inst->sharding();
    return s.GetSubSharding(inst->shape(), {operand});
  };

  switch (inst->opcode()) {
    case HloOpcode::kCall: {
      auto* comp = inst->to_apply();
      return comp->parameter_instruction(operand)->sharding();
    }
    case HloOpcode::kWhile: {
      auto* comp = inst->while_body();
      return comp->parameter_instruction(operand)->sharding();
    }
    case HloOpcode::kConditional: {
      if (operand == 0) {
        return inst->operand(0)->sharding();
      } else {
        auto* comp = inst->branch_computation(operand - 1);
        return comp->parameter_instruction(0)->sharding();
      }
    }
    case HloOpcode::kTuple: {
      return get_sub_sharding();
    }
    default: {
      if (IsPoplarInstruction(PoplarOp::Barrier)(inst) ||
          IsGCLWithinReplicaOp(inst)) {
        return get_sub_sharding();
      } else {
        return inst->sharding();
      }
    }
  }
}

const HloSharding& GetShardingOfOutputTensor(const HloInstruction* inst) {
  return inst->sharding();
}

std::vector<int64_t> GetShardingDeviceIdVector(const HloSharding& sharding) {
  std::vector<int64_t> ids;
  if (sharding.IsTuple()) {
    for (const auto& s : sharding.tuple_elements()) {
      ids.push_back(s.GetUniqueDevice());
    }
  } else {
    ids.push_back(sharding.GetUniqueDevice());
  }
  return ids;
}

bool HaveSharding(HloComputation* comp) {
  for (auto* inst : comp->instructions()) {
    if (inst->has_sharding()) {
      return true;
    }

    // Having a GCL withinReplica inst implicitly means there will be sharding,
    // since it generates sharding output and expects sharded input.
    if (IsGCLWithinReplicaOp(inst)) {
      return true;
    }
  }
  return false;
}

bool HaveSharding(HloModule* module) {
  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // If there is no sharding information, no need to continue
    if (HaveSharding(comp)) {
      return true;
    }
  }
  return false;
}

int64_t GetSingleShardingDeviceId(const HloInstruction* inst) {
  if (inst->has_sharding()) {
    return GetShardingDeviceIdVector(inst->sharding())[0];
  } else {
    return 0;
  }
}

bool IsAllowedTupleSharding(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kTuple:
    case HloOpcode::kParameter:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kGetTupleElement:
      return true;
    case HloOpcode::kCustomCall:
      return IsPoplarInstruction(PoplarOp::RemoteParameterLoad, inst) ||
             IsPoplarInstruction(PoplarOp::Barrier, inst) ||
             IsGCLWithinReplicaOp(inst);
    default:
      return false;
  }
}

void CopyShardingIfPresent(HloInstruction* const from,
                           HloInstruction* const to) {
  if (from->has_sharding()) {
    to->set_sharding(from->sharding());
  }
}

int64_t CountShapes(const Shape& shape) {
  int64_t n = 0;
  if (shape.IsTuple()) {
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
}

ShapeIndex RootShapeIndex() { return {}; }

int64_t InsertIntoTuple(const Shape& tuple, int64_t tuple_index,
                        int64_t original_index) {
  // Count up the base tensors inside all tuple element preceeding the
  // tuple_index one.
  int64_t tensor_count = 0;
  for (int64_t i = 0; i < tuple_index; i++) {
    tensor_count += CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  return tensor_count + original_index;
}

int64_t ExtractFromTuple(const Shape& tuple, int64_t tuple_index,
                         int64_t original_index) {
  int64_t index = original_index;
  for (int64_t i = 0; i < tuple_index; i++) {
    index -= CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  int64_t n = CountShapes(ShapeUtil::GetTupleElementShape(tuple, tuple_index));
  if (index < 0 || index >= n) {
    return -1;
  }
  return index;
}

template <typename F>
static void WalkShape(const Shape& shape, const F& f) {
  if (shape.IsTuple()) {
    for (const auto& s : shape.tuple_shapes()) {
      WalkShape(s, f);
    }
    return;
  }
  f(shape);
}

std::vector<Shape> FlattenedXlaShape(const Shape& shape) {
  std::vector<Shape> out;
  WalkShape(shape, [&](const Shape& s) { out.push_back(s); });
  return out;
}

int64_t GetByteSizeOfTotalShape(const Shape& shape) {
  int64_t size = 0;
  WalkShape(shape, [&](const Shape& s) { size += ShapeUtil::ByteSizeOf(s); });
  return size;
}

int64_t GetByteSizeOfTotalShapeSafe(const Shape& shape) {
  int64_t size = 0;
  WalkShape(shape, [&](const Shape& s) {
    if (s.IsOpaque()) {
      return;
    }
    // If we hit a token byte size of will return zero.
    // And alternative approach would be to just add the
    // size of operand 0 if it is a token. It's a bit of
    // a guess but for most cases a good one.
    size += ShapeUtil::ByteSizeOf(s);
  });
  return size;
}

template <typename NativeT>
StatusOr<NativeT> LiteralScalarToNativeType(const Literal& lit) {
  auto primitive_type = primitive_util::NativeToPrimitiveType<NativeT>();
  if (ShapeUtil::ElementsIn(lit.shape()) != 1) {
    return FailedPrecondition("Literal is not scalar");
  }

  TF_ASSIGN_OR_RETURN(Literal converted_lit, lit.Convert(primitive_type));

  return *static_cast<const NativeT*>(converted_lit.untyped_data());
}

template <typename NativeT>
StatusOr<std::vector<NativeT>> LiteralVectorToNativeType(const Literal& lit) {
  auto primitive_type = primitive_util::NativeToPrimitiveType<NativeT>();
  if (lit.shape().dimensions_size() != 1) {
    return FailedPrecondition("Literal rank != 1");
  }

  TF_ASSIGN_OR_RETURN(Literal converted_lit, lit.Convert(primitive_type));

  const NativeT* start =
      static_cast<const NativeT*>(converted_lit.untyped_data());
  return std::vector<NativeT>(start,
                              start + converted_lit.shape().dimensions(0));
}

template <typename NativeT>
StatusOr<std::vector<NativeT>> WideConstToNativeType(
    const HloInstruction* wide_const) {
  CHECK_EQ(wide_const->opcode(), HloOpcode::kBroadcast);
  if (wide_const->shape().dimensions_size() != 1) {
    return FailedPrecondition("Literal rank != 1");
  }
  const HloInstruction* constant = wide_const->operand(0);
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(NativeT val,
                      LiteralScalarToNativeType<NativeT>(constant->literal()));
  return std::vector<NativeT>(wide_const->shape().dimensions(0), val);
}

template StatusOr<uint8> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<uint16> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<uint32> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<uint64> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<int8> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<int16> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<int32> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<int64_t> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<half> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<bfloat16> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<float> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<double> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<complex64> LiteralScalarToNativeType(const Literal& lit);
template StatusOr<bool> LiteralScalarToNativeType(const Literal& lit);

#define INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(func) \
  template StatusOr<std::vector<uint8>> func;        \
  template StatusOr<std::vector<uint16>> func;       \
  template StatusOr<std::vector<uint32>> func;       \
  template StatusOr<std::vector<uint64>> func;       \
  template StatusOr<std::vector<int8>> func;         \
  template StatusOr<std::vector<int16>> func;        \
  template StatusOr<std::vector<int32>> func;        \
  template StatusOr<std::vector<int64_t>> func;      \
  template StatusOr<std::vector<half>> func;         \
  template StatusOr<std::vector<bfloat16>> func;     \
  template StatusOr<std::vector<float>> func;        \
  template StatusOr<std::vector<double>> func;       \
  template StatusOr<std::vector<complex64>> func;    \
  template StatusOr<std::vector<bool>> func;

INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(
    LiteralVectorToNativeType(const Literal& lit));
INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES(
    WideConstToNativeType(const HloInstruction* wide_const));

#undef INITIALISE_FOR_ALL_NATIVE_VECTOR_TYPES

namespace {

bool IsFusionComputationWithPrefix(const HloComputation* comp,
                                   const std::string& prefix) {
  return comp->IsFusionComputation() &&
         tensorflow::str_util::StartsWith(comp->name(), prefix);
}

}  //  namespace

bool IsInstructionInEntryComputation(const HloInstruction* inst) {
  return inst->parent() == inst->GetModule()->entry_computation();
}

bool IsPopOpsFusion(const HloComputation* comp, const std::string& postfix) {
  return IsFusionComputationWithPrefix(comp,
                                       absl::StrCat("_pop_op_" + postfix));
}

bool IsPopOpsFusion(const HloInstruction* inst, const std::string& postfix) {
  return inst->opcode() == HloOpcode::kFusion &&
         IsPopOpsFusion(inst->fused_instructions_computation(), postfix);
}

bool IsFusion(const HloInstruction* inst, const std::string& name) {
  return inst->opcode() == HloOpcode::kFusion &&
         IsFusionComputationWithPrefix(inst->fused_instructions_computation(),
                                       name);
}

namespace {
PoplarBackendConfig ParsePoplarBackendConfig(const HloInstruction* inst) {
  auto status_or = inst->backend_config<PoplarBackendConfig>();
  if (!status_or.ok()) {
    LOG(FATAL)
        << "Could not parse the PoplarBackendConfig for HloInstruction ''"
        << inst->name() << "' " << status_or.status().error_message();
  }
  return status_or.ValueOrDie();
}
bool CallConfigHasType(const HloInstruction* inst,
                       const PoplarBackendConfig::CallConfig::Type type) {
  if (inst->opcode() == HloOpcode::kCall) {
    PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
    return cfg.call_config().type() == type;
  }
  return false;
}
int64_t PipelineBatchSerializationIterations(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCall) {
    PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
    if (cfg.call_config().has_pipeline_config()) {
      return std::max<int64_t>(
          1,
          cfg.call_config().pipeline_config().batch_serialization_iterations());
    }
  }
  return 1;
}
}  // namespace

bool IsRepeatLoop(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::RepeatLoop);
}

int64_t GetRepeatLoopCount(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().repeat_config().repeat_count();
}

bool GetRepeatLoopAllowFinerAliasAnalysis(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().repeat_config().allow_finer_alias_analysis();
}

bool IsPipelineStage(const HloInstruction* inst) {
  return CallConfigHasType(inst,
                           PoplarBackendConfig::CallConfig::PipelineStage);
}

bool IsPipelineStageBackward(const HloInstruction* inst) {
  return CallConfigHasType(
      inst, PoplarBackendConfig::CallConfig::PipelineStageBackward);
}

bool IsPipelineStageRecomputation(const HloInstruction* inst) {
  return CallConfigHasType(
      inst, PoplarBackendConfig::CallConfig::PipelineStageRecomputation);
}

bool IsResourceUpdate(const HloInstruction* inst) {
  return CallConfigHasType(inst,
                           PoplarBackendConfig::CallConfig::ResourceUpdate);
}

bool IsFunction(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::Function);
}

bool IsCall(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::Call);
}

bool IsMultiConv(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::MultiConv);
}

bool IsPipelineOp(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::Pipeline);
}

bool IsBatchSerializedPipelineOp(const HloInstruction* inst) {
  return IsPipelineOp(inst) && (PipelineBatchSerializationIterations(inst) > 1);
}

int64_t GetPipelineRepeatCount(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().repeat_count();
}

int64_t GetAccumulationCountOperandIndex(const HloInstruction* inst) {
  CHECK(IsPipelineOp(inst));
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().gradient_accumulation_index();
}

const HloInstruction* GetGradientAccumulationCountInstruction(
    const HloInstruction* inst) {
  int64_t index = GetAccumulationCountOperandIndex(inst);
  CHECK_LT(index, inst->operands().size());
  return inst->operand(index);
}

bool IsGCLWithinReplicaOp(const HloInstruction* inst) {
  return DynCast<HloWithinReplicaInstruction>(inst) != nullptr;
}

template <typename NativeT>
absl::optional<NativeT> GetConstantValue(const HloInstruction* inst) {
  if (inst->IsConstant()) {
    auto value = LiteralScalarToNativeType<NativeT>(inst->literal());
    return absl::optional<NativeT>(value.ValueOrDie());
  }
  // If instruction is not a constant can try a little harder by trying to
  // evaluate the instruction

  // To keep this function taking a const HloInstruction (and a lot of
  // callers also taking a const HloInstruction) clone the instruction here
  // as the evaluator takes a non const instruction. Though will
  // try to create a const evaluator
  auto cloned_inst = inst->Clone();
  Literal result;
  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  if (!evaluator.TryEvaluate(cloned_inst.get(), &result)) {
    return absl::nullopt;
  }
  auto value = LiteralScalarToNativeType<NativeT>(result);
  return absl::optional<NativeT>(value.ValueOrDie());
}

template absl::optional<uint8> GetConstantValue(const HloInstruction* inst);
template absl::optional<uint16> GetConstantValue(const HloInstruction* inst);
template absl::optional<uint32> GetConstantValue(const HloInstruction* inst);
template absl::optional<uint64> GetConstantValue(const HloInstruction* inst);
template absl::optional<int8> GetConstantValue(const HloInstruction* inst);
template absl::optional<int16> GetConstantValue(const HloInstruction* inst);
template absl::optional<int32> GetConstantValue(const HloInstruction* inst);
template absl::optional<int64_t> GetConstantValue(const HloInstruction* inst);
template absl::optional<half> GetConstantValue(const HloInstruction* inst);
template absl::optional<bfloat16> GetConstantValue(const HloInstruction* inst);
template absl::optional<float> GetConstantValue(const HloInstruction* inst);
template absl::optional<double> GetConstantValue(const HloInstruction* inst);
template absl::optional<complex64> GetConstantValue(const HloInstruction* inst);
template absl::optional<bool> GetConstantValue(const HloInstruction* inst);

absl::optional<int64_t> GetAccumulationConstantsValue(
    const HloInstruction* inst) {
  return GetConstantValue<int64_t>(inst);
}

absl::optional<int64_t> GetGradientAccumulationCount(
    const HloInstruction* inst) {
  const auto gradient_accumulation_operand =
      GetGradientAccumulationCountInstruction(inst);
  auto result = GetAccumulationConstantsValue(gradient_accumulation_operand);
  return result;
}

int64_t GetPipelineBatchSerializationIterations(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().batch_serialization_iterations();
}

ThreeState GetPipelineOffloadActivations(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().offload_activations();
}

ThreeState GetPipelineOffloadGradientAccumulationBuffers(
    const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config()
      .pipeline_config()
      .offload_gradient_accumulation_buffers();
}

ThreeState GetPipelinePartitionVariables(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().partition_variables();
}

ThreeState GetPipelineOffloadVariables(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().offload_variables();
}

int64_t GetPipelineStageID(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_stage_config().stage_id();
}

const HloInstruction* GetResourceUpdateNumMiniBatchesInstruction(
    const HloInstruction* inst) {
  CHECK(IsResourceUpdate(inst));
  const HloComputation* comp = inst->to_apply();
  const auto instructions = comp->instructions();
  auto it = absl::c_find_if(instructions, [](const HloInstruction* candidate) {
    return IsPoplarInstruction(PoplarOp::GradientAccumulationCount)(candidate);
  });
  // There must be a gradient accumulation count instruction
  // inside the computation.
  CHECK(it != instructions.end());
  CHECK_EQ(it->operands().size(), 1);
  return it->operand(0);
}

HloInstruction* GetResourceUpdateNumMiniBatchesInstruction(
    HloInstruction* inst) {
  CHECK(IsResourceUpdate(inst));
  HloComputation* comp = inst->to_apply();
  auto instructions = comp->instructions();
  auto it = absl::c_find_if(instructions, [](const HloInstruction* candidate) {
    return IsPoplarInstruction(PoplarOp::GradientAccumulationCount)(candidate);
  });
  // There must be a gradient accumulation count instruction
  // inside the computation.
  CHECK(it != instructions.end());
  CHECK_EQ(it->operands().size(), 1);
  return it->mutable_operand(0);
}

absl::optional<int64_t> GetResourceUpdateBatchesToAccumulate(
    const HloInstruction* inst) {
  auto result = GetAccumulationConstantsValue(
      GetResourceUpdateNumMiniBatchesInstruction(inst));
  return result;
}

ThreeState GetResourceUpdateOffloadVariables(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().resource_update_config().offload_variables();
}

ThreeState GetResourceUpdatePartitionOffloadedVariables(
    const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config()
      .resource_update_config()
      .partition_offloaded_variables();
}

bool GetFunctionPartitionedElementwiseCluster(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().function_config().partitioned_elementwise_cluster();
}

bool GetFunctionKeepInputLayouts(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().function_config().keep_input_layouts();
}

bool GetFunctionUniqueSharding(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().function_config().unique_sharding();
}

int64_t GetFunctionNumberModifiedRemoteBufferInputs(
    const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config()
      .function_config()
      .num_modified_remote_buffer_inputs();
}

int64_t GetFunctionNumberUnmodifiedRemoteBufferInputs(
    const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config()
      .function_config()
      .num_unmodified_remote_buffer_inputs();
}

// TODO(T53775): Support colocated InterIpuCopy

HloInstruction* LookThroughCopies(HloInstruction* inst) {
  while (inst->opcode() == HloOpcode::kCopy ||
         IsPoplarInstruction(PoplarOp::InterIpuCopy, inst)) {
    inst = inst->mutable_operand(0);
  }
  return inst;
}

const HloInstruction* LookThroughCopies(const HloInstruction* inst) {
  while (inst->opcode() == HloOpcode::kCopy ||
         IsPoplarInstruction(PoplarOp::InterIpuCopy, inst)) {
    inst = inst->operand(0);
  }
  return inst;
}

bool UseSyntheticDataFor(SyntheticDataCategory category) {
  assert(category != SyntheticDataCategory::Unknown);

  const auto& flags = PoplarXlaFlags::Get();
  return flags.use_synthetic_data ||
         flags.synthetic_data_categories.count(category) == 1;
}

bool UseSyntheticDataInitializer() {
  return !PoplarXlaFlags::Get().synthetic_data_initializer.empty();
}

std::string GetDebugName(const HloInstruction* inst) {
  const std::string& tf_core_name = inst->metadata().op_name();
  return tf_core_name + "/" + inst->name();
}

void GetAllDeps(const HloInstruction* base,
                std::vector<HloInstruction*>& deps) {
  for (auto* inst : base->operands()) {
    if (inst->opcode() != HloOpcode::kAfterAll) {
      deps.push_back(inst);
    } else {
      GetAllDeps(inst, deps);
    }
  }
}

void GetAllDepNames(const HloInstruction* base,
                    std::vector<std::string>& names) {
  std::vector<HloInstruction*> deps;
  GetAllDeps(base, deps);
  for (const auto* d : deps) {
    names.push_back(d->name());
  }
}

namespace {
void SetInplaceBackendField(HloInstruction* inst, bool inplace) {
  auto backend_config =
      inst->backend_config<PoplarBackendConfig>().ValueOrDie();
  backend_config.set_is_inplace(inplace);
  TF_CHECK_OK(inst->set_backend_config(backend_config));
}
}  // namespace

void MakeUsedInplace(HloInstruction* inst) {
  SetInplaceBackendField(inst, true);
}

void MakeUsedNotInplace(HloInstruction* inst) {
  SetInplaceBackendField(inst, false);
}

bool IsLoweredInplace(const HloInstruction* inst) {
  auto backend_config =
      inst->backend_config<PoplarBackendConfig>().ValueOrDie();
  return backend_config.is_inplace();
}

void MarkInstructionAsReplicaIdentical(HloInstruction* inst,
                                       bool replica_identical) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  cfg.set_is_replica_identical(replica_identical);
  TF_CHECK_OK(inst->set_backend_config(cfg));
}

bool IsInstructionReplicaIdentical(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.is_replica_identical();
}

absl::flat_hash_set<const HloInstruction*> GetInplaceInstructions(
    const HloComputation* comp) {
  absl::flat_hash_set<const HloInstruction*> result;
  absl::c_copy_if(
      comp->instructions(), std::inserter(result, std::begin(result)),
      [](const HloInstruction* inst) { return IsLoweredInplace(inst); });
  return result;
}

absl::flat_hash_set<const HloInstruction*> GetInplaceInstructions(
    const HloModule* module) {
  absl::flat_hash_set<const HloInstruction*> result;
  for (auto comp : module->computations()) {
    auto comp_inplace = GetInplaceInstructions(comp);
    result.insert(comp_inplace.begin(), comp_inplace.end());
  }
  return result;
}

HloInstruction* ConvertInstruction(HloInstruction* inst,
                                   const PrimitiveType& new_type) {
  HloInstruction* new_inst;
  HloComputation* computation = inst->parent();
  if (inst->opcode() == HloOpcode::kConstant) {
    // For constants - replace it with the new constant.
    const auto shape = ShapeUtil::ChangeElementType(inst->shape(), new_type);
    auto literal_new_type = inst->literal().ConvertToShape(shape);
    new_inst = computation->AddInstruction(HloInstruction::CreateConstant(
        std::move(literal_new_type.ValueOrDie())));
  } else {
    // Otherwise clone and change the desired shape.
    new_inst = computation->AddInstruction(inst->Clone());
    new_inst->mutable_shape()->set_element_type(new_type);
  }

  new_inst->set_raw_backend_config_string(inst->raw_backend_config_string());

  new_inst->set_metadata(inst->metadata());
  if (inst->has_sharding()) {
    new_inst->set_sharding(inst->sharding());
  }
  return new_inst;
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(const HloInstruction& hlo,
                                 const absl::flat_hash_set<HloInstruction*>&
                                     instructions_in_subcomputation) {
  return absl::c_any_of(hlo.users(), [&](HloInstruction* user) {
    return !instructions_in_subcomputation.contains(user);
  });
}
}  // anonymous namespace

// This code is from HloModule::OutlineExpressionFromComputation.
// It creates fusion instead of call.
HloInstruction* OutlineExpressionFromComputationWithFusion(
    absl::Span<HloInstruction* const> instructions_to_outline,
    const string& outlined_computation_name, HloComputation* computation,
    const std::vector<HloInstruction*>& explicit_parameters, bool replace) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new
  // outlined function.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  absl::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64_t parameter_count = 0;

  for (HloInstruction* parameter : explicit_parameters) {
    arguments.push_back(parameter);
    InsertOrDie(&outlined_instructions, parameter,
                builder.AddInstruction(HloInstruction::CreateParameter(
                    parameter_count, parameter->shape(), "p")));
    ++parameter_count;
  }

  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64_t operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), "p"));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      absl::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a fusion to the nested computation.
  HloComputation* nested_computation =
      computation->parent()->AddEmbeddedComputation(
          builder.Build(FindOrDie(outlined_instructions, output)));

  HloInstruction* fusion =
      computation->AddInstruction(HloInstruction::CreateFusion(
          output->shape(), HloInstruction::FusionKind::kCustom, arguments,
          nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a fusion " << fusion->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  if (replace) {
    TF_CHECK_OK(output->ReplaceAllUsesWith(fusion));
    for (auto i = instructions_to_outline.rbegin();
         i != instructions_to_outline.rend(); ++i) {
      TF_CHECK_OK(computation->RemoveInstruction(*i));
    }
  }
  return fusion;
}

SliceInfo GetSliceInfo(const Shape& shape_to_slice, const Shape& slice_shape) {
  return GetSliceInfo(
      {shape_to_slice.dimensions().begin(), shape_to_slice.dimensions().end()},
      {slice_shape.dimensions().begin(), slice_shape.dimensions().end()});
}

SliceInfo GetSliceInfo(const std::vector<size_t>& shape_to_slice,
                       const std::vector<size_t>& slice_shape) {
  CHECK_EQ(shape_to_slice.size(), slice_shape.size());
  SliceInfo slice_info;
  // Get the dimensions we slice in and the slice sizes.
  for (uint64 dim = 0; dim != slice_shape.size(); ++dim) {
    size_t slice_size = slice_shape[dim];
    if (slice_size != shape_to_slice[dim]) {
      slice_info.sliced_dims.push_back(dim);
      slice_info.slice_sizes.push_back(slice_size);
    }
  }

  return slice_info;
}

Shape GetConcatenatedShape(std::vector<HloInstruction*> insts,
                           const int64_t dimension) {
  std::vector<const Shape*> inst_shapes;
  absl::c_transform(insts, std::back_inserter(inst_shapes),
                    [](HloInstruction* inst) { return &inst->shape(); });
  auto statusor = ShapeInference::InferConcatOpShape(inst_shapes, dimension);
  if (!statusor.ok()) {
    LOG(FATAL) << "Failed concatenating shapes together.";
  }
  return statusor.ValueOrDie();
}

StatusOr<HloInstruction*> GetUniqueGTEUser(HloInstruction* inst,
                                           int64_t tuple_index) {
  absl::flat_hash_set<HloInstruction*> gtes;
  for (HloInstruction* user : inst->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() == tuple_index) {
      gtes.insert(user);
    }
  }
  if (gtes.size() != 1) {
    return InternalErrorStrCat("Expected the ", inst->ToString(),
                               " to only have a single user, but it has ",
                               gtes.size(), " users.");
  }
  return *std::begin(gtes);
}

bool AllUsersUniqueGTEs(const HloInstruction* inst) {
  absl::flat_hash_map<int64_t, int64_t> gtes;
  for (const HloInstruction* user : inst->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      gtes[user->tuple_index()]++;
    } else {
      return false;
    }
  }

  // Check each output has a GTE.
  if (gtes.size() != ShapeUtil::TupleElementCount(inst->shape())) {
    return false;
  }

  // Check each GTE is unique.
  return absl::c_all_of(gtes,
                        [](const std::pair<int64_t, int64_t>& pair) -> bool {
                          return pair.second == 1;
                        });
}

size_t HloComputationHash::operator()(const HloComputation* comp) const {
  // A computation hash is the hash of all its parameters and its root
  // instruction. We are reluctant to hash all the instructions as the order
  // might not be the same but the instructions still represent the same
  // computation.
  size_t hash = 7;
  for (HloInstruction* param : comp->parameter_instructions()) {
    hash = tensorflow::Hash64Combine(hash, param->Hash());
  }
  return tensorflow::Hash64Combine(hash, comp->root_instruction()->Hash());
}

bool HloComputationEquals::operator()(const HloComputation* a,
                                      const HloComputation* b) const {
  return a->Equal(*b, false, true);
}

Status CreateDirIfMissing(const std::string& path) {
  CHECK(!path.empty());
  auto* env = tensorflow::Env::Default();

  // Two threads could race to observe the absence of the directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(path).ok()) {
    const auto status = env->RecursivelyCreateDir(path);
    if (!status.ok() && !env->IsDirectory(path).ok()) {
      return status;
    }
  }

  return Status::OK();
}

StatusOr<Tileset> GetTileset(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(const auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  return backend_config.tileset();
}

std::vector<HloInstruction*> FindUnreachableRoots(
    const HloComputation* computation) {
  std::vector<HloInstruction*> unreachable_roots;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->user_count() == 0 &&
        instruction->control_successors().empty() &&
        instruction != computation->root_instruction()) {
      unreachable_roots.push_back(instruction);
    }
  }

  return unreachable_roots;
}

StatusOr<HloInstruction*> CloneComputationSubtree(const HloInstruction* root,
                                                  HloComputation* to,
                                                  const string& suffix,
                                                  HloCloneContext* context) {
  std::queue<HloInstruction*> to_clone;
  HloInstruction* new_root = to->AddInstruction(root->Clone(suffix, context));
  to_clone.push(new_root);
  while (!to_clone.empty()) {
    HloInstruction* next = to_clone.front();
    to_clone.pop();
    for (int64_t op_idx = 0; op_idx < next->operand_count(); ++op_idx) {
      const HloInstruction* op = next->operand(op_idx);
      HloInstruction* clone = context ? context->FindInstruction(op) : nullptr;
      if (!clone) {
        clone = to->AddInstruction(op->Clone(suffix, context));
        to_clone.push(clone);
      }
      TF_RETURN_IF_ERROR(next->ReplaceOperandWith(op_idx, clone));
    }
  }
  return new_root;
}

namespace {
StatusOr<absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>>
GetDuplicateOperands(const HloInstruction* inst) {
  absl::flat_hash_map<const HloInstruction*, int64_t> first_occurrence;
  absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>> duplicate_operands;
  // Go through all the operands in order. First time we see it, add to
  // first_occurrence when we first saw it, next time we see it add it to the
  // duplicate operands.
  for (int64_t op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
    const HloInstruction* operand = inst->operand(op_idx);
    auto itr = first_occurrence.find(operand);
    if (itr == first_occurrence.end()) {
      first_occurrence[operand] = op_idx;
    } else {
      duplicate_operands[itr->second].insert(op_idx);
    }
  }
  return duplicate_operands;
}
}  // anonymous namespace

StatusOr<absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>>
GetDuplicateCallOutputs(const HloInstruction* call) {
  return GetDuplicateOperands(call->to_apply()->root_instruction());
}

StatusOr<absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>>
GetDuplicateCallInputs(const HloInstruction* call) {
  return GetDuplicateOperands(call);
}

StatusOr<absl::flat_hash_set<int64_t>> GetUnusedCallOutputIndices(
    const HloInstructionSet& calls) {
  absl::flat_hash_set<int64_t> unused_outputs;
  if (!calls.empty()) {
    // An arbitrary call.
    const HloInstruction* inst = *calls.begin();
    for (int64_t i = 0; i != ShapeUtil::TupleElementCount(inst->shape()); ++i) {
      unused_outputs.insert(i);
    }
  }

  // Search through all users to check which outputs they use.
  for (const HloInstruction* call : calls) {
    if (call->parent()->root_instruction() == call) {
      // If one of the calls is the root of a computation, any of its outputs
      // might be used in calling computations.
      unused_outputs.clear();
      break;
    } else {
      for (HloInstruction* user : call->users()) {
        CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
        unused_outputs.erase(user->tuple_index());
      }
    }
  }
  return unused_outputs;
}

StatusOr<absl::flat_hash_set<int64_t>> GetUnusedCallOutputIndices(
    HloInstruction* call) {
  HloInstructionSet call_set{call};
  return GetUnusedCallOutputIndices(call_set);
}

StatusOr<absl::flat_hash_set<int64_t>> GetUnusedParametersInCall(
    const HloInstruction* call) {
  const HloComputation* called_computation = call->to_apply();
  absl::flat_hash_set<int64_t> unused_params;
  for (int64_t param_number = 0;
       param_number != called_computation->num_parameters(); ++param_number) {
    const HloInstruction* parameter =
        called_computation->parameter_instruction(param_number);
    if (parameter->user_count() == 0 &&
        parameter != called_computation->root_instruction()) {
      unused_params.insert(param_number);
    }
  }
  return unused_params;
}

Status ReplaceDuplicateCallOutputs(
    HloInstruction* call,
    const absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>&
        duplicate_outputs) {
  // Get all the GTEs by tuple index.
  absl::flat_hash_map<int64_t, HloInstructionSet> gte_users;
  for (HloInstruction* user : call->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    gte_users[user->tuple_index()].insert(user);
  }

  // Change tuple indices all to the same value for gtes targeting duplicates.
  for (auto pair : duplicate_outputs) {
    int64_t output_idx = pair.first;
    const absl::flat_hash_set<int64_t>& duplicate_indices = pair.second;
    VLOG(3) << "Replacing duplicate output indices "
            << absl::StrJoin(duplicate_indices, ", ") << " with output index "
            << output_idx;
    for (int64_t duplicate_idx : duplicate_indices) {
      for (HloInstruction* gte : gte_users[duplicate_idx]) {
        gte->set_tuple_index(output_idx);
      }
    }
  }
  return Status::OK();
}

Status ReplaceDuplicateCallInputs(
    HloInstruction* call,
    const absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>&
        duplicate_inputs) {
  HloComputation* called_comp = call->to_apply();
  // Replace any duplicate inputs which will make parameters unused.
  for (auto pair : duplicate_inputs) {
    int64_t param_number = pair.first;
    const absl::flat_hash_set<int64_t>& duplicate_indices = pair.second;
    VLOG(3) << "Replacing duplicate parameter numbers "
            << absl::StrJoin(duplicate_indices, ", ")
            << " with parameter number " << param_number;
    HloInstruction* parameter =
        called_comp->parameter_instruction(param_number);
    for (int64_t duplicate_idx : duplicate_indices) {
      HloInstruction* parameter_to_replace =
          called_comp->parameter_instruction(duplicate_idx);
      TF_RETURN_IF_ERROR(parameter_to_replace->ReplaceAllUsesWith(parameter));
    }
  }
  return Status::OK();
}

Status RemoveOutputsFromCall(
    HloInstruction* call,
    const absl::flat_hash_set<int64_t>& outputs_to_remove) {
  // Nothing to remove.
  if (outputs_to_remove.empty()) {
    return Status::OK();
  }
  const int64_t num_outputs_old = ShapeUtil::TupleElementCount(call->shape());
  HloComputation* call_computation = call->to_apply();
  HloInstruction* root = call_computation->root_instruction();

  VLOG(3) << "Removing outputs " << absl::StrJoin(outputs_to_remove, ", ")
          << " from " << call->ToString();

  // Get all the GTEs.
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>
      tuple_index_to_gte;
  for (HloInstruction* user : call->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    tuple_index_to_gte[user->tuple_index()].insert(user);
  }

  // Get the new outputs, preserving the relative order.
  std::vector<HloInstruction*> new_outputs;
  new_outputs.reserve(num_outputs_old - outputs_to_remove.size());
  for (int64_t output_idx = 0; output_idx != num_outputs_old; ++output_idx) {
    if (outputs_to_remove.contains(output_idx)) {
      // Sanity check that this output has no users.
      CHECK(tuple_index_to_gte[output_idx].empty());
    } else {
      for (HloInstruction* gte : tuple_index_to_gte[output_idx]) {
        // Change the gte tuple index.
        gte->set_tuple_index(new_outputs.size());
      }
      new_outputs.push_back(root->mutable_operand(output_idx));
    }
  }

  HloInstruction* new_root;
  if (new_outputs.size() == 1 && new_outputs[0]->shape().IsTuple()) {
    // If there's only one output and it is already a tuple.
    new_root = new_outputs[0];
  } else {
    // Otherwise create a new tuple root.
    new_root = call_computation->AddInstruction(
        HloInstruction::CreateTuple(new_outputs));
  }

  // Change the computation shape
  std::vector<Shape>* mutable_call_tuple_shapes =
      call->mutable_shape()->mutable_tuple_shapes();
  *mutable_call_tuple_shapes = new_root->shape().tuple_shapes();
  if (call->has_sharding()) {
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, call));
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, new_root));
  }
  call_computation->set_root_instruction(new_root, true);

  if (root->user_count() == 0) {
    TF_RETURN_IF_ERROR(
        call_computation->RemoveInstructionAndUnusedOperands(root));
  }

  // In the case we didn't create a tuple, replace all users of GTE with the
  // call.
  if (new_outputs.size() == 1 && new_outputs[0]->shape().IsTuple()) {
    for (auto user : call->users()) {
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(call));
    }
  }

  return Status::OK();
}

Status SetTupleUniqueDeviceSharding(const HloInstruction* source,
                                    HloInstruction* dest) {
  auto optional_sharding = source->sharding().ExtractSingleSharding();
  if (!optional_sharding) {
    return FailedPrecondition("Could not extract single sharding.");
  }
  dest->set_sharding(
      HloSharding::SingleTuple(dest->shape(), *optional_sharding));
  return Status::OK();
}

// Replace call instruction with a new one with a new computation.
StatusOr<HloInstruction*> ReplaceCallWith(
    HloInstruction* call, std::unique_ptr<HloComputation> new_computation,
    const std::vector<HloInstruction*> new_operands,
    bool remove_unused_operands, HloCloneContext* context,
    const std::function<void(const HloCloneContext*)>&
        instructions_cloned_callback) {
  HloComputation* parent_computation = call->parent();
  HloComputation* call_computation = call->to_apply();
  HloModule* module = call->GetModule();

  HloComputation* new_call_computation =
      module->AddEmbeddedComputation(std::move(new_computation));

  HloInstruction* new_call =
      parent_computation->AddInstruction(HloInstruction::CreateCall(
          new_call_computation->root_instruction()->shape(), new_operands,
          new_call_computation));
  call->SetupDerivedInstruction(new_call);
  new_call->set_raw_backend_config_string(call->raw_backend_config_string());
  if (call->has_sharding()) {
    TF_RETURN_IF_ERROR(SetTupleUniqueDeviceSharding(call, new_call));
  }

  // Record the mapping of old->new in the clone context.
  if (context) {
    context->MapInstruction(call, new_call);
  }
  // Invoke instructions_cloned_callback if one has been specified.
  if (instructions_cloned_callback) {
    instructions_cloned_callback(context);
  }

  VLOG(3) << "Replacing " << call->ToString() << " and computation:";
  XLA_VLOG_LINES(3, call_computation->ToString());
  VLOG(3) << "With " << new_call->ToString() << " and computation:";
  XLA_VLOG_LINES(3, new_call_computation->ToString());

  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWithDifferentShape(new_call));
  if (remove_unused_operands) {
    TF_RETURN_IF_ERROR(
        parent_computation->RemoveInstructionAndUnusedOperands(call));
    // Manually remove CreateBuffer custom-calls which were unused operands
    // but were not deleted because they are marked as having side effects.
    for (auto* inst : FindUnreachableRoots(parent_computation)) {
      if (IsPoplarInstruction(PoplarOp::CreateBuffer)(inst)) {
        TF_RETURN_IF_ERROR(parent_computation->ForceRemoveInstruction(inst));
      }
    }
  } else {
    TF_RETURN_IF_ERROR(parent_computation->RemoveInstruction(call));
  }
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(call_computation));

  return new_call;
}

StatusOr<HloInstruction*> RemoveParametersFromCall(
    HloInstruction* call,
    const absl::flat_hash_set<int64_t>& parameters_to_remove,
    HloCloneContext* context,
    const std::function<void(const HloCloneContext*)>&
        instructions_cloned_callback) {
  // Nothing to remove.
  if (parameters_to_remove.empty()) {
    return call;
  }

  HloComputation* call_computation = call->to_apply();

  VLOG(3) << "Removing the following parameters from " << call->ToString();
  for (int64_t param_number : parameters_to_remove) {
    VLOG(3)
        << "\t* " << param_number << " "
        << call_computation->parameter_instruction(param_number)->ToString();
  }
  // A mapping from instructions in the old computation to the new one which is
  // currently being built.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_computation;
  auto builder = HloComputation::Builder(call_computation->name());

  // Lower/remove the parameters first.
  const int64_t old_num_parameters = call_computation->num_parameters();
  std::vector<HloInstruction*> new_call_operands;
  new_call_operands.reserve(old_num_parameters - parameters_to_remove.size());
  for (int64_t param_number = 0; param_number != old_num_parameters;
       ++param_number) {
    HloInstruction* old_parameter =
        call_computation->parameter_instruction(param_number);
    if (parameters_to_remove.contains(param_number)) {
      // Sanity check that the parameter we are removing has no users.
      CHECK_EQ(old_parameter->user_count(), 0);
    } else {
      // Otherwise lower it with the right index.
      HloInstruction* new_parameter =
          builder.AddInstruction(HloInstruction::CreateParameter(
              new_call_operands.size(), old_parameter->shape(),
              old_parameter->name()));
      // Record the mapping of old->new in the clone context.
      if (context) {
        context->MapInstruction(old_parameter, new_parameter);
      }
      old_parameter->SetupDerivedInstruction(new_parameter);
      old_to_new_computation[old_parameter] = new_parameter;
      new_call_operands.push_back(call->mutable_operand(param_number));
    }
  }

  // Lower all the other instructions.
  for (HloInstruction* old_inst :
       call_computation->MakeInstructionPostOrder()) {
    if (old_inst->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Get the operands for the instruction we are about to lower.
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [&old_to_new_computation](HloInstruction* old_operand) {
                        return old_to_new_computation.at(old_operand);
                      });
    // Clone new instruction.
    HloInstruction* new_inst =
        builder.AddInstruction(old_inst->CloneWithNewOperands(
            old_inst->shape(), new_operands, context));
    old_inst->SetupDerivedInstruction(new_inst);
    old_to_new_computation[old_inst] = new_inst;
  }

  // Build the new computation and the new call with new operands.
  HloInstruction* new_root =
      old_to_new_computation.at(call_computation->root_instruction());
  std::unique_ptr<HloComputation> new_computation = builder.Build(new_root);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_call,
      ReplaceCallWith(call, std::move(new_computation), new_call_operands,
                      /*remove_unused_operands=*/true, context,
                      instructions_cloned_callback));
  return new_call;
}

StatusOr<HloInstruction*> AddParametersToCall(
    HloInstruction* call, const std::vector<HloInstruction*>& parameters_to_add,
    HloCloneContext* context,
    const std::function<void(const HloCloneContext*)>&
        instructions_cloned_callback) {
  // Nothing to add.
  if (parameters_to_add.empty()) {
    return call;
  }

  HloComputation* call_computation = call->to_apply();

  int64_t param_number = call->operand_count();
  VLOG(3) << "Adding the following parameters to " << call->ToString();
  for (const HloInstruction* parameter : parameters_to_add) {
    VLOG(3) << "\t* " << param_number++ << " " << parameter->ToString();
  }

  std::vector<std::unique_ptr<HloInstruction>> new_parameters;
  std::vector<const HloInstruction*> new_parameter_ptrs;
  new_parameters.reserve(parameters_to_add.size());
  new_parameter_ptrs.reserve(new_parameters.size());

  // Create the new parameter instructions to be added to the computation.
  // These get cloned along with the original computation to create the new
  // computation.
  for (const HloInstruction* parameter : parameters_to_add) {
    string name = tensorflow::strings::StrCat("param_", param_number);
    new_parameters.emplace_back(HloInstruction::CreateParameter(
        call->operand_count() + new_parameters.size(), parameter->shape(),
        name));
    new_parameter_ptrs.push_back(new_parameters.back().get());
  }

  // Clone the original computation, adding in the new parameter instructions.
  std::unique_ptr<HloComputation> new_computation =
      call_computation->CloneWithReplacements({}, new_parameter_ptrs, context);

  // Create an updated operand list for the call (old operands + new operands).
  std::vector<HloInstruction*> new_call_operands(call->operands().begin(),
                                                 call->operands().end());
  new_call_operands.insert(new_call_operands.end(), parameters_to_add.begin(),
                           parameters_to_add.end());
  // Create a new call with the updated operand list, targeting the new
  // computation (which has parameters corresponding to the new operands).
  // This also deletes the original call, replacing usages with the new call.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_call,
      ReplaceCallWith(call, std::move(new_computation), new_call_operands,
                      /*remove_unused_operands=*/true, context,
                      instructions_cloned_callback));
  return new_call;
}

StatusOr<ShapeTree<CloneMethod>> GetCopyCloneMethod(
    const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  ShapeTree<CloneMethod> clone_method_tree(inst->shape());
  auto& clone_method_proto = poplar_backend_config.copy_config().clone_method();
  if (!clone_method_proto.empty()) {
    std::size_t tuple_idx = 0;
    for (auto& leaf : clone_method_tree.leaves()) {
      leaf.second =
          static_cast<CloneMethod>(clone_method_proto.at(tuple_idx++));
    }

    CHECK_EQ(ShapeUtil::GetLeafCount(inst->shape()), tuple_idx);
  }
  return clone_method_tree;
}

Status SetCopyCloneMethod(HloInstruction* inst,
                          const ShapeTree<CloneMethod>& clone_method_tree) {
  CHECK_EQ(ShapeUtil::GetLeafCount(inst->shape()),
           clone_method_tree.leaf_count());
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  auto* copy_config = backend_config.mutable_copy_config();
  copy_config->clear_clone_method();
  for (auto& leaf : clone_method_tree.leaves()) {
    copy_config->add_clone_method(leaf.second);
  }
  TF_RETURN_IF_ERROR(inst->set_backend_config(backend_config));

  return Status::OK();
}

Status SetPoplarUserDescriptions(HloInstruction* inst,
                                 const HloPoplarUseDescriptions& use_descs,
                                 bool allow_non_inplace) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  auto* cfg = backend_config.mutable_fusion_config();
  cfg->set_allow_non_inplace(allow_non_inplace);
  for (auto& use_desc : use_descs) {
    auto* proto = cfg->add_inplace_descriptions();
    *proto = use_desc.ToProto();
  }
  return inst->set_backend_config(backend_config);
}

StatusOr<HloInstruction*> TransposeToFront(HloInstruction* inst,
                                           absl::Span<const int64_t> dims) {
  std::vector<int64_t> permutations;
  permutations.reserve(inst->shape().rank());
  absl::c_copy(dims, std::back_inserter(permutations));
  for (int64_t dim = 0; dim != inst->shape().rank(); ++dim) {
    if (absl::c_find(dims, dim) == dims.end()) {
      permutations.push_back(dim);
    }
  }
  return MakeTransposeHlo(inst, permutations);
}

StatusOr<HloInstruction*> InverseTranspose(
    HloInstruction* inst, absl::Span<const int64_t> permutation) {
  std::vector<int64_t> inverse_permutation(permutation.size());
  for (int64_t i = 0; i != permutation.size(); ++i) {
    inverse_permutation[permutation[i]] = i;
  }
  return MakeTransposeHlo(inst, inverse_permutation);
}

StatusOr<HloInstruction*> ReshapeIfDifferent(HloInstruction* inst,
                                             const Shape& shape) {
  if (ShapeUtil::Compatible(shape, inst->shape())) {
    return inst;
  }
  return MakeReshapeHlo(shape, inst);
}

StatusOr<HloInstruction*> ReshapeIfDifferent(HloInstruction* inst,
                                             absl::Span<const int64_t> shape) {
  if (absl::c_equal(shape, inst->shape().dimensions())) {
    return inst;
  }
  return MakeReshapeHlo(shape, inst);
}

StatusOr<HloInstruction*> Flatten(HloInstruction* inst) {
  if (inst->shape().rank() == 1) {
    return inst;
  }
  std::vector<int64_t> shape(1, ShapeUtil::ElementsIn(inst->shape()));
  return MakeReshapeHlo(shape, inst);
}

}  // namespace poplarplugin
}  // namespace xla
