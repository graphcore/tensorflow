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
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ipu_inter_copy.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

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

int64 GetResourceVariableParameterCount(const HloModule* module) {
  /*
   * An XLA entry computation has a set of input parameters.  These map to a
   * combination of the inputs to the _XlaRun TF Op, and the resources which
   * are used by it.
   *
   * The `num_arguments` variable stores the total number of arguments in the
   * original _XlaRun operation. This does not include the number of resource
   * variables, or compile time constants.
   */

  const auto& inputs = module->entry_computation()->parameter_instructions();
  uint64 num_arguments = module->config().argument_count();
  return inputs.size() - num_arguments;
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
      auto s = inst->sharding();
      return s.GetSubSharding(inst->shape(), {operand});
    }
    default: { return inst->sharding(); }
  }
}

const HloSharding& GetShardingOfOutputTensor(const HloInstruction* inst) {
  return inst->sharding();
}

std::vector<int64> GetShardingDeviceIdVector(const HloSharding& sharding) {
  std::vector<int64> ids;
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

int64 GetSingleShardingDeviceId(const HloInstruction* inst) {
  if (inst->has_sharding()) {
    return GetShardingDeviceIdVector(inst->sharding())[0];
  } else {
    return 0;
  }
}

int64 CountShapes(const Shape& shape) {
  int64 n = 0;
  if (shape.IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      n += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return n;
  } else {
    return 1;
  }
}

int64 InsertIntoTuple(const Shape& tuple, int64 tuple_index,
                      int64 original_index) {
  // Count up the base tensors inside all tuple element preceeding the
  // tuple_index one.
  int64 tensor_count = 0;
  for (int64 i = 0; i < tuple_index; i++) {
    tensor_count += CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  return tensor_count + original_index;
}

std::vector<Shape> FlattenedXlaShape(const Shape& shape) {
  std::vector<Shape> out;
  if (shape.IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      std::vector<Shape> shapes =
          FlattenedXlaShape(ShapeUtil::GetTupleElementShape(shape, i));
      out.insert(out.end(), shapes.begin(), shapes.end());
    }
  } else {
    out.push_back(shape);
  }

  return out;
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
template StatusOr<int64> LiteralScalarToNativeType(const Literal& lit);
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
  template StatusOr<std::vector<int64>> func;        \
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

bool IsArithmeticExpressionFusion(const HloComputation* comp) {
  return IsFusionComputationWithPrefix(comp, "_arithmetic_expression");
}

bool IsArithmeticExpressionFusion(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kFusion &&
         IsArithmeticExpressionFusion(inst->fused_instructions_computation());
}

namespace {
PoplarBackendConfig ParsePoplarBackendConfig(const HloInstruction* inst) {
  auto status_or = inst->backend_config<PoplarBackendConfig>();
  if (!status_or.ok()) {
    LOG(FATAL)
        << "Could not parse the PoplarBackendConfig for HloInstruction ''"
        << inst->name() << "'";
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
}  // namespace

bool IsRepeatLoop(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::RepeatLoop);
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

bool IsPipelineResourceUpdate(const HloInstruction* inst) {
  return CallConfigHasType(
      inst, PoplarBackendConfig::CallConfig::PipelineResourceUpdate);
}

bool IsPipelineOp(const HloInstruction* inst) {
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::Pipeline);
}

bool CallCanBeInlined(const HloInstruction* inst) {
  // Only allow inlining for actual calls.
  return CallConfigHasType(inst, PoplarBackendConfig::CallConfig::Call);
}

int64 GetPipelineRepeatCount(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().pipeline_depth();
}

bool GetPipelineOffloadWUVariables(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_config().offload_wu_variables();
}

int64 GetPipelineStageID(const HloInstruction* inst) {
  PoplarBackendConfig cfg = ParsePoplarBackendConfig(inst);
  return cfg.call_config().pipeline_stage_config().stage_id();
}

const HloInstruction* GetOperandLookThroughInterIpuCopy(
    const HloInstruction* inst, const int64 operand_idx) {
  const HloInstruction* operand = inst->operand(operand_idx);
  return IsPoplarInstruction(PoplarOp::IpuInterCopy)(operand)
             ? operand->operand(0)
             : operand;
}

bool UseSyntheticData() { return PoplarXlaFlags::Get().use_synthetic_data; }

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
  inst->set_backend_config(backend_config);
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
    const string& outlined_computation_name, HloComputation* computation) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new
  // outlined function.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  absl::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64 parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64 operand_num = 0;
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

  TF_CHECK_OK(output->ReplaceAllUsesWith(fusion));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
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
                           const int64 dimension) {
  std::vector<const Shape*> inst_shapes;
  absl::c_transform(insts, std::back_inserter(inst_shapes),
                    [](HloInstruction* inst) { return &inst->shape(); });
  auto statusor = ShapeInference::InferConcatOpShape(inst_shapes, dimension);
  if (!statusor.ok()) {
    LOG(FATAL) << "Failed concatentating shapes together.";
  }
  return statusor.ValueOrDie();
}

}  // namespace poplarplugin
}  // namespace xla
