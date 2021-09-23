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

#include "tensorflow/compiler/plugin/poplar/driver/passes/add_stochastic_rounding_options.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<bool> NeedsSpecificSeedType(const HloInstruction* inst) {
  // Instruction types that dont require a specific seed.
  switch (inst->opcode()) {
    // We don't need to worry about seeds for these instruction types since
    // they're mostly reading/restructuring data and so won't perform stochastic
    // rounding. More generally we only need to care about seeding for leaf
    // instructions, since higher level calls/control flow instructions will be
    // returning the results of those.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAnd:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kCompare:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kMap:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kPad:
    case HloOpcode::kPartitionId:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kSelect:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kWhile:
    case HloOpcode::kXor:
      return false;
    default:
      break;
  }

  const bool uses_stochastic_rounding =
      ShapeUtil::HasPrimitiveType(inst->shape(), F16);
  if (!uses_stochastic_rounding) {
    // If we're not using SR then it doesn't matter what
    // seed we use.
    return false;
  } else {
    // Instruction types that require a specific seed.
    switch (inst->opcode()) {
      case HloOpcode::kAbs:
      case HloOpcode::kAdd:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllToAll:
      case HloOpcode::kAtan2:
      case HloOpcode::kBatchNormGrad:
      case HloOpcode::kBatchNormInference:
      case HloOpcode::kBatchNormTraining:
      case HloOpcode::kBitcast:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kCeil:
      case HloOpcode::kCholesky:
      case HloOpcode::kClamp:
      case HloOpcode::kClz:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kComplex:
      case HloOpcode::kConcatenate:
      case HloOpcode::kConvert:
      case HloOpcode::kConvolution:
      case HloOpcode::kCos:
      case HloOpcode::kCustomCall:
      case HloOpcode::kDivide:
      case HloOpcode::kDomain:
      case HloOpcode::kDot:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFft:
      case HloOpcode::kFloor:
      case HloOpcode::kFusion:
      case HloOpcode::kGather:
      case HloOpcode::kImag:
      case HloOpcode::kIota:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kMultiply:
      case HloOpcode::kNegate:
      case HloOpcode::kPopulationCount:
      case HloOpcode::kPower:
      case HloOpcode::kReal:
      case HloOpcode::kReduce:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kRemainder:
      case HloOpcode::kReverse:
      case HloOpcode::kRng:
      case HloOpcode::kRngGetAndUpdateState:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRsqrt:
      case HloOpcode::kScatter:
      case HloOpcode::kSelectAndScatter:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kShiftRightLogical:
      case HloOpcode::kSign:
      case HloOpcode::kSin:
      case HloOpcode::kSort:
      case HloOpcode::kSqrt:
      case HloOpcode::kSubtract:
      case HloOpcode::kTanh:
      case HloOpcode::kTrace:
      case HloOpcode::kTriangularSolve:
        return true;
      default:
        break;
    }
  }

  return tensorflow::errors::FailedPrecondition(
      "Missing stochastic rounding seed usage support for instruction type '",
      inst->opcode(), "'");
}

}  // namespace

AddStochasticRoundingOptions::AddStochasticRoundingOptions(
    const StochasticRoundingBehaviour& default_stochastic_rounding_behaviour,
    bool enable_experimental_prng_stability)
    : default_stochastic_rounding_behaviour_(
          default_stochastic_rounding_behaviour),
      enable_experimental_prng_stability_(enable_experimental_prng_stability) {}

StatusOr<bool> AddStochasticRoundingOptions::Run(HloModule* module) {
  bool modified = false;

  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool added_option,
                          ConfigureStochasticRoundingOption(inst));
      modified |= added_option;

      TF_ASSIGN_OR_RETURN(added_option,
                          ConfigureDeterministicWorkersOption(inst));
      modified |= added_option;
    }
  }

  return modified;
}

StatusOr<bool> AddStochasticRoundingOptions::ConfigureStochasticRoundingOption(
    HloInstruction* inst) const {
  TF_ASSIGN_OR_RETURN(ThreeState stochastic_rounding,
                      ParseFrontendStochasticRoundingAttr(inst));

  // We only want to apply the default stochastic rounding option
  // if the instruction did not have one specified via its frontend
  // attributes. In this case stochastic rounding is either turned
  // on or off for all instructions, or just enabled for those that
  // that are replica identical.
  const bool use_default = stochastic_rounding == THREESTATE_UNDEFINED;
  if (use_default) {
    if (enable_experimental_prng_stability_ &&
        default_stochastic_rounding_behaviour_ ==
            StochasticRounding_ReplicaIdenticalOnly) {
      stochastic_rounding =
          IsInstructionReplicaIdentical(inst) ? THREESTATE_ON : THREESTATE_OFF;
    } else {
      stochastic_rounding =
          default_stochastic_rounding_behaviour_ == StochasticRounding_On
              ? THREESTATE_ON
              : THREESTATE_OFF;
    }
  }

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());

  backend_config.set_stochastic_rounding(stochastic_rounding);
  if (stochastic_rounding != StochasticRounding_Off) {
    TF_ASSIGN_OR_RETURN(StochasticRoundingMethod stochastic_rounding_method,
                        GetStochasticRoundingMethod(inst));
    backend_config.set_stochastic_rounding_method(stochastic_rounding_method);
  }

  TF_RETURN_IF_ERROR(inst->set_backend_config(backend_config));

  return true;
}

StatusOr<ThreeState>
AddStochasticRoundingOptions::ParseFrontendStochasticRoundingAttr(
    const HloInstruction* inst) const {
  ThreeState stochastic_rounding = THREESTATE_UNDEFINED;

  auto attributes = inst->frontend_attributes();
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());

  auto stochastic_rounding_attribute =
      attributes.map().find(FrontendAttributeId_Name(STOCHASTIC_ROUNDING));
  if (stochastic_rounding_attribute != attributes.map().end()) {
    if (!ThreeState_Parse(stochastic_rounding_attribute->second,
                          &stochastic_rounding)) {
      return FailedPrecondition(
          "Could not parse the stochastic rounding value");
    }
  }

  return stochastic_rounding;
}

StatusOr<StochasticRoundingMethod>
AddStochasticRoundingOptions::GetStochasticRoundingMethod(
    const HloInstruction* inst) const {
  // Switching seeds is not free, we can save some time by not switching seeds
  // for instructions which won't be effected by it.
  TF_ASSIGN_OR_RETURN(bool needs_sepecific_seed, NeedsSpecificSeedType(inst));
  if (needs_sepecific_seed) {
    return IsInstructionReplicaIdentical(inst)
               ? StochasticRoundingMethod_IdenticalSeeds
               : StochasticRoundingMethod_DifferingSeeds;
  }

  return StochasticRoundingMethod_Any;
}

StatusOr<bool>
AddStochasticRoundingOptions::ConfigureDeterministicWorkersOption(
    HloInstruction* inst) const {
  const ThreeState deterministic_workers = IsInstructionReplicaIdentical(inst)
                                               ? THREESTATE_ON
                                               : THREESTATE_UNDEFINED;

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  backend_config.set_deterministic_workers(deterministic_workers);
  TF_RETURN_IF_ERROR(inst->set_backend_config(backend_config));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
