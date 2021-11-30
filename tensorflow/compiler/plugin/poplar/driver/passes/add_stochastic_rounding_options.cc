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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

Shape CollectInputAndOutputShapes(const HloInstruction* inst) {
  std::vector<Shape> shapes;
  shapes.push_back(inst->shape());

  for (auto* operand : inst->operands()) {
    shapes.push_back(operand->shape());
  }

  return ShapeUtil::MakeTupleShape(shapes);
}

StatusOr<bool> NeedsSpecificSeedType(const HloInstruction* inst) {
  // Custom poplar instructions which don't need a specific seed since
  // they don't do any compute.
  const std::vector<PoplarOp> non_compute_poplar_ops = {
      PoplarOp::Assert,
      PoplarOp::ExecutionCounter,
      PoplarOp::CopyInto,
      PoplarOp::Fifo,
      PoplarOp::InterTilesetCopy,
      PoplarOp::InterIpuCopy,
      PoplarOp::StatefulNoop,
      PoplarOp::GradientAccumulatorCreate,
      PoplarOp::GradientAccumulatorSink};
  const bool skippable = absl::c_any_of(
      non_compute_poplar_ops,
      [&](PoplarOp op) { return IsPoplarInstruction(op, inst); });
  if (skippable) {
    return false;
  }

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

  // We assume that if an instruction has no f16 inputs or outputs
  // then it won't use SR.
  const auto all_inst_shapes = CollectInputAndOutputShapes(inst);
  const bool uses_stochastic_rounding =
      ShapeUtil::HasPrimitiveType(all_inst_shapes, F16);
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
    if (ReplicaIdenticalOnlySR()) {
      // In ReplicaIdenticalOnly mode we want SR for replica identical
      // instructions, in order to support this functionality alongside the
      // existing SR behaviour we always enable the SR backend option and
      // instead use StochasticRoundingMethod_None to disable it for non replica
      // identical instructions.
      stochastic_rounding = THREESTATE_ON;
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
  VLOG(3) << "Setting SR to " << ThreeState_Name(stochastic_rounding)
          << " for instruction '" << inst->name() << "'";

  if (stochastic_rounding != THREESTATE_OFF) {
    TF_ASSIGN_OR_RETURN(StochasticRoundingMethod stochastic_rounding_method,
                        GetStochasticRoundingMethod(inst));
    backend_config.set_stochastic_rounding_method(stochastic_rounding_method);
    VLOG(3) << "Setting SR method to "
            << StochasticRoundingMethod_Name(stochastic_rounding_method)
            << " for instruction '" << inst->name() << "'";
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
  // We need to disable SR for allReduce since it has data dependencies
  // that can cause the seed to diverge. Additionally when run with a non
  // replica identical seed it can create non replica identical results.
  if (inst->opcode() == HloOpcode::kAllReduce) {
    return StochasticRoundingMethod_None;
  }
  // Switching seeds is not free, we can save some time by not switching seeds
  // for instructions which won't be effected by it. Additionally it makes it
  // easier to manage the seed state since it's offten tuples/gtes and other
  // 'structural' instructions that get reordered during lowering, which
  // otherwise would require explicit handling to makesure the seed was valid
  // afterwards.
  TF_ASSIGN_OR_RETURN(bool needs_sepecific_seed, NeedsSpecificSeedType(inst));
  if (needs_sepecific_seed) {
    const auto default_method =
        DefaultStochasticRoundingMethod(default_stochastic_rounding_behaviour_);
    return IsInstructionReplicaIdentical(inst)
               ? StochasticRoundingMethod_IdenticalSeeds
               : default_method;
  }

  return StochasticRoundingMethod_Any;
}

bool AddStochasticRoundingOptions::ReplicaIdenticalOnlySR() const {
  return enable_experimental_prng_stability_ &&
         default_stochastic_rounding_behaviour_ ==
             StochasticRounding_ReplicaIdenticalOnly;
}

StochasticRoundingMethod DefaultStochasticRoundingMethod(
    StochasticRoundingBehaviour mode) {
  switch (mode) {
    case StochasticRounding_On:
      return StochasticRoundingMethod_DifferingSeeds;
    case StochasticRounding_ReplicaIdenticalOnly:
    case StochasticRounding_Off:
      return StochasticRoundingMethod_None;
    default:
      LOG(FATAL) << "Got unexpected stochastic rounding mode.";
  }

  return StochasticRoundingMethod_Undefined;
}

}  // namespace poplarplugin
}  // namespace xla
