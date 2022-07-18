/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_instructions.h"

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

bool IsInstructionCacheable(HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      return true;

    case HloOpcode::kFusion:
      return IsPopOpsFusion(inst, "conv_with_reverse") ||
             IsPopOpsFusion(inst, "conv_scaled_inplace") ||
             IsPopOpsFusion(inst, "bias_apply");

    case HloOpcode::kCustomCall:
      return IsPoplarInstruction(PoplarOp::ConvWithReverse, inst) ||
             IsPoplarInstruction(PoplarOp::Conv_scaled_inplace, inst) ||
             IsPoplarInstruction(PoplarOp::MultiConv, inst) ||
             IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY, inst) ||

             IsPoplarInstruction(PoplarOp::GroupNormInference, inst) ||
             IsPoplarInstruction(PoplarOp::GroupNormTraining, inst) ||
             IsPoplarInstruction(PoplarOp::GroupNormGrad, inst) ||
             IsPoplarInstruction(PoplarOp::GroupNormStatistics, inst) ||
             IsPoplarInstruction(PoplarOp::BatchNormStatistics, inst) ||

             IsPoplarInstruction(PoplarOp::GRULayerFwd, inst) ||
             IsPoplarInstruction(PoplarOp::GRULayerBwd, inst) ||
             IsPoplarInstruction(PoplarOp::DynamicGRULayerFwd, inst) ||
             IsPoplarInstruction(PoplarOp::DynamicGRULayerBwd, inst) ||
             IsPoplarInstruction(PoplarOp::AUGRULayerFwd, inst) ||
             IsPoplarInstruction(PoplarOp::AUGRULayerBwd, inst) ||

             IsPoplarInstruction(PoplarOp::LstmLayerFwd, inst) ||
             IsPoplarInstruction(PoplarOp::LstmLayerBwd, inst) ||
             IsPoplarInstruction(PoplarOp::DynamicLstmLayerFwd, inst) ||
             IsPoplarInstruction(PoplarOp::DynamicLstmLayerBwd, inst);

    default:
      return false;
  }

  return false;
}

std::vector<HloInstruction*> GetCachableInstructions(HloModule* module) {
  std::vector<HloInstruction*> insts;

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp, "")) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsInstructionCacheable(inst)) {
        insts.push_back(inst);
      }
    }
  }
  return insts;
}

Status MoveInstructionIntoCacheableComputation(HloInstruction* inst) {
  auto* comp = inst->parent();
  auto* module = comp->parent();

  HloComputation::Builder comp_builder(inst->name() + "_instruction_cache");

  std::vector<HloInstruction*> params;
  for (int64_t i = 0; i < inst->operand_count(); i++) {
    const std::string param_name = absl::StrCat("arg_", i);
    const auto* inst_op = inst->operand(i);
    auto* param = comp_builder.AddInstruction(
        HloInstruction::CreateParameter(i, inst_op->shape(), param_name));
    if (inst_op->has_sharding()) {
      param->set_sharding(
          HloSharding::AssignDevice(inst_op->sharding().GetUniqueDevice()));
    }
    params.push_back(param);
  }

  auto* new_inst = comp_builder.AddInstruction(
      inst->CloneWithNewOperands(inst->shape(), params));

  auto* new_comp = module->AddEmbeddedComputation(comp_builder.Build(new_inst));

  auto* call = comp->AddInstruction(
      HloInstruction::CreateCall(inst->shape(), inst->operands(), new_comp));

  inst->SetupDerivedInstruction(call);

  TF_RETURN_IF_ERROR(call->CopyAllControlDepsFrom(inst));
  TF_RETURN_IF_ERROR(inst->DropAllControlDeps());

  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  poplar_backend_config.clear_ml_type();
  auto* call_config = poplar_backend_config.mutable_call_config();
  auto* function_config = call_config->mutable_function_config();
  function_config->set_keep_input_layouts(true);
  call_config->set_type(PoplarBackendConfig::CallConfig::Function);
  TF_RETURN_IF_ERROR(call->set_backend_config(poplar_backend_config));

  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(inst, call));

  return Status::OK();
}

};  // namespace

StatusOr<bool> OutlineInstructions::Run(HloModule* module) {
  bool changed = false;

  for (auto* inst : GetCachableInstructions(module)) {
    TF_RETURN_IF_ERROR(MoveInstructionIntoCacheableComputation(inst));
    changed = true;
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
