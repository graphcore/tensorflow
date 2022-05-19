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
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/ctc_preplanning.h"

#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ctc_loss.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <popnn/CTCInference.hpp>
#include <popnn/CTCLoss.hpp>

namespace xla {
namespace poplarplugin {

static bool IsCTCInferenceInst(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::CTCBeamSearchWithLogits)(inst) ||
         IsPoplarInstruction(PoplarOp::CTCBeamSearchWithLogProbs)(inst);
}

static bool HasCTCPlan(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::CTCLossWithLogits)(inst) ||
         IsPoplarInstruction(PoplarOp::CTCLossWithLogProbs)(inst) ||
         IsCTCInferenceInst(inst);
}

static StatusOr<popnn::ctc::Plan> GetPlanForLossInstruction(
    const HloInstruction* inst, CompilerResources& resources) {
  poplar::Graph& graph = GetGraph(resources, inst);
  const HloCTCLossInstructionBase* ctc_inst =
      Cast<HloCTCLossInstructionBase>(inst);
  TF_ASSIGN_OR_RETURN(poplar::Type in_dtype,
                      PoplarDataType(ctc_inst->in_dtype()));
  TF_ASSIGN_OR_RETURN(poplar::Type out_dtype,
                      PoplarDataType(ctc_inst->out_dtype()));

  const auto& data_shape = ctc_inst->operand(0)->shape();
  const auto& labels_shape = ctc_inst->operand(1)->shape();
  const int64_t max_time = data_shape.dimensions(0);
  const int64_t batch_size = data_shape.dimensions(1);
  const int64_t num_classes = data_shape.dimensions(2);
  const int64_t max_label_length = labels_shape.dimensions(1);
  return popnn::ctc::plan(graph, in_dtype, out_dtype, batch_size, max_time,
                          max_label_length, num_classes);
}

static StatusOr<popnn::ctc::Plan> GetPlanForInferenceInstruction(
    const HloInstruction* inst, CompilerResources& resources) {
  poplar::Graph& graph = GetGraph(resources, inst);
  const HloCTCInferenceInstructionBase* ctc_inst =
      Cast<HloCTCInferenceInstructionBase>(inst);
  TF_ASSIGN_OR_RETURN(poplar::Type in_dtype,
                      PoplarDataType(ctc_inst->in_dtype()));
  const auto& data_shape = ctc_inst->operand(0)->shape();
  const int64_t max_time = data_shape.dimensions(0);
  const int64_t batch_size = data_shape.dimensions(1);
  const int64_t num_classes = data_shape.dimensions(2);
  const int64_t beam_width = ctc_inst->beam_width();
  return popnn::ctc_infer::plan(graph, in_dtype, batch_size, max_time,
                                num_classes, beam_width);
}

/*
 * This visitor iterates over all ops in the graph, and creates a Plan for
 * every HloCTCLossInstruction and HloCTCInferenceInstruction found.
 */
StatusOr<bool> CTCPreplanning::Run(HloModule* module) {
  VLOG(2) << "Preplanning CTC operations.";

  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (HasCTCPlan(inst)) {
        auto plan = IsCTCInferenceInst(inst)
                        ? GetPlanForInferenceInstruction(inst, resources_)
                        : GetPlanForLossInstruction(inst, resources_);

        if (!plan.ok()) {
          return plan.status();
        }

        resources_.ctc_plans.insert(
            std::pair<const HloInstruction*, const popnn::ctc::Plan>(
                inst, std::move(plan.ValueOrDie())));
      }
    }
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
