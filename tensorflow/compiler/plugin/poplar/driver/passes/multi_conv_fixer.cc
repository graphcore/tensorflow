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
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_conv_fixer.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/option_flag.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/human_readable_json.h"

namespace xla {
namespace poplarplugin {
namespace {
Status ReplaceConvsWithMultiConv(
    HloComputation* comp, std::vector<HloInstruction*>& convs,
    const std::vector<HloMultiConvInstruction::OptionFlag>& option_flags,
    CallInliner::InlinedInstructionMap& inline_map, bool is_wu) {
  const int64_t num_convs = convs.size();
  if (num_convs < 2) {
    return Status::OK();
  }
  std::vector<Shape> output_shapes(num_convs);
  std::vector<HloInstruction*> operands(num_convs * 2);
  std::vector<HloMultiConvInstruction::ConvolutionSpec> convolution_specs(
      num_convs);

  for (int64_t i = 0; i != num_convs; ++i) {
    HloInstruction* new_conv = inline_map[convs[i]];
    CHECK_EQ(new_conv->operand_count(), 2);

    output_shapes[i] = new_conv->shape();
    operands[i] = new_conv->mutable_operand(0);
    operands[num_convs + i] = new_conv->mutable_operand(1);

    HloMultiConvInstruction::ConvolutionSpec convolution_spec;
    if (new_conv->opcode() == HloOpcode::kConvolution) {
      convolution_spec.type = ConvType::Conv;
    } else if (IsPopOpsConvolutionWithReverse(new_conv)) {
      convolution_spec.type = ConvType::ConvWithReverse;
    } else {
      return InternalErrorStrCat("Could not classify the ",
                                 new_conv->ToString(),
                                 " convolution for the MultiConv operation.");
    }
    TF_ASSIGN_OR_RETURN(convolution_spec.window,
                        GetConvolutionWindow(new_conv));
    TF_ASSIGN_OR_RETURN(convolution_spec.dims, GetConvolutionDims(new_conv));
    TF_ASSIGN_OR_RETURN(convolution_spec.feature_group_count,
                        GetFeatureGroupCount(new_conv));
    TF_ASSIGN_OR_RETURN(convolution_spec.batch_group_count,
                        GetBatchGroupCount(new_conv));
    convolution_specs[i] = convolution_spec;
  }

  HloInstruction* multi_conv_inst = comp->AddInstruction(
      CreateMultiConv(ShapeUtil::MakeTupleShape(output_shapes), operands,
                      convolution_specs, option_flags, is_wu));

  // Replace all the uses with the outputs from the multi conv.
  for (int64_t i = 0; i != num_convs; ++i) {
    TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                        MakeGetTupleElementHlo(multi_conv_inst, i));

    HloInstruction* new_conv = inline_map[convs[i]];
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(new_conv, gte));
  }

  return Status::OK();
}
}  // namespace

Status MultiConvFixer::FixMultiConv(HloInstruction* multi_conv_op) {
  HloComputation* parent_comp = multi_conv_op->parent();
  HloComputation* comp = multi_conv_op->to_apply();

  auto reachability_map = HloReachabilityMap::Build(comp);

  std::vector<HloInstruction*> all_convolution_ops;
  // Split out any convolutions marked as wu.
  std::vector<HloInstruction*> wu_convolution_ops;
  std::vector<HloInstruction*> non_wu_convolution_ops;
  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (IsPopOpsConvolution(inst) ||
        inst->opcode() == HloOpcode::kConvolution) {
      // Make sure the convolutions are independent.
      if (absl::c_any_of(all_convolution_ops,
                         [&](const HloInstruction* other) -> bool {
                           return reachability_map->IsConnected(inst, other);
                         })) {
        return FailedPrecondition(
            "A MultiConvolution operation requires all convolutions to be "
            "independent of each other, however a data dependency was detected "
            "for %s.",
            multi_conv_op->ToString().c_str());
      }
      all_convolution_ops.push_back(inst);

      bool is_wu_conv = false;
      const auto attributes = inst->frontend_attributes();
      auto itr = attributes.map().find(FrontendAttributeId_Name(ML_TYPE));
      if (itr != attributes.map().end()) {
        is_wu_conv = itr->second == MLType_Name(MLType::TRAINING_WU);
      }

      if (is_wu_conv) {
        wu_convolution_ops.push_back(inst);
      } else {
        non_wu_convolution_ops.push_back(inst);
      }
    }
  }
  const auto attributes = multi_conv_op->frontend_attributes();
  auto itr = attributes.map().find(FrontendAttributeId_Name(OPTION_FLAGS));
  if (itr == attributes.map().end()) {
    return FailedPrecondition(
        "Could not find option flags in a multi conv operation.");
  }
  PoplarOptionFlags option_flags_proto;
  TF_RETURN_IF_ERROR(
      tensorflow::HumanReadableJsonToProto(itr->second, &option_flags_proto));

  std::vector<HloMultiConvInstruction::OptionFlag> option_flags(
      option_flags_proto.flags_size());

  for (int64_t i = 0; i != option_flags_proto.flags_size(); ++i) {
    auto& flag = option_flags_proto.flags(i);
    option_flags[i] = {flag.option(), flag.value()};
  }

  // Inline the multi conv.
  TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap map,
                      CallInliner::Inline(multi_conv_op));

  TF_RETURN_IF_ERROR(ReplaceConvsWithMultiConv(
      parent_comp, wu_convolution_ops, option_flags, map, /*is_wu*/ true));
  TF_RETURN_IF_ERROR(ReplaceConvsWithMultiConv(
      parent_comp, non_wu_convolution_ops, option_flags, map, /*is_wu*/ false));
  return Status::OK();
}

StatusOr<bool> MultiConvFixer::Run(HloModule* module) {
  VLOG(2) << "Before MultiConvFixer:";
  XLA_VLOG_LINES(2, module->ToString());

  // Find all the multi convs.
  std::vector<HloInstruction*> multi_convs;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsMultiConv(inst)) {
        multi_convs.push_back(inst);
      }
    }
  }

  for (HloInstruction* inst : multi_convs) {
    TF_RETURN_IF_ERROR(FixMultiConv(inst));
  }

  if (multi_convs.size()) {
    VLOG(2) << "After MultiConvFixer:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes.";
  }

  return multi_convs.size();
}

}  // namespace poplarplugin
}  // namespace xla
