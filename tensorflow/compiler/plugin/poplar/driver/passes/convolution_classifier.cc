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

#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"

#include <set>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_ipu_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_conv.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

using ArgMap = std::multimap<HloInstruction*, HloInstruction*>;

/*
 * 1) find groups of convolutions which share the same inputs
 * 2) ops that share a common ARG0 input are analyzed to see which is likely
 *    to be the gradient w.r.t. arg1 (Filter Grad)
 * 3) any remaining convs which share the same ARG1 as one of the forward
 *    convs is a Input Grad
 * 4) any remaining ones are inference only
 */

namespace {
using ConvClassification = absl::flat_hash_map<HloInstruction*, MLType>;

bool IsTransposeLike(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kTranspose: {
      return true;
    }
    case HloOpcode::kReshape: {
      auto output_shape = inst->shape();
      auto input_shape = inst->operand(0)->shape();

      if (output_shape.rank() != 2 || input_shape.rank() != 2) {
        return false;
      }

      // If the input and output shape dims are a permutation of each
      // other then this is a transpose
      return ShapeUtil::Compatible(
          output_shape, ShapeUtil::PermuteDimensions({1, 0}, input_shape));
    }
    default:
      return false;
  }
}

bool IsAcceptableReshape(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReshape) {
    Shape output_shape = inst->shape();
    Shape input_shape = inst->operand(0)->shape();
    output_shape = ShapeUtil::DropDegenerateDimensions(output_shape);
    input_shape = ShapeUtil::DropDegenerateDimensions(input_shape);
    return ShapeUtil::Compatible(input_shape, output_shape);
  }
  return false;
}

bool IsOuterProductOfVectors(const HloInstruction* inst) {
  if (inst->operand(0)->shape().rank() != 1 ||
      inst->operand(1)->shape().rank() != 1) {
    return false;
  }
  const DotDimensionNumbers& dot_dims = inst->dot_dimension_numbers();
  auto lhs = dot_dims.lhs_contracting_dimensions();
  auto rhs = dot_dims.rhs_contracting_dimensions();
  return lhs.size() == 0 && rhs.size() == 0;
}

// Find the actual source of an input. Entry/Exit from tuples and kFusion
// instructions are traced though.
HloInstruction* FindOperand(HloInstruction* inst,
                            const std::unique_ptr<CallGraph>& call_graph) {
  HloInstruction* source = inst;
  std::vector<int64> tuple_stack;
  bool done = false;
  while (!done) {
    if (source->opcode() == HloOpcode::kParameter) {
      const auto* comp = source->parent();
      const auto& sites = call_graph->GetNode(comp).caller_callsites();
      if (sites.size() > 0) {
        int64 param_num = source->parameter_number();
        source = sites[0].instruction()->mutable_operand(param_num);
      } else {
        done = true;
      }
    } else if (source->opcode() == HloOpcode::kGetTupleElement) {
      // push tuple element index onto stack
      tuple_stack.push_back(source->tuple_index());
      source = source->mutable_operand(0);
    } else if (source->opcode() == HloOpcode::kTuple) {
      // pull tuple element index off stack and move to that operand
      int64 op_num = tuple_stack.back();
      tuple_stack.pop_back();
      source = source->mutable_operand(op_num);
    } else if (IsTransposeLike(source)) {
      // We look through transpose ops
      source = source->mutable_operand(0);
    } else if (IsAcceptableReshape(source)) {
      // We look through transpose ops
      source = source->mutable_operand(0);
    } else if (IsPoplarInstruction(PoplarOp::Fifo)(source)) {
      // We look through FIFO ops
      source = source->mutable_operand(0);
    } else if (IsPoplarInstruction(PoplarOp::InterIpuCopy)(source)) {
      // We look through inter-ipu copy ops.
      source = source->mutable_operand(0);
    } else if (IsPopOpsFusion(source, "zero_pad")) {
      // We look through zero pads.
      source = source->mutable_operand(0);
    } else {
      done = true;
    }
  }
  return source;
}

StatusOr<bool> IsArg1Gradient(const HloInstruction* inst,
                              const HloInstruction* arg0) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution: {
      // We assume that if the batch dimension (the non-reducing matrix
      // dimensions) isn't 0, then it is a grad
      TF_ASSIGN_OR_RETURN(auto d, GetConvolutionDims(inst));
      return d.input_batch_dimension() != 0;
    }
    case HloOpcode::kDot: {
      // We assume that if the arg0 input is transposed, then it is a grad
      // or if the matmul is an outer product of 2 vectors
      return IsTransposeLike(arg0) || IsOuterProductOfVectors(inst);
    }
    case HloOpcode::kFusion: {
      return IsPopOpsFusion(inst, "conv_scaled_inplace");
    }
    case HloOpcode::kCustomCall: {
      if (IsPoplarInstruction(PoplarOp::MultiConv)(inst)) {
        return Cast<HloMultiConvInstruction>(inst)->IsWeightUpdate();
      }
      return false;
    }
    default:
      return false;
  }
}

absl::optional<MLType> GetTypeFromInstruction(const HloInstruction* inst) {
  const auto attributes = inst->frontend_attributes();
  auto itr = attributes.map().find(FrontendAttributeId_Name(ML_TYPE));
  if (itr != attributes.map().end()) {
    MLType type;
    CHECK(MLType_Parse(itr->second, &type));
    return type;
  }
  return absl::nullopt;
}
}  // namespace

StatusOr<bool> ConvolutionClassifier::Run(HloModule* module) {
  ConvClassification classifications;

  auto* flattened = annotations_.flattened_module.get();
  if (flattened == nullptr) {
    return FailedPrecondition("Null flattened module found for %s",
                              module->name());
  }

  std::map<HloInstruction*, std::pair<int, int>> operands;

  for (auto comp : flattened->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      switch (inst->opcode()) {
        case HloOpcode::kConvolution: {
          classifications[inst] = MLType::INFERENCE_FWD;
          operands[inst] = std::make_pair(0, 1);
          break;
        }
        case HloOpcode::kDot: {
          classifications[inst] = MLType::INFERENCE_FWD;
          operands[inst] = std::make_pair(0, 1);
          break;
        }
        case HloOpcode::kFusion: {
          if (IsPopOpsConvolution(inst)) {
            classifications[inst] = MLType::INFERENCE_FWD;
            operands[inst] = std::make_pair(0, 1);
          } else if (IsPopOpsFusion(inst, "conv_scaled_inplace")) {
            classifications[inst] = MLType::INFERENCE_FWD;
            operands[inst] = std::make_pair(1, 2);
          }
          break;
        }
        case HloOpcode::kCustomCall: {
          if (IsPoplarInstruction(PoplarOp::MultiConv)(inst)) {
            classifications[inst] = MLType::INFERENCE_FWD;
            operands[inst] = std::make_pair(0, inst->operand_count() / 2);
          } else if (IsPopOpsConvolutionWithReverse(inst)) {
            classifications[inst] = MLType::INFERENCE_FWD;
            operands[inst] = std::make_pair(0, 1);
          }
          break;
        }
        default:
          break;
      }
    }
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(flattened);

  ArgMap arg0_fwd_map;
  ArgMap arg1_fwd_map;
  ArgMap arg1_rev_map;

  for (auto it : classifications) {
    auto& args = operands.at(it.first);
    HloInstruction* arg0 =
        FindOperand(it.first->mutable_operand(args.first), call_graph);
    arg0_fwd_map.insert(std::make_pair(arg0, it.first));
    HloInstruction* arg1 =
        FindOperand(it.first->mutable_operand(args.second), call_graph);
    arg1_fwd_map.insert(std::make_pair(arg1, it.first));
    arg1_rev_map.insert(std::make_pair(it.first, arg1));
  }

  std::set<HloInstruction*> arg0_set;
  for (auto it : arg0_fwd_map) {
    arg0_set.insert(it.first);
  }

  for (auto it : arg0_set) {
    std::set<HloInstruction*> fwd;
    std::set<HloInstruction*> wu;
    const auto& targets = arg0_fwd_map.equal_range(it);

    for (auto t = targets.first; t != targets.second; ++t) {
      auto* arg0 = t->second->operand(operands.at(t->second).first);
      TF_ASSIGN_OR_RETURN(bool arg1_gradient, IsArg1Gradient(t->second, arg0));
      if (arg1_gradient) {
        wu.insert(t->second);
      } else {
        fwd.insert(t->second);
      }
    }

    if (fwd.size() > 0 && wu.size() > 0) {
      for (HloInstruction* i : fwd) {
        classifications[i] = MLType::TRAINING_FWD;
      }
      for (HloInstruction* i : wu) {
        classifications[i] = MLType::TRAINING_WU;
      }
    }
  }

  for (auto& it : classifications) {
    if (it.second == MLType::INFERENCE_FWD) {
      auto weight = arg1_rev_map.find(it.first);
      auto targets = arg1_fwd_map.equal_range(weight->second);
      for (auto t = targets.first; t != targets.second; ++t) {
        if (classifications[t->second] == MLType::TRAINING_FWD) {
          it.second = MLType::TRAINING_BWD;
        }
      }
    }
  }

  for (auto& it : classifications) {
    MLType type = it.second;
    HloInstruction* mapped = annotations_.flattened_inst_map_bwd.at(it.first);
    // It's possible that the outter context has set the ML type directly.
    auto optional_type = GetTypeFromInstruction(mapped);
    if (optional_type && (*optional_type) != MLType::INFERENCE_FWD) {
      type = *optional_type;
    }

    TF_RETURN_IF_ERROR(SetInstructionMLType(it.first, type));
    TF_RETURN_IF_ERROR(SetInstructionMLType(mapped, type));

    if (VLOG_IS_ON(2)) {
      VLOG(2) << mapped->name() << " : " << MLType_Name(type);
    }
  }

  return true;
}

ConvolutionClassifier::ConvolutionClassifier(CompilerAnnotations& annotations)
    : annotations_(annotations) {}

}  // namespace poplarplugin
}  // namespace xla
