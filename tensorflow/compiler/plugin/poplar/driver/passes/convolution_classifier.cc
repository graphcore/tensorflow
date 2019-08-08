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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

using ArgMap = std::multimap<HloInstruction*, HloInstruction*>;

/*
 * 1) find groups of convolutions which share the same inputs
 * 2) if any such group has >= 1 conv which has a graph parameter as an input,
 *    and >= 1 conv which does not have a graph parameter as an input, then
 *    mark the ones with a graph parameter as forwards, and the rest as
 *    backprop-filters
 * 3) any remaining convs which share the same weights as one of the forward
 *    convs is a backprop-input
 * 4) any remaining ones are inference only
 */

namespace {
using ConvClassification = absl::flat_hash_map<HloInstruction*, MLType>;

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
    } else if (source->opcode() == HloOpcode::kTranspose) {
      // We allow ourselves to look through transpose ops
      source = source->mutable_operand(0);
    } else {
      done = true;
    }
  }
  return source;
}

}  // namespace

StatusOr<bool> ConvolutionClassifier::Run(HloModule* module) {
  ConvClassification classifications;

  std::map<HloInstruction*, std::pair<int, int>> operands;

  const int64 num_res = GetResourceVariableParameterCount(module);
  std::set<HloInstruction*> variable_inputs(
      module->entry_computation()->parameter_instructions().end() - num_res,
      module->entry_computation()->parameter_instructions().end());

  for (HloComputation* comp : module->computations()) {
    if (!IsPopOpsFusion(comp)) {
      for (HloInstruction* inst : comp->instructions()) {
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
            std::string name = inst->fused_instructions_computation()->name();
            if (IsPopOpsConvolution(inst)) {
              classifications[inst] = MLType::INFERENCE_FWD;
              operands[inst] = std::make_pair(0, 1);
            } else if (IsPopOpsFusion(inst, "conv_scaled_inplace")) {
              classifications[inst] = MLType::INFERENCE_FWD;
              operands[inst] = std::make_pair(1, 2);
            }
            break;
          }
          default:
            break;
        }
      }
    }
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

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

  std::set<HloInstruction*> fwd;
  std::set<HloInstruction*> wu;

  for (auto it : arg0_set) {
    if (arg0_fwd_map.count(it) > 1) {
      const auto& targets = arg0_fwd_map.equal_range(it);

      for (auto t = targets.first; t != targets.second; ++t) {
        auto weight = arg1_rev_map.find(t->second);
        if (weight != arg1_rev_map.end()) {
          if (variable_inputs.count(weight->second) > 0) {
            fwd.insert(t->second);
          } else {
            wu.insert(t->second);
          }
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
    TF_RETURN_IF_ERROR(SetInstructionMLType(it.first, it.second));
  }

  if (VLOG_IS_ON(2)) {
    for (const auto& it : classifications) {
      VLOG(2) << it.first->name() << " : " << MLType_Name(it.second);
    }
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
