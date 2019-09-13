/* Copyright 2017 Graphcore Ltd
 */

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
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_subcomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "absl/strings/str_cat.h"

#include <poplar/Tensor.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

SubComputationVisitor::SubComputationVisitor(
    CompilerResources& res, const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : DeferredAllocationVisitor(res),
      temp_inputs_(inputs),
      inputs_(inputs.size()),
      dependent_subcomputations_(dependent_subcomputations),
      used_tensors_(inputs.size()),
      allocated_tensors_(inputs.size()),
      has_allocation_target_(inputs.size()) {
  for (int64 i = 0; i < inputs.size(); i++) {
    inputs_[i].resize(inputs[i].size());
    used_tensors_[i].resize(inputs[i].size());
    allocated_tensors_[i].resize(inputs[i].size());
    has_allocation_target_[i].resize(inputs[i].size());
  }
}

InplaceSubComputationVisitor::InplaceSubComputationVisitor(
    CompilerResources& res, const ArgVectors& inputs,
    const TensorInputDescription& input_has_layout,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : SubComputationVisitor(res, inputs, dependent_subcomputations),
      input_has_layout_(input_has_layout) {}

InplaceSubComputationVisitor::InplaceSubComputationVisitor(
    CompilerResources& res, const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : SubComputationVisitor(res, inputs, dependent_subcomputations),
      input_has_layout_(inputs.size()) {
  for (int64 i = 0; i != inputs.size(); ++i) {
    input_has_layout_[i].resize(inputs[i].size(), true);
  }
}

bool SubComputationVisitor::InputIsUsedInThisSubComputation(
    HloParameterInstruction* inst, const std::vector<xla::Shape>& shapes,
    unsigned int index) {
  if (inst->parent()->root_instruction() == inst) {
    return true;
  }

  if (inst->user_count() == 0) {
    return false;
  }

  // Non-tuples are considered always used
  if (!inst->shape().IsTuple()) {
    return true;
  }

  // We ignore nested tuples
  if (shapes.size() != ShapeUtil::TupleElementCount(inst->shape())) {
    return true;
  }

  for (auto user : inst->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return true;
    }

    if (user->tuple_index() == index) {
      return true;
    }
  }
  return false;
}

bool SubComputationVisitor::InputIsUsedInDependentSubComputations(
    HloParameterInstruction* inst, unsigned int index) {
  const auto param_num = inst->parameter_number();
  for (const auto subcomputation : dependent_subcomputations_) {
    if (subcomputation->InputIsUsed(param_num, index)) {
      return true;
    }
  }
  return false;
}

Status SubComputationVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  HloParameterInstruction* param_inst =
      static_cast<HloParameterInstruction*>(inst);
  const auto param_num = param_inst->parameter_number();

  std::vector<xla::Shape> shapes = FlattenedXlaShape(param_inst->shape());
  auto& inputs = inputs_[param_num];
  auto& used = used_tensors_[param_num];
  auto& allocated = allocated_tensors_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  for (unsigned int i = 0; i < shapes.size(); i++) {
    auto& t = temp_inputs_[param_num][i];
    used[i] = InputIsUsedInThisSubComputation(param_inst, shapes, i);
    allocated[i] =
        InputIsUsedInDependentSubComputations(param_inst, i) || used[i];
    // If we have a deferred allocation then we can't add the output tensor yet
    // for the tensor mapping.
    bool add_output_tensor = true;
    if (!allocated[i]) {
      // For tensors which are not allocated we just forward them.
      inputs[i] = t;
    } else {
      // Handle the allocated tensor depending on whether this is inplace or
      // not.
      TF_ASSIGN_OR_RETURN(add_output_tensor,
                          HandleTensor(param_inst, shapes[i], i, t));
    }
    if (add_output_tensor) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, inputs[i]));
    }
  }

  return Status::OK();
}

StatusOr<bool> SubComputationVisitor::HandleTensor(
    HloParameterInstruction* inst, Shape& shape, const uint64 tuple_index,
    poplar::Tensor& tensor) {
  const auto param_num = inst->parameter_number();
  auto src = std::make_pair(inst, tuple_index);

  auto& inputs = inputs_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  // If we have a deferred allocation then we can't add the output tensor yet
  // for the tensor mapping.
  bool add_output_tensor = true;
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, tuple_index);

  // For used inputs we have the following cases:
  // 1. This is a deferred allocation.
  // 2. This input has an allocation target.
  // 3. This input contains constants.
  // 4. This input does not have a target.
  // For cases 2 and 3 we allocate a new tensor. For case 3 we clone the
  // layout of the input.
  if (CanDeferAllocation(inst, tuple_index)) {
    VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
            << tuple_index << ".";
    DeferAllocation(inst, tuple_index);
    add_output_tensor = false;
  } else if (HasTensorAllocationTarget(src, resources_) ||
             tensor.containsConstant()) {
    TF_ASSIGN_OR_RETURN(inputs[tuple_index],
                        AddTensor(graph, src, shape, resources_, tensor_map));
  } else {
    auto name = StrCat(GetDebugName(inst), "_in_", tuple_index);
    inputs[tuple_index] = graph.clone(tensor, name);
  }
  return add_output_tensor;
}

StatusOr<bool> InplaceSubComputationVisitor::HandleTensor(
    HloParameterInstruction* inst, Shape& shape, const uint64 tuple_index,
    poplar::Tensor& tensor) {
  const auto param_num = inst->parameter_number();
  auto src = std::make_pair(inst, tuple_index);

  auto& inputs = inputs_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  // If we have a deferred allocation then we can't add the output tensor yet
  // for the tensor mapping.
  bool add_output_tensor = true;
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, tuple_index);

  // For inplace inputs, we can still allocated the input (and then add a copy)
  // iff there is a (deferred) allocation target and the input doesn't have a
  // layout.
  const bool input_has_layout = input_has_layout_[param_num][tuple_index];
  inputs[tuple_index] = tensor;
  if (!input_has_layout) {
    if (CanDeferAllocation(inst, tuple_index)) {
      VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
              << tuple_index << ".";
      DeferAllocation(inst, tuple_index);
      add_output_tensor = false;
    } else if (HasTensorAllocationTarget(src, resources_)) {
      // If the input has an allocation target, then we use that layout
      // rather than the input layout.
      TF_ASSIGN_OR_RETURN(inputs[tuple_index],
                          AddTensor(graph, src, shape, resources_, tensor_map));
      allocated_targets[tuple_index] = true;
    }
  }
  return add_output_tensor;
}

poplar::program::Sequence InplaceSubComputationVisitor::GetPreambleCopies() {
  poplar::program::Sequence seq;
  for (int64 op_idx = 0; op_idx != temp_inputs_.size(); ++op_idx) {
    for (int64 tuple_idx = 0; tuple_idx != temp_inputs_[op_idx].size();
         ++tuple_idx) {
      if (InputHasAllocationTarget(op_idx, tuple_idx)) {
        VLOG(1) << "Adding a copy for input tensor (" << op_idx << ", "
                << tuple_idx << ").";
        seq.add(poplar::program::Copy(temp_inputs_[op_idx][tuple_idx],
                                      inputs_[op_idx][tuple_idx]));
      }
    }
  }
  return seq;
}

StatusOr<poplar::Tensor> SubComputationVisitor::PostProcessParameterAllocation(
    const HloInstruction* inst, int64 flat_tuple_index, const Shape&,
    poplar::Tensor tensor) {
  const auto param_num = inst->parameter_number();
  inputs_[param_num][flat_tuple_index] = tensor;
  has_allocation_target_[param_num][flat_tuple_index] = true;
  return tensor;
}

Status SubComputationVisitor::FinishVisit(HloInstruction* inst) {
  outputs_ = FindInstructionOutputs(tensor_map, inst);

  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);

  return Status::OK();
}

const ArgVectors& SubComputationVisitor::inputs() const { return inputs_; }

const OutVector& SubComputationVisitor::outputs() const { return outputs_; }

bool SubComputationVisitor::InputIsAllocated(int64 param,
                                             unsigned int index) const {
  return allocated_tensors_[param][index];
}

bool SubComputationVisitor::InputIsUsed(int64 param, unsigned int index) const {
  return used_tensors_[param][index];
}

bool SubComputationVisitor::InputHasAllocationTarget(int64 param,
                                                     unsigned int index) const {
  return has_allocation_target_[param][index];
}

StatusOr<TensorInputDescription>
InplaceSubComputationVisitor::GetInplaceSubcomputationLayoutInfo(
    CompilerResources& res, const HloInstruction* inst) {
  TensorInputDescription input_has_layout(inst->operand_count());
  // For each operand to the inplace subcomputation, check if the tensor coming
  // in has a layout. If the tensor does not have a layout then the inplace
  // subcomputation visitor might create one for this tensor.
  for (int64 i = 0; i < inst->operand_count(); i++) {
    auto* operand = inst->operand(i);
    std::vector<xla::Shape> shapes = FlattenedXlaShape(operand->shape());
    input_has_layout[i].reserve(shapes.size());
    for (int64 tuple_index = 0; tuple_index < shapes.size(); tuple_index++) {
      auto tensor_source = std::make_pair(operand, tuple_index);
      input_has_layout[i].push_back(
          res.annotations.tensors_with_layout.contains(tensor_source));
    }
  }
  return input_has_layout;
}

std::pair<int64, int64>
InplaceSubComputationVisitor::GetParameterNumberAndFlatIndex(
    int64 output_flat_index) {
  int64 paramter_number = 0;
  int64 flat_index = output_flat_index;
  while (flat_index > inputs_[paramter_number].size()) {
    flat_index -= inputs_[paramter_number].size();
    paramter_number++;
  }
  return {paramter_number, flat_index};
}

poplar::program::Sequence&
InplaceSubComputationVisitor::GetSequenceForAliasingCopy(
    int64, const HloComputation*) {
  // Be default just add the copies to the main sequence.
  return sequence;
}

StatusOr<ArgVector>
InplaceSubComputationVisitor::AddLoopInputOutputAliasingCopies(
    poplar::Graph& graph, const HloComputation* computation,
    const std::string& debug_name) {
  enum class AliasType {
    NO_ALIAS_NOT_USED,
    NO_ALIAS_USED,
    PARTIAL_ALIAS_OUTPUT_ONLY,
    PARTIAL_ALIAS,
    IDENTICAL_ALIAS,
  };
  // A loop output at shape-index `o` can:
  // 1. contain no aliases to any of the inputs and the input `o` is not used in
  // the computation (NO_ALIAS_NOT_USED).
  // 2. contain no aliases to any of the inputs and the input `o` is used in the
  // computation (NO_ALIAS_USED).
  // 3. contain an alias to one of the inputs and the input `o` is not used in
  // the computation (PARTIAL_ALIAS_OUTPUT_ONLY).
  // 4. contain an alias to one of the inputs and the input `o` is used in the
  // computation (PARTIAL_ALIAS).
  // 5. be the exact same tensor as input `o` (IDENTICAL_ALIAS).

  int64 num_tensors = outputs_.size();
  std::vector<AliasType> alias_type(num_tensors, AliasType::NO_ALIAS_USED);

  // Create a flat version of the loop inputs.
  ArgVector loop_inputs(num_tensors);
  auto input_itr = loop_inputs.begin();
  for (int64 input_idx = 0; input_idx != inputs_.size(); ++input_idx) {
    absl::c_copy(inputs_[input_idx], input_itr);
    input_itr = std::next(input_itr, inputs_[input_idx].size());
  }
  // Outputs are already flat.
  ArgVector loop_outputs = outputs_;

  // Find all the alias information index by output tensor.
  for (unsigned int o = 0; o < num_tensors; o++) {
    int64 param_number, param_index;
    std::tie(param_number, param_index) = GetParameterNumberAndFlatIndex(o);
    const bool input_used = InputIsAllocated(param_number, param_index);

    if (input_used) {
      if (loop_inputs[o] == loop_outputs[o]) {
        alias_type[o] = AliasType::IDENTICAL_ALIAS;
      }
      // Check if we need to add a temporary copy.
      for (unsigned int i = 0; i < num_tensors; i++) {
        int64 input_param_number, input_param_index;
        std::tie(input_param_number, input_param_index) =
            GetParameterNumberAndFlatIndex(i);

        if ((alias_type[o] != AliasType::IDENTICAL_ALIAS || i != o) &&
            InputIsAllocated(input_param_number, input_param_index)) {
          if (loop_outputs[o].intersectsWith(loop_inputs[i])) {
            alias_type[o] = AliasType::PARTIAL_ALIAS;
          }
        }
      }
    } else {
      // If the input is not used, check that the output at that index does not
      // alias any of the inputs which might have changed during
      // computation.
      alias_type[o] = AliasType::NO_ALIAS_NOT_USED;
      for (unsigned int i = 0; i < num_tensors; i++) {
        int64 input_param_number, input_param_index;
        std::tie(input_param_number, input_param_index) =
            GetParameterNumberAndFlatIndex(i);
        if (InputIsAllocated(input_param_number, input_param_index)) {
          if (loop_outputs[i].intersectsWith(loop_inputs[o])) {
            alias_type[o] = AliasType::PARTIAL_ALIAS_OUTPUT_ONLY;
          }
        }
      }
    }
  }

  // For partial aliasing types, we create temporary tensors from outputs in
  // order to remove any aliasing.
  ArgVector unaliased_loop_outputs(loop_outputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::PARTIAL_ALIAS: {
        VLOG(1) << "Adding a partial copy in " << debug_name
                << " for tuple index " << i;
        auto name = StrCat(debug_name, "_bodyout_temp_", i);
        unaliased_loop_outputs[i] = graph.clone(loop_outputs[i], name);
        poplar::program::Sequence& seq =
            GetSequenceForAliasingCopy(i, computation);
        seq.add(
            poplar::program::Copy(loop_outputs[i], unaliased_loop_outputs[i]));
        break;
      }
      default:
        break;
    }
  }

  ArgVector loop_state(loop_inputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS:
      case AliasType::NO_ALIAS_USED: {
        VLOG(1) << "Adding a output to input copy in " << debug_name
                << " for tuple index " << i;
        // Get the input ready for the next iteration.
        poplar::program::Sequence& seq =
            GetSequenceForAliasingCopy(i, computation);
        seq.add(
            poplar::program::Copy(unaliased_loop_outputs[i], loop_inputs[i]));
        break;
      }
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::NO_ALIAS_NOT_USED: {
        // The input is never used so we don't need a copy - just change the
        // while loop state as by default it contains the input tensors.
        loop_state[i] = unaliased_loop_outputs[i];
        break;
      }
      case AliasType::IDENTICAL_ALIAS:
      default:
        // nothing required
        break;
    }
  }
  return loop_state;
}

}  // namespace poplarplugin
}  // namespace xla
