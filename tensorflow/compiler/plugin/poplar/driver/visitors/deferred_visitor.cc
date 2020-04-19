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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

#include <poputil/Util.hpp>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "google/protobuf/util/message_differencer.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

Status DeferredAllocations::AddDeferredAllocation(
    TensorLocation location, DeferredAllocateFunction allocate_fn,
    DeferredPostProcessFunction post_process_fn) {
  switch (location.instruction->opcode()) {
    case HloOpcode::kInfeed:
    case HloOpcode::kParameter: {
      // Create a new set for this location.
      to_allocate_locations_[location].insert(location);
      // Move the allocation function.
      allocation_functions_[location] = std::move(allocate_fn);
      // Move the post process function.
      post_process_functions_[location] = std::move(post_process_fn);
      // Add it to the lookup table.
      location_lookup_table_[location] = std::move(location);
      return Status::OK();
    }
    default: {
      return FailedPrecondition(
          "Instruction %s cannot be a deferred allocation.",
          location.instruction->ToString());
    }
  }
}

Status DeferredAllocations::AddDeferredAllocationUser(
    TensorLocation user_input_location, TensorLocation user_output_location) {
  // First lookup which allocation set the user input location belongs to.
  auto itr = location_lookup_table_.find(user_input_location);
  if (itr == location_lookup_table_.end()) {
    return FailedPrecondition(
        "Could not find deferred allocation set for (%s,%d).",
        user_input_location.instruction->name(),
        user_input_location.flattened_output_tuple_index);
  }

  auto& input_location = itr->second;
  // Add the output location to the set.
  to_allocate_locations_.at(input_location).insert(user_output_location);
  location_lookup_table_[user_output_location] = input_location;
  return Status::OK();
}

Status DeferredAllocations::MakeDeferredAllocation(
    TensorLocation allocation_location,
    TensorLocation input_to_allocation_location) {
  TF_RETURN_IF_ERROR(AddDeferredAllocationUser(input_to_allocation_location,
                                               allocation_location));
  // Get the deferred allocation location from the
  // `input_to_allocation_location` as that must be in the look up table.
  TensorLocation input_location =
      location_lookup_table_.at(input_to_allocation_location);
  return MakeAllocation(input_location, allocation_location);
}

bool DeferredAllocations::IsDeferredAllocationLocation(
    CompilerResources& res, TensorLocation location) {
  return res.deferred_allocation_scopes.empty()
             ? false
             : res.deferred_allocation_scopes.top()
                   .IsDeferredAllocationLocation(location);
}

bool DeferredAllocations::IsDeferredAllocationLocation(
    TensorLocation location) {
  return location_lookup_table_.find(location) != location_lookup_table_.end();
}

void DeferredAllocations::AllocateIfExists(
    CompilerResources& res, const HloInstruction* inst,
    absl::optional<int64> opt_tensors_start,
    absl::optional<int64> opt_tensors_end) {
  if (!res.deferred_allocation_scopes.empty()) {
    auto s = res.deferred_allocation_scopes.top().AllocateIfExists(
        inst, DefaultToFirst(opt_tensors_start),
        DefaultToLast(opt_tensors_end));
    if (!s.ok()) {
      LOG(FATAL) << s;
    }
  }
}

Status DeferredAllocations::AllocateIfExists(const HloInstruction* inst,
                                             int64 tensors_start,
                                             int64 tensors_end) {
  TensorLocation start_location(inst, tensors_start);
  TensorLocation end_location(inst, tensors_end - 1);
  // Find any deferred allocations in the range, and allocate them.
  auto itr = location_lookup_table_.lower_bound(start_location);
  while (itr != location_lookup_table_.upper_bound(end_location)) {
    VLOG(1) << "Forced allocation for input location ("
            << itr->second.instruction->name() << ","
            << itr->second.flattened_output_tuple_index << ") from location ("
            << itr->first.instruction->name() << ","
            << itr->first.flattened_output_tuple_index << ").";
    TF_RETURN_IF_ERROR(MakeAllocation(itr->second, itr->first));
    itr = location_lookup_table_.lower_bound(start_location);
  }
  return Status::OK();
}

Status DeferredAllocations::AllocateRemainingLocations() {
  // Get all input locations left to allocate.
  std::vector<TensorLocation> input_locations;
  for (auto pair : to_allocate_locations_) {
    input_locations.push_back(pair.first);
  }

  for (auto input_location : input_locations) {
    TF_RETURN_IF_ERROR(MakeAllocation(input_location, input_location));
  }
  return Status::OK();
}

Status DeferredAllocations::MakeAllocation(TensorLocation input_location,
                                           TensorLocation allocation_location) {
  VLOG(1) << "Allocating for input location ("
          << input_location.instruction->name() << ","
          << input_location.flattened_output_tuple_index
          << ") from allocation location ("
          << allocation_location.instruction->name() << ","
          << allocation_location.flattened_output_tuple_index << ").";
  // Call the allocation function with the location of the allocation.
  TF_ASSIGN_OR_RETURN(auto tensor, allocation_functions_.at(input_location)(
                                       allocation_location));
  CHECK_EQ(allocation_functions_.erase(input_location), 1);

  return PostProcessAllocation(allocation_location, tensor);
}

Status DeferredAllocations::PostProcessAllocation(
    TensorLocation allocation_location, poplar::Tensor tensor) {
  TensorLocation input_location =
      location_lookup_table_.at(allocation_location);
  // Call the post process function.
  TF_ASSIGN_OR_RETURN(tensor,
                      post_process_functions_.at(input_location)(tensor));

  // Set the output tensor for all the allocations which were deferred.
  DeferredAllocationsLocationsSet& locations =
      to_allocate_locations_.at(input_location);
  locations.insert(allocation_location);
  for (const TensorLocation location : locations) {
    VLOG(1) << "Setting the allocation at (" << location.instruction->name()
            << "," << location.flattened_output_tuple_index << ").";
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map_, location.instruction,
                                       location.flattened_output_tuple_index,
                                       tensor));
    CHECK_EQ(location_lookup_table_.erase(location), 1);
  }

  // Mark this tensor as allocated.
  allocated_locations_.insert(input_location);
  CHECK_EQ(to_allocate_locations_.erase(input_location), 1);
  CHECK_EQ(post_process_functions_.erase(input_location), 1);

  return Status::OK();
}

DeferredVisitor::DeferredVisitor(
    CompilerResources& res, const DeferredArgVectors& callsite_inputs,
    const bool mark_all_input_tensors_as_used,
    const bool allocate_all_input_tensors,
    const std::vector<const DeferredVisitor*>& dependent_computations)
    : FullVisitor(res),
      callsite_inputs_(callsite_inputs),
      computation_inputs_(callsite_inputs.size()),
      dependent_computations_(dependent_computations),
      used_tensors_(callsite_inputs.size()),
      allocated_tensors_(callsite_inputs.size()),
      mark_all_input_tensors_as_used_(mark_all_input_tensors_as_used),
      allocate_all_input_tensors_(allocate_all_input_tensors) {
  for (size_t i = 0; i < callsite_inputs.size(); i++) {
    computation_inputs_[i].resize(callsite_inputs[i].size());
    used_tensors_[i].resize(callsite_inputs[i].size());
    allocated_tensors_[i].resize(callsite_inputs[i].size());
  }
  // When creating a new visitor explicitly push a new deferred scope
  resources_.deferred_allocation_scopes.push(DeferredAllocations{tensor_map});
}

const TensorVectors& DeferredVisitor::inputs() const {
  return computation_inputs_;
}

const TensorVector& DeferredVisitor::outputs() const { return outputs_; }

bool DeferredVisitor::InputIsAllocated(int64 param, unsigned int index) const {
  return allocated_tensors_[param][index];
}

bool DeferredVisitor::InputIsUsed(int64 param, unsigned int index) const {
  return used_tensors_[param][index];
}

StatusOr<poplar::program::Sequence*> DeferredVisitor::GetSequenceForInstruction(
    const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kInfeed: {
      return resources_.merge_infeed_io_copies ? &merged_infeed_sequence
                                               : &sequence;
    }
    default: { return &sequence; }
  }
}

Status DeferredVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  const auto param_num = inst->parameter_number();

  // Check whether this is a remote parameter - if so do not allocate the tensor
  // as the RemoteParameterLoad will allocate it and add the copy.
  if (IsRemoteParameter(inst, resources_)) {
    CHECK(IsInstructionInEntryComputation(inst));
    CHECK_EQ(inst->user_count(), 0);
    return Status::OK();
  }

  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
  auto& used = used_tensors_[param_num];
  auto& allocated = allocated_tensors_[param_num];

  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape shape = shapes[i];
    TensorLocation input_location(inst, i);

    // For some computations, like entry computation, every input is forced to
    // be marked as used.
    used[i] = InputIsUsedInThisComputation(input_location, shapes) ||
              mark_all_input_tensors_as_used_;
    allocated[i] =
        InputIsUsedInDependentComputations(input_location) || used[i];
    // Delegate the handling of parameter tensor.
    TF_RETURN_IF_ERROR(HandleParameterTensor(input_location, shape));
  }
  return Status::OK();
}

Status DeferredVisitor::HandleParameterTensor(TensorLocation input_location,
                                              const Shape shape) {
  const auto param_num = input_location.instruction->parameter_number();

  auto& callsite_inputs = callsite_inputs_[param_num];
  auto& inputs = computation_inputs_[param_num];
  auto& allocated = allocated_tensors_[param_num];

  // Whether this input has an allocation target.
  const bool has_allocation_target =
      HasTensorAllocationTarget(input_location, resources_);

  auto callsite_tensor =
      callsite_inputs_[param_num][input_location.flattened_output_tuple_index];

  // Function which is called when allocating this tensor.
  DeferredAllocateFunction allocate_fn =
      [this, shape, callsite_tensor](
          TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
    TF_ASSIGN_OR_RETURN(auto tensor, AllocateInput(allocation_location, shape,
                                                   callsite_tensor));
    return tensor;
  };

  // Function which is called to post processing the allocation of this input.
  DeferredPostProcessFunction post_process_fn =
      [this, input_location, shape,
       param_num](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
    // Call the post process for this location.
    TF_ASSIGN_OR_RETURN(tensor,
                        PostProcessInputTensor(tensor, input_location, shape));

    // Add the tensor to the computation inputs.
    computation_inputs_[param_num]
                       [input_location.flattened_output_tuple_index] = tensor;

    return tensor;
  };

  poplar::Tensor output;
  bool add_output_tensor = true;

  if (!allocated[input_location.flattened_output_tuple_index] &&
      callsite_tensor) {
    // If it is not allocated in this computation and there is a tensor layout
    // for this location, then just forward the tensor.

    // Do not call the post process for tensors which are not allocated.
    output = *callsite_tensor;
  } else {
    // The tensor is used and/or it doesn't have a layout.
    if (has_allocation_target) {
      // If a tensor can allocated now, then do it immediately.
      TF_ASSIGN_OR_RETURN(output, allocate_fn(input_location));
      TF_ASSIGN_OR_RETURN(output, post_process_fn(output));
    } else {
      // Otherwise defer the allocation.
      VLOG(1) << "Deferring allocation of "
              << input_location.instruction->name() << " sub tensor "
              << input_location.flattened_output_tuple_index << ".";
      TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
      TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
          input_location, std::move(allocate_fn), std::move(post_process_fn)));
      add_output_tensor = false;
    }
  }

  if (add_output_tensor) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, input_location.instruction,
                                input_location.flattened_output_tuple_index,
                                output));
  }

  return Status::OK();
}

Status DeferredVisitor::HandleInfeed(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // There is currently no way of ordering infeeds in the same computations -
  // this can result in unexpected results.
  if (has_infeed_) {
    return xla::FailedPrecondition(
        "Currently calling `get_next()` multiple times on the same "
        "IPUInfeedQueue in the same computation block is not supported.");
  }

  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());
  // Multiple infeed queues are not supported.
  if (absl::c_any_of(resources_.annotations.infeed_infos,
                     [&](const FeedInfo& info) {
                       return info.config.feed_id() != infeed_config.feed_id();
                     })) {
    return xla::FailedPrecondition(
        "Currently multiple IPUInfeedQueue in the same program are not "
        "supported.");
  }

  // Check that the replication factor matches.
  if (resources_.replication_factor != infeed_config.replication_factor()) {
    return xla::FailedPrecondition(
        "Current program has been created with replication_factor %d, however "
        "the IPUInfeedQueue has been configured with replication_factor %d. "
        "Either reduce the number of IPUs in your TensorFlow device, or set "
        "the `replication_factor` to % d when creating IPUInfeedQueue.",
        resources_.replication_factor, infeed_config.replication_factor(),
        resources_.replication_factor);
  }

  std::vector<Shape> shapes = FlattenedXlaShape(infeed->infeed_shape());
  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape shape = shapes[i];
    TensorLocation input_location(inst, i);

    // Function which is called when this input is allocated.
    DeferredAllocateFunction allocate_fn =
        [this, shape](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(auto tensor, AllocateInput(allocation_location, shape,
                                                     absl::nullopt));
      return tensor;
    };

    // Function which is called when post processing the allocation of this
    // input.
    DeferredPostProcessFunction post_process_fn =
        [this, input_location,
         shape](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
      // Call the post process for this location.
      TF_ASSIGN_OR_RETURN(
          tensor, PostProcessInputTensor(tensor, input_location, shape));

      return tensor;
    };

    if (HasTensorAllocationTarget(input_location, resources_)) {
      // Call the allocation function immediately.
      TF_ASSIGN_OR_RETURN(poplar::Tensor tensor, allocate_fn(input_location));
      TF_ASSIGN_OR_RETURN(tensor, post_process_fn(tensor));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, tensor));
    } else {
      // Otherwise defer the allocation.
      VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
              << i << ".";
      TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
      TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
          input_location, std::move(allocate_fn), std::move(post_process_fn)));
    }
  }
  has_infeed_ = true;

  FeedInfo info(infeed->name(), infeed_config, infeed->shape());
  resources_.annotations.infeed_infos.push_back(info);

  return Status::OK();
}

Status DeferredVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  // Go through all the shapes for inst, don't allocate any tensors which can be
  // deferred.
  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
  const uint64 start_flat_tuple_index =
      FindGetTupleElementTupleIndices(inst).first;
  for (size_t i = 0; i < shapes.size(); i++) {
    const uint64 flat_tuple_index = start_flat_tuple_index + i;
    // Get the correct graph for this output index.
    auto& graph = GetGraphWithOutputIndex(resources_, inst, i);

    // Set the input and output locations.
    TensorLocation output_location(inst, i);
    TensorLocation input_location(inst->operand(0), flat_tuple_index);

    // Whether this input has an allocation target.
    const bool has_allocation_target =
        HasTensorAllocationTarget(output_location, resources_);
    const bool input_location_is_deferred =
        deferred_allocation->IsDeferredAllocationLocation(input_location);

    if (has_allocation_target && input_location_is_deferred) {
      // There is an allocation for this location, therefore make it.
      TF_RETURN_IF_ERROR(deferred_allocation->MakeDeferredAllocation(
          output_location, input_location));
    } else {
      const bool is_lowered_inplace =
          IsLoweredInplace(output_location.instruction);

      // Try to defer the allocation, otherwise get the input tensor and forward
      // it. Note that getting a tensor means that it will be allocated.
      const bool can_defer = is_lowered_inplace && input_location_is_deferred;
      if (can_defer) {
        VLOG(1) << "Deferring use of " << inst->name() << " sub tensor " << i
                << ".";
        TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocationUser(
            input_location, output_location));
      } else {
        TF_ASSIGN_OR_RETURN(poplar::program::Sequence * seq,
                            GetSequenceForInstruction(inst));

        // Cannot defer the use of this tensor, hence get the input tensor and
        // set it as output.
        auto outputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, 0,
            {flat_tuple_index, flat_tuple_index + 1}, *seq, false);
        CHECK_EQ(outputs.size(), 1);
        poplar::Tensor output = outputs[0];
        // Duplicate the tensor if this is not an inplace lowering.
        if (!is_lowered_inplace) {
          output = poputil::duplicate(
              graph, output, *seq, absl::StrCat(GetDebugName(inst), "/", i),
              poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        }
        TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, output));
      }
    }
  }
  return Status::OK();
}

namespace {
std::vector<std::vector<bool>> GetAllowedDeferLocationsFromInputs(
    const HloInstruction* inst) {
  // Deferred allocations are not supported if the same tensor is used twice.
  // For example:
  // t = tuple(x, y, x, x, z)
  // Here x is used thrice, at indices 0, 2 and 3, so x cannot be deferred.
  std::vector<std::vector<bool>> deferred_locations(inst->operand_count());
  for (int64 operand_idx = 0; operand_idx != inst->operand_count();
       ++operand_idx) {
    const HloInstruction* operand = inst->operand(operand_idx);
    const bool allowed_to_defer = inst->OperandIndices(operand).size() == 1;
    auto shapes = FlattenedXlaShape(operand->shape());
    deferred_locations[operand_idx].resize(shapes.size(), allowed_to_defer);
  }
  return deferred_locations;
}
}  // namespace

Status DeferredVisitor::HandleTuple(HloInstruction* inst) {
  return HandleDeferredAllocationTuple(inst);
}

StatusOr<DeferredArgVectors>
DeferredVisitor::GetInputsForDeferredInplaceInstruction(
    const HloInstruction* inst, bool preserve_aliasing) {
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());

  auto allowed_deferred_allocation_locations =
      GetAllowedDeferLocationsFromInputs(inst);
  const bool is_lowered_inplace = IsLoweredInplace(inst);

  auto inplace_description = HloInstructionDescription(inst);
  if (!inplace_description.IsInplaceType()) {
    return InternalErrorStrCat("Trying to execute ", inst->name(),
                               " as an inplace operation, but it is not.");
  }
  auto inplace_indexes = inplace_description.GetInplaceOperandIndexes();
  absl::flat_hash_set<int64> inplace_indexes_set(inplace_indexes.begin(),
                                                 inplace_indexes.end());
  // Go through all the operands and get the input tensors for any input that
  // cannot be deferred.
  DeferredArgVectors inputs(inst->operand_count());
  for (int64 operand_idx = 0; operand_idx != inst->operand_count();
       ++operand_idx) {
    const HloInstruction* input_inst = inst->operand(operand_idx);

    // Dummy parameter outputs don't have a tensor.
    if (IsPoplarInstruction(PoplarOp::RemoteParameterDummyOutput)(input_inst)) {
      CHECK(IsInstructionInEntryComputation(input_inst));
      CHECK_EQ(inst->opcode(), HloOpcode::kTuple);
      continue;
    }

    auto shapes = FlattenedXlaShape(input_inst->shape());

    inputs[operand_idx].resize(shapes.size());

    // Go through all the tensors.
    for (size_t i = 0; i < shapes.size(); i++) {
      TensorLocation input_location(input_inst, i);

      bool is_tensor_lowered_inplace =
          is_lowered_inplace &&
          allowed_deferred_allocation_locations[operand_idx][i];

      const bool can_defer =
          is_tensor_lowered_inplace &&
          deferred_allocation->IsDeferredAllocationLocation(input_location);

      if (!can_defer) {
        // Cannot defer the allocation, hence get the output tensor for the
        // operand.
        auto outputs = FindInstructionOutputsInRange(tensor_map, resources_,
                                                     input_inst, {i, i + 1});
        CHECK_EQ(outputs.size(), 1);
        poplar::Tensor tensor = outputs[0];
        // Make sure to process any inplace operands correctly.
        if (inplace_indexes_set.contains(operand_idx)) {
          TF_ASSIGN_OR_RETURN(poplar::program::Sequence * seq,
                              GetSequenceForInstruction(inst));
          tensor = GetTensorForInplaceOp(
              outputs[0], resources_, inst, operand_idx, i, *seq,
              is_tensor_lowered_inplace, !preserve_aliasing);
        }
        inputs[operand_idx][i] = tensor;
      }
    }
  }
  return inputs;
}

Status DeferredVisitor::HandleDeferredAllocationTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  const bool preserve_aliasing = TupleOutputsNeedToPreserveAliasing(inst);
  TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredInplaceInstruction(
                                       inst, preserve_aliasing));

  CHECK_EQ(inputs.size(), inst->operand_count());

  uint64 output_tuple_index = 0;
  for (int64 operand_idx = 0; operand_idx != inst->operand_count();
       ++operand_idx) {
    const HloInstruction* input_inst = inst->operand(operand_idx);

    auto shapes = FlattenedXlaShape(input_inst->shape());

    // Dummy parameter outputs don't set a tensor output.
    if (IsPoplarInstruction(PoplarOp::RemoteParameterDummyOutput)(input_inst)) {
      CHECK(IsInstructionInEntryComputation(input_inst));
      output_tuple_index += shapes.size();
      continue;
    }

    CHECK_EQ(inputs[operand_idx].size(), shapes.size());

    for (size_t i = 0; i < shapes.size(); i++, output_tuple_index++) {
      // Set the input and output locations.
      TensorLocation output_location(inst, output_tuple_index);
      TensorLocation input_location(input_inst, i);
      if (inputs[operand_idx][i]) {
        // If a tensor exists then just forward it.
        poplar::Tensor output = *inputs[operand_idx][i];
        TF_RETURN_IF_ERROR(
            AddOutputTensor(tensor_map, inst, output_tuple_index, output));
      } else {
        // No tensor, therefore defer allocation.
        VLOG(1) << "Deferring use of " << inst->name() << " operand "
                << input_inst->name() << " sub tensor " << i << ".";
        TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocationUser(
            input_location, output_location));
      }
    }
  }
  return Status::OK();
}

Status DeferredVisitor::HandleCall(HloInstruction* inst) {
  return HandleDeferredAllocationCall(inst);
}

Status DeferredVisitor::HandleDeferredAllocationCall(HloInstruction* inst) {
  if (IsRepeatLoop(inst)) {
    // Get inputs preserving any deferred allocations.
    TF_ASSIGN_OR_RETURN(auto inputs,
                        GetInputsForDeferredInplaceInstruction(inst));
    TF_ASSIGN_OR_RETURN(
        auto seq, CreateRepeatOp(resources_, inst, inputs, GetOutputShape(inst),
                                 tensor_map));
    sequence.add(seq);
    return Status::OK();
  } else if (IsPipelineOp(inst)) {
    // Get inputs preserving any deferred allocations.
    TF_ASSIGN_OR_RETURN(auto inputs,
                        GetInputsForDeferredInplaceInstruction(inst));
    TF_ASSIGN_OR_RETURN(auto seq,
                        CreatePipelineOp(resources_, inst, inputs,
                                         GetOutputShape(inst), tensor_map));
    sequence.add(seq);
    return Status::OK();
  } else {
    return FullVisitor::HandleCall(inst);
  }
}

Status DeferredVisitor::HandleWhile(HloInstruction* inst) {
  return HandleDeferredAllocationWhile(inst);
}

Status DeferredVisitor::HandleDeferredAllocationWhile(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto inputs,
                      GetInputsForDeferredInplaceInstruction(inst));
  TF_ASSIGN_OR_RETURN(
      auto seq, CreateWhileOp(resources_, inst, inputs, GetOutputShape(inst),
                              tensor_map));
  sequence.add(seq);
  return Status::OK();
}

Status DeferredVisitor::FinishVisit(HloInstruction* inst) {
  // By default allocate all inputs into a callsite.
  if (allocate_all_input_tensors_) {
    // Force all deferred allocations to be executed.
    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AllocateRemainingLocations());
  }

  // Delegate.
  TF_RETURN_IF_ERROR(FinishDeferedAllocationVisit(inst));
  outputs_ = FindInstructionOutputs(tensor_map, resources_, inst);
  resources_.tensor_maps.AddTensorMapForComputation(inst->parent()->name(),
                                                    std::move(tensor_map));
  resources_.deferred_allocation_scopes.pop();
  return Status::OK();
}

Status DeferredVisitor::FinishDeferedAllocationVisit(HloInstruction* inst) {
  return Status::OK();
}

Status DeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst) {
  return PropagateDeferredAllocations(
      callsite_inst, std::vector<bool>(callsite_inst->operand_count(), true));
}

Status DeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst, std::vector<bool> add_clone) {
  // Note that this is called after finish visit, so tensors are being added to
  // the scope which called the visitor.
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());

  for (int64 operand_idx = 0; operand_idx != callsite_inst->operand_count();
       ++operand_idx) {
    const HloInstruction* input_inst = callsite_inst->operand(operand_idx);
    auto shapes = FlattenedXlaShape(input_inst->shape());
    for (size_t i = 0; i < shapes.size(); i++) {
      if (!callsite_inputs_[operand_idx][i]) {
        VLOG(1) << "Propagating allocated tensor at callsite "
                << callsite_inst->name() << " for input (" << input_inst->name()
                << ", " << i << ").";
        poplar::Tensor t = computation_inputs_[operand_idx][i];
        // Depending on the visitor inplace usage, the tensor might need to be
        // cloned so that it is not clobbered unexpectedly.
        if (add_clone[operand_idx]) {
          poplar::Graph& graph =
              GetGraphWithOutputIndex(resources_, input_inst, i);
          const std::string name =
              absl::StrCat(GetDebugName(input_inst), "/", i);
          t = graph.clone(
              t, name, poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        }
        TF_RETURN_IF_ERROR(
            deferred_allocation->PostProcessAllocation({input_inst, i}, t));
      }
    }
  }
  return Status::OK();
}

StatusOr<poplar::Tensor> DeferredVisitor::AllocateInput(
    TensorLocation allocation_location, const Shape& shape,
    absl::optional<poplar::Tensor> tensor_like) {
  poplar::Graph& graph =
      GetGraphWithOutputIndex(resources_, allocation_location.instruction,
                              allocation_location.flattened_output_tuple_index);

  VLOG(2) << "Allocating input tensor for "
          << allocation_location.instruction->name() << ":"
          << allocation_location.flattened_output_tuple_index << " shape "
          << shape.ToString() << " on shard "
          << GetShardForOutputIndex(
                 allocation_location.instruction,
                 allocation_location.flattened_output_tuple_index);

  poplar::Tensor out;
  // Allocate the tensor if:
  // * It has an allocation target so that it is allocated with the right
  //   layout, or
  // * It should have some layout, but that layout has constants in it, or
  // * It does not have some layout.
  // Otherwise just clone the desired layout.
  if (HasTensorAllocationTarget(allocation_location, resources_) ||
      (tensor_like && tensor_like->containsConstant()) || (!tensor_like)) {
    // Do the allocation.
    TF_ASSIGN_OR_RETURN(out, AddTensor(graph, allocation_location, shape,
                                       resources_, tensor_map));
  } else {
    auto name = absl::StrCat(GetDebugName(allocation_location.instruction), "_",
                             allocation_location.flattened_output_tuple_index);
    out = graph.clone(*tensor_like, name);
  }

  return out;
}

StatusOr<poplar::Tensor> DeferredVisitor::PostProcessInputTensor(
    poplar::Tensor tensor, TensorLocation input_location, const Shape& shape) {
  // Get the sequence to which any post processing operations should be added to
  // (for example stream copies).
  TF_ASSIGN_OR_RETURN(auto seq,
                      GetSequenceForInstruction(input_location.instruction));
  // Each visitor might need to post process the allocation, for example change
  // add copies to a sequence. Use the input location for that information.
  switch (input_location.instruction->opcode()) {
    case HloOpcode::kInfeed: {
      TF_ASSIGN_OR_RETURN(tensor, PostProcessInfeedAllocation(
                                      input_location, shape, *seq, tensor));
      break;
    }
    case HloOpcode::kParameter: {
      TF_ASSIGN_OR_RETURN(tensor, PostProcessParameterAllocation(
                                      input_location, shape, *seq, tensor));
      break;
    }
    default: {
      return xla::FailedPrecondition(
          "Unsupported input allocation for opcode %s.",
          HloOpcodeString(input_location.instruction->opcode()).c_str());
    }
  }
  return tensor;
}

StatusOr<poplar::Tensor> DeferredVisitor::PostProcessInfeedAllocation(
    TensorLocation location, const Shape& shape,
    poplar::program::Sequence& sequence, poplar::Tensor tensor) {
  TF_ASSIGN_OR_RETURN(
      auto prog,
      CreateInfeed(resources_, location.instruction,
                   location.flattened_output_tuple_index, shape, tensor));
  sequence.add(prog);
  return tensor;
}

StatusOr<DeferredAllocations*> DeferredVisitor::GetDeferredAllocations() {
  if (resources_.deferred_allocation_scopes.empty()) {
    return FailedPrecondition("Cannot get the DeferredAllocations.");
  }
  return &resources_.deferred_allocation_scopes.top();
}

bool DeferredVisitor::InputIsUsedInThisComputation(
    TensorLocation location, const std::vector<xla::Shape>& shapes) {
  if (location.instruction->parent()->root_instruction() ==
      location.instruction) {
    return true;
  }

  if (location.instruction->user_count() == 0) {
    return false;
  }

  // Non-tuples are considered always used
  if (!location.instruction->shape().IsTuple()) {
    return true;
  }

  // Ignore nested tuples
  if (static_cast<int64>(shapes.size()) !=
      ShapeUtil::TupleElementCount(location.instruction->shape())) {
    return true;
  }

  for (auto user : location.instruction->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return true;
    }

    if (user->tuple_index() == location.flattened_output_tuple_index) {
      return true;
    }
  }
  return false;
}

bool DeferredVisitor::InputIsUsedInDependentComputations(
    TensorLocation location) {
  const auto param_num = location.instruction->parameter_number();
  for (const auto computation : dependent_computations_) {
    if (computation->InputIsUsed(param_num,
                                 location.flattened_output_tuple_index)) {
      return true;
    }
  }
  return false;
}

InplaceDeferredVisitor::InplaceDeferredVisitor(
    CompilerResources& res, const DeferredArgVectors& inputs,
    const std::vector<const DeferredVisitor*>& dependent_subcomputations,
    bool reallocate_inputs)
    : DeferredVisitor(res, inputs, false, true, dependent_subcomputations),
      reallocate_inputs_(reallocate_inputs) {}

Status InplaceDeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst) {
  return DeferredVisitor::PropagateDeferredAllocations(
      callsite_inst, std::vector<bool>(callsite_inst->operand_count(), false));
}

StatusOr<poplar::program::Sequence>
InplaceDeferredVisitor::GetPreambleCopies() {
  poplar::program::Sequence seq;
  for (uint64 i = 0; i != callsite_inputs_.size(); ++i) {
    for (uint64 j = 0; j != callsite_inputs_[i].size(); ++j) {
      if (callsite_inputs_[i][j] &&
          *callsite_inputs_[i][j] != computation_inputs_[i][j]) {
        // For inplace vistors, they should only differ if we allow relocation.
        if (!reallocate_inputs_) {
          return FailedPrecondition("Input should have not been reallocated.");
        }
        VLOG(1) << "Adding a copy for input (" << i << ", " << j << ").";
        seq.add(poplar::program::Copy(*callsite_inputs_[i][j],
                                      computation_inputs_[i][j]));
      }
    }
  }
  return seq;
}

Status InplaceDeferredVisitor::HandleParameterTensor(
    TensorLocation input_location, const Shape shape) {
  const auto param_num = input_location.instruction->parameter_number();

  // Whether this input has an allocation target.
  const bool has_allocation_target =
      HasTensorAllocationTarget(input_location, resources_);

  auto callsite_tensor =
      callsite_inputs_[param_num][input_location.flattened_output_tuple_index];

  // Function which is called when tensor is allocated for this input.
  DeferredAllocateFunction allocate_fn =
      [this, shape, callsite_tensor](
          TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
    TF_ASSIGN_OR_RETURN(auto tensor, AllocateInput(allocation_location, shape,
                                                   callsite_tensor));
    return tensor;
  };

  // Function which is called when post processing the allocation of this input.
  DeferredPostProcessFunction post_process_fn =
      [this, input_location, shape,
       param_num](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
    // Call the post process for this location.
    TF_ASSIGN_OR_RETURN(tensor,
                        PostProcessInputTensor(tensor, input_location, shape));
    // Add the tensor to the computation inputs.
    computation_inputs_[param_num]
                       [input_location.flattened_output_tuple_index] = tensor;

    return tensor;
  };

  poplar::Tensor output;
  bool add_output_tensor = true;

  if (callsite_tensor && !reallocate_inputs_) {
    // If a tensor is passed as an input and we are not reallocating inputs then
    // use it and post process it immediately.
    output = *callsite_tensor;
    TF_RETURN_IF_ERROR(post_process_fn(output).status());
  } else {
    // The tensor doesn't yet have a layout, so try and defer it.
    if (has_allocation_target) {
      // If there is an allocation target then do it immediately.
      TF_ASSIGN_OR_RETURN(output, allocate_fn(input_location));
      TF_ASSIGN_OR_RETURN(output, post_process_fn(output));
    } else {
      // Otherwise defer the allocation.
      VLOG(1) << "Deferring allocation of "
              << input_location.instruction->name() << " sub tensor "
              << input_location.flattened_output_tuple_index << ".";
      TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
      TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
          input_location, std::move(allocate_fn), std::move(post_process_fn)));
      add_output_tensor = false;
    }
  }

  if (add_output_tensor) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, input_location.instruction,
                                input_location.flattened_output_tuple_index,
                                output));
  }

  return Status::OK();
}

StatusOr<TensorVector> InplaceDeferredVisitor::AddLoopInputOutputAliasingCopies(
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
  TensorVector loop_inputs(num_tensors);
  auto input_itr = loop_inputs.begin();
  for (size_t input_idx = 0; input_idx != computation_inputs_.size();
       ++input_idx) {
    absl::c_copy(computation_inputs_[input_idx], input_itr);
    input_itr = std::next(input_itr, computation_inputs_[input_idx].size());
  }
  // Outputs are already flat.
  TensorVector loop_outputs = outputs_;

  // Find all the alias information index by output tensor.
  for (unsigned int o = 0; o < num_tensors; o++) {
    int64 param_number, param_index;
    std::tie(param_number, param_index) = GetParameterNumberAndFlatIndex(o);
    const bool input_used = InputIsAllocated(param_number, param_index);
    if (input_used) {
      if (loop_inputs[o] == loop_outputs[o]) {
        alias_type[o] = AliasType::IDENTICAL_ALIAS;
      }
      // Check whether a temporary copy is required.
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

  // For partial aliasing types, create temporary tensors from outputs in order
  // to remove any aliasing.
  TensorVector unaliased_loop_outputs(loop_outputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::PARTIAL_ALIAS: {
        VLOG(1) << "Adding a partial copy in " << debug_name
                << " for tuple index " << i;
        const std::string name = absl::StrCat(debug_name, "_bodyout_temp_", i);
        unaliased_loop_outputs[i] = graph.clone(loop_outputs[i], name);
        poplar::program::Sequence& seq = GetSequenceForAliasingCopy();
        seq.add(
            poplar::program::Copy(loop_outputs[i], unaliased_loop_outputs[i]));
        break;
      }
      default:
        break;
    }
  }

  TensorVector loop_state(loop_inputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS:
      case AliasType::NO_ALIAS_USED: {
        VLOG(1) << "Adding a output to input copy in " << debug_name
                << " for tuple index " << i;
        // Get the input ready for the next iteration.
        poplar::program::Sequence& seq = GetSequenceForAliasingCopy();
        seq.add(
            poplar::program::Copy(unaliased_loop_outputs[i], loop_inputs[i]));
        break;
      }
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::NO_ALIAS_NOT_USED: {
        // The input is never used so don't need a copy - just change the while
        // loop state as by default it contains the input tensors.
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

poplar::program::Sequence&
InplaceDeferredVisitor::GetSequenceForAliasingCopy() {
  // By default just add the copies to the main sequence.
  return sequence;
}

std::pair<int64, int64> InplaceDeferredVisitor::GetParameterNumberAndFlatIndex(
    int64 output_flat_index) {
  int64 paramter_number = 0;
  int64 flat_index = output_flat_index;
  while (flat_index >
         static_cast<int64>(computation_inputs_[paramter_number].size())) {
    flat_index -= computation_inputs_[paramter_number].size();
    paramter_number++;
  }
  return {paramter_number, flat_index};
}

}  // namespace poplarplugin
}  // namespace xla
