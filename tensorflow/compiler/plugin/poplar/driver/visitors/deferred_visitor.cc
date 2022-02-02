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

#include <popops/Zero.hpp>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace poplarplugin {

Status DeferredAllocations::AddDeferredAllocation(
    bool allocate_now, TensorLocation location,
    DeferredAllocateFunction allocate_fn,
    DeferredPostProcessFunction post_process_fn) {
  switch (location.instruction->opcode()) {
    case HloOpcode::kCopy:
    case HloOpcode::kInfeed:
    case HloOpcode::kParameter: {
      break;
    }
    case HloOpcode::kFusion: {
      if (IsWideConstant(location.instruction)) {
        break;
      }
      return FailedPrecondition("Fusion %s cannot be a deferred allocation.",
                                location.instruction->ToString());
    }
    case HloOpcode::kCustomCall: {
      if (IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate,
                              location.instruction) ||
          IsPoplarInstruction(PoplarOp::CreateBuffer, location.instruction) ||
          IsPoplarInstruction(PoplarOp::RemoteParameterLoad,
                              location.instruction) ||
          IsPoplarInstruction(PoplarOp::BufferLoadSlice,
                              location.instruction)) {
        break;
      }
      return FailedPrecondition(
          "Custom call %s cannot be a deferred allocation.",
          location.instruction->ToString());
    }
    default: {
      return FailedPrecondition(
          "Instruction %s cannot be a deferred allocation.",
          location.instruction->ToString());
    }
  }
  if (allocate_now) {
    // Call the allocation functions immediately.
    TF_ASSIGN_OR_RETURN(poplar::Tensor tensor, allocate_fn(location));
    TF_ASSIGN_OR_RETURN(tensor, post_process_fn(tensor));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map_, location.instruction,
                                       location.flattened_output_tuple_index,
                                       tensor));
    return Status::OK();
  }

  // Otherwise defer the allocation.
  VLOG(1) << "Deferring allocation of " << location.instruction->name()
          << " sub tensor " << location.flattened_output_tuple_index << ".";

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

const std::vector<TensorLocation>
DeferredAllocations::GetNotAllocatedLocations() const {
  // Get all input locations left to allocate.
  std::vector<TensorLocation> input_locations;
  for (auto pair : to_allocate_locations_) {
    input_locations.push_back(pair.first);
  }
  absl::c_sort(input_locations);
  return input_locations;
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
  std::vector<TensorLocation> flat_locations(locations.begin(),
                                             locations.end());
  absl::c_sort(flat_locations);
  for (const TensorLocation& location : flat_locations) {
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

namespace {
ReallocateInputsInfo GetReallocateInputsInfo(const DeferredArgRBVectors& inputs,
                                             bool reallocate) {
  ReallocateInputsInfo output;
  output.reserve(inputs.size());
  for (const auto& input : inputs) {
    output.emplace_back(input.size(), reallocate);
  }
  return output;
}
}  // namespace

DeferredVisitor::DeferredVisitor(
    CompilerResources& res, const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool allocate_all_input_tensors,
    const std::vector<const DeferredVisitor*>& dependent_computations,
    bool reallocate_inputs)
    : DeferredVisitor(
          res, callsite_inputs, debug_name_and_id, allocate_all_input_tensors,
          dependent_computations,
          GetReallocateInputsInfo(callsite_inputs, reallocate_inputs)) {}

DeferredVisitor::DeferredVisitor(
    CompilerResources& res, const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool allocate_all_input_tensors,
    const std::vector<const DeferredVisitor*>& dependent_computations,
    const ReallocateInputsInfo& reallocate_inputs_info)
    : FullVisitor(res, debug_name_and_id),
      callsite_inputs_(callsite_inputs),
      computation_inputs_(callsite_inputs.size()),
      dependent_computations_(dependent_computations),
      used_tensors_(callsite_inputs.size()),
      allocated_tensors_(callsite_inputs.size()),
      reallocate_inputs_info_(reallocate_inputs_info),
      allocate_all_input_tensors_(allocate_all_input_tensors) {
  for (size_t i = 0; i < callsite_inputs.size(); i++) {
    computation_inputs_[i].resize(callsite_inputs[i].size());
    used_tensors_[i].resize(callsite_inputs[i].size());
    allocated_tensors_[i].resize(callsite_inputs[i].size());
  }
  // When creating a new visitor explicitly push a new deferred scope
  resources_.deferred_allocation_scopes.push(DeferredAllocations{tensor_map});
}

const TensorOrRemoteBufferVectors& DeferredVisitor::inputs() const {
  return computation_inputs_;
}

const TensorOrRemoteBufferVector& DeferredVisitor::outputs() const {
  return outputs_;
}

bool DeferredVisitor::InputIsAllocated(int64 param, unsigned int index) const {
  return allocated_tensors_[param][index];
}

bool DeferredVisitor::InputIsUsed(int64 param, unsigned int index) const {
  return used_tensors_[param][index];
}

void DeferredVisitor::EnterVariableScope() {
  // Push a new vector for tracking zeroing tensors onto the stack.
  resources_.gradient_accumulation_zeroing_tensors.push({});
  // Push a new vector for tracking zeroing remote buffers onto the stack.
  resources_.gradient_accumulation_zeroing_remote_buffers.push({});
  // Push a new vector for the write undef sequences onto the stack.
  resources_.pipelining_write_undef_sequences.push({});
}

Status DeferredVisitor::ExitVariableScope() {
  // Pop the vector for tracking zeroing tensors off the stack.
  if (resources_.gradient_accumulation_zeroing_tensors.empty()) {
    return xla::FailedPrecondition(
        "Trying to pop from gradient_accumulation_zeroing_tensors, but it is "
        "empty. Was there a matching call to EnterVariableScope?");
  }
  resources_.gradient_accumulation_zeroing_tensors.pop();

  // Pop the vector for tracking zeroing remote buffers off the stack.
  if (resources_.gradient_accumulation_zeroing_remote_buffers.empty()) {
    return xla::FailedPrecondition(
        "Trying to pop from gradient_accumulation_zeroing_remote_buffers, but "
        "it is empty. Was there a matching call to EnterVariableScope?");
  }
  resources_.gradient_accumulation_zeroing_remote_buffers.pop();

  // Pop the vector for tracking write undef sequences off the stack.
  if (resources_.pipelining_write_undef_sequences.empty()) {
    return xla::FailedPrecondition(
        "Trying to pop from pipelining_write_undef_sequences, but it is "
        "empty. Was there a matching call to EnterVariableScope?");
  }
  resources_.pipelining_write_undef_sequences.pop();

  return Status::OK();
}

Status DeferredVisitor::AddSequenceForInstruction(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  if (inst->opcode() == HloOpcode::kInfeed &&
      resources_.merge_infeed_io_copies) {
    // Group all the copies for the infeed together in one sequence.
    return BaseVisitor::AppendSequenceGroupedByInstruction(inst, seq);
  } else {
    return FullVisitor::AddSequenceForInstruction(inst, seq);
  }
}

Status DeferredVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_RETURN_IF_ERROR(PreProcessParameter(inst));

  const auto param_num = inst->parameter_number();

  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());

  // Check whether this is a remote parameter.
  if (IsRemoteParameter(inst, resources_)) {
    CHECK(IsInstructionInEntryComputation(inst));
    CHECK_EQ(shapes.size(), 1);
    CHECK(shapes[0].IsArray());

    poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, 0);
    TF_ASSIGN_OR_RETURN(poplar::Type element_type, PoplarDataType(shapes[0]));
    const int64 element_count = ShapeUtil::ElementsIn(shapes[0]);

    const auto info =
        FindRemoteParameterInfo(inst->parameter_number(),
                                resources_.annotations.remote_parameter_infos);
    CHECK_NE(info, nullptr);

    TF_ASSIGN_OR_RETURN(
        auto output,
        GetOrCreateRemoteBuffer(
            graph, resources_, info->buffer_name, element_type, element_count,
            /*num_repeats=*/1, info->num_merged, info->is_replica_partitioned));

    TF_CHECK_OK(AddOutput(tensor_map, inst, 0, output));
    return Status::OK();
  }

  auto& used = used_tensors_[param_num];
  auto& allocated = allocated_tensors_[param_num];

  size_t flat_tuple_index = 0;
  for (auto index_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
    const Shape shape = index_shape.shape;
    TensorLocation input_location(inst, flat_tuple_index);
    const int64 tuple_index =
        index_shape.index.empty() ? 0 : index_shape.index[0];
    // For some computations, like entry computation, every input is forced to
    // be marked as used.
    used[flat_tuple_index] = InputIsUsedInThisComputation(inst, tuple_index);
    allocated[flat_tuple_index] =
        InputIsUsedInDependentComputations(input_location) ||
        used[flat_tuple_index];
    // Delegate the handling of parameter tensor.
    TF_RETURN_IF_ERROR(HandleParameterTensor(input_location, shape));

    flat_tuple_index++;
  }

  return Status::OK();
}

DeferredAllocateFunction DeferredVisitor::MakeParameterAllocationFunction(
    TensorLocation allocation_location, const Shape& shape,
    absl::optional<TensorOrRemoteBuffer> tensor_like,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return [this, shape, tensor_like, debug_name_and_id](
             TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
    TF_ASSIGN_OR_RETURN(auto tensor,
                        AllocateInput(allocation_location, shape, tensor_like,
                                      debug_name_and_id));
    return tensor;
  };
}

DeferredPostProcessFunction DeferredVisitor::MakeParameterPostProcessFunction(
    TensorLocation input_location, int64 param_num, const Shape& shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return [this, input_location, shape, param_num, debug_name_and_id](
             poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
    // Call the post process for this location.
    TF_ASSIGN_OR_RETURN(tensor,
                        PostProcessInputTensor(tensor, input_location, shape,
                                               debug_name_and_id));

    // Add the tensor to the computation inputs.
    computation_inputs_[param_num]
                       [input_location.flattened_output_tuple_index] = tensor;

    return tensor;
  };
}

Status DeferredVisitor::HandleParameterTensor(TensorLocation input_location,
                                              const Shape shape) {
  poplar::DebugNameAndId debug_name_and_id =
      GetDebugNameAndId(input_location.instruction);
  const auto param_num = input_location.instruction->parameter_number();
  auto& allocated = allocated_tensors_[param_num];

  // Whether this input has an allocation target.
  const bool has_allocation_target =
      HasTensorAllocationTarget(input_location, resources_);

  auto callsite_tensor =
      callsite_inputs_[param_num][input_location.flattened_output_tuple_index];

  // Function which is called when allocating this tensor.
  DeferredAllocateFunction allocate_fn = MakeParameterAllocationFunction(
      input_location, shape, callsite_tensor, debug_name_and_id);

  // Function which is called to post processing the allocation of this input.
  DeferredPostProcessFunction post_process_fn =
      MakeParameterPostProcessFunction(input_location, param_num, shape,
                                       debug_name_and_id);

  if (callsite_tensor &&
      (callsite_tensor->IsRemoteBuffer() || callsite_tensor->IsOpaque())) {
    // Add the remote buffer to the computation inputs.
    computation_inputs_[param_num]
                       [input_location.flattened_output_tuple_index] =
                           *callsite_tensor;

    TF_CHECK_OK(AddOutput(tensor_map, input_location.instruction,
                          input_location.flattened_output_tuple_index,
                          *callsite_tensor));

    return Status::OK();
  }

  if (!allocated[input_location.flattened_output_tuple_index] &&
      callsite_tensor) {
    // If it is not allocated in this computation and there is a tensor layout
    // for this location, then just forward the tensor.

    // Do not call the post process for tensors which are not allocated.
    poplar::Tensor output = *callsite_tensor;
    TF_CHECK_OK(AddOutputTensor(tensor_map, input_location.instruction,
                                input_location.flattened_output_tuple_index,
                                output));
    return Status::OK();
  }

  const bool reallocate_input =
      reallocate_inputs_info_[param_num]
                             [input_location.flattened_output_tuple_index];

  if (callsite_tensor && !reallocate_input) {
    // If a tensor is passed as an input and we are not reallocating inputs then
    // use it and post process it immediately.
    auto& graph =
        GetGraphWithOutputIndex(resources_, input_location.instruction,
                                input_location.flattened_output_tuple_index);
    poplar::Tensor output = TensorCloneAndRebalanceAliasing(
        graph, resources_, *callsite_tensor, debug_name_and_id);

    TF_RETURN_IF_ERROR(post_process_fn(output).status());
    TF_CHECK_OK(AddOutputTensor(tensor_map, input_location.instruction,
                                input_location.flattened_output_tuple_index,
                                output));
  } else {
    // The tensor is used and/or it doesn't have a layout.
    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        has_allocation_target, input_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }

  return Status::OK();
}

Status DeferredVisitor::PreProcessParameter(HloInstruction* parameter) {
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

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());

  // Multiple infeed queues are not supported.
  if (absl::c_any_of(resources_.annotations.infeed_infos,
                     [&](const CanonicalFeedInfo& info) {
                       return info.config.feed_id() != infeed_config.feed_id();
                     })) {
    return xla::FailedPrecondition(
        "Currently multiple IPUInfeedQueue in the same program are not "
        "supported.");
  }

  std::vector<Shape> shapes = FlattenedXlaShape(infeed->infeed_shape());
  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape shape = shapes[i];
    TensorLocation input_location(inst, i);

    // Function which is called when this input is allocated.
    DeferredAllocateFunction allocate_fn =
        [this, shape, debug_name_and_id](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(auto tensor, AllocateInput(allocation_location, shape,
                                                     debug_name_and_id));
      return tensor;
    };

    // Function which is called when post processing the allocation of this
    // input.
    DeferredPostProcessFunction post_process_fn =
        [this, input_location, shape,
         debug_name_and_id](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
      // Call the post process for this location.
      TF_ASSIGN_OR_RETURN(tensor,
                          PostProcessInputTensor(tensor, input_location, shape,
                                                 debug_name_and_id));

      return tensor;
    };

    const bool allocate_now =
        HasTensorAllocationTarget(input_location, resources_);

    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        allocate_now, input_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }
  has_infeed_ = true;

  CanonicalFeedInfo info(infeed_config, infeed->shape());
  TF_RETURN_IF_ERROR(AddInfeedInfo(resources_.annotations, info));

  return Status::OK();
}

Status DeferredVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

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
        poplar::program::Sequence seq({}, debug_name_and_id);

        // Cannot defer the use of this tensor, hence get the input tensor and
        // set it as output.

        TensorOrRemoteBufferVector outputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, 0,
            {flat_tuple_index, flat_tuple_index + 1}, seq, debug_name_and_id,
            false);
        CHECK_EQ(outputs.size(), 1);
        if (outputs[0].IsTensor()) {
          poplar::Tensor output = outputs[0].AsTensor();
          // Duplicate the tensor if this is not an inplace lowering.
          if (!is_lowered_inplace) {
            output = poputil::duplicate(
                graph, output, seq, {debug_name_and_id, std::to_string(i)},
                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
          }
          TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, output));
        } else {
          if (!is_lowered_inplace) {
            return xla::FailedPrecondition(
                "Unable to add copy on inplace output remote buffer at "
                "instruction %s input %d.",
                inst->name(), i);
          }
          TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, i, outputs[0]));
        }

        TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
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

StatusOr<DeferredArgRBVectors>
DeferredVisitor::GetInputsForDeferredRBInstruction(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  auto allowed_deferred_allocation_locations =
      GetAllowedDeferLocationsFromInputs(inst);
  const bool is_lowered_inplace = IsLoweredInplace(inst);

  auto inplace_description = GetInplaceDescription(inst);
  const auto& inplace_indices_set = inplace_description.GetInplaceOperandSet();
  // Go through all the operands and get the input tensors for any input that
  // cannot be deferred.
  DeferredArgRBVectors inputs(inst->operand_count());
  for (int64 operand_idx = 0; operand_idx != inst->operand_count();
       ++operand_idx) {
    const HloInstruction* input_inst = inst->operand(operand_idx);
    auto shapes = FlattenedXlaShape(input_inst->shape());

    inputs[operand_idx].resize(shapes.size());
    const bool inplace_operand = inplace_description.IsInplaceType() &&
                                 inplace_indices_set.contains(operand_idx);

    // Go through all the tensors.
    for (size_t i = 0; i < shapes.size(); i++) {
      TensorLocation input_location(input_inst, i);

      bool can_defer =
          deferred_allocation->IsDeferredAllocationLocation(input_location) &&
          allowed_deferred_allocation_locations[operand_idx][i];

      if (inplace_operand) {
        can_defer &= is_lowered_inplace;
      }

      if (!can_defer) {
        // Cannot defer the allocation, hence get the output tensor for the
        // operand.
        TF_ASSIGN_OR_RETURN(
            TensorOrRemoteBufferVector outputs,
            FindInstructionOutputsInRange(tensor_map, resources_, input_inst,
                                          {i, i + 1}));

        CHECK_EQ(outputs.size(), 1);
        inputs[operand_idx][i] = outputs[0];
      }
    }
  }
  return inputs;
}

Status DeferredVisitor::HandleDeferredAllocationTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));

  CHECK_EQ(inputs.size(), inst->operand_count());

  uint64 output_tuple_index = 0;
  for (int64 operand_idx = 0; operand_idx != inst->operand_count();
       ++operand_idx) {
    const HloInstruction* input_inst = inst->operand(operand_idx);

    auto shapes = FlattenedXlaShape(input_inst->shape());
    CHECK_EQ(inputs[operand_idx].size(), shapes.size());

    for (size_t i = 0; i < shapes.size(); i++, output_tuple_index++) {
      // Set the input and output locations.
      TensorLocation output_location(inst, output_tuple_index);
      TensorLocation input_location(input_inst, i);
      if (inputs[operand_idx][i] && inputs[operand_idx][i]->IsTensor()) {
        // If a tensor exists then just forward it.
        poplar::Tensor output = *inputs[operand_idx][i];
        TF_RETURN_IF_ERROR(
            AddOutputTensor(tensor_map, inst, output_tuple_index, output));
      } else if (inputs[operand_idx][i] &&
                 (inputs[operand_idx][i]->IsRemoteBuffer() ||
                  inputs[operand_idx][i]->IsOpaque())) {
        // If a tensor exists then just forward it.
        TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, output_tuple_index,
                                     *inputs[operand_idx][i]));
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

Status DeferredVisitor::HandleCopy(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  CHECK_EQ(inst->operand_count(), 1);
  const HloInstruction* op = inst->operand(0);

  auto dnai = GetDebugNameAndId(inst);
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, 0);
  poplar::program::Sequence seq({}, dnai);

  TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));
  TF_ASSIGN_OR_RETURN(auto clone_method_tree, GetCopyCloneMethod(inst));
  TF_ASSIGN_OR_RETURN(auto* deferred_allocation, GetDeferredAllocations());

  int64 tuple_idx = 0;
  for (auto& leaf : clone_method_tree.leaves()) {
    const auto& shape_index = leaf.first;
    const auto clone_method = leaf.second;
    TensorLocation input_location{op, tuple_idx};
    TensorLocation output_location{inst, tuple_idx};
    const Shape& input_subshape =
        ShapeUtil::GetSubshape(op->shape(), shape_index);
    auto& input = inputs[0][tuple_idx];

    const bool allocate_now =
        HasTensorAllocationTarget(input_location, resources_);

    // Deferred allocation without copy.
    if (!input && clone_method == CloneMethod_Bypass) {
      if (allocate_now) {
        TF_RETURN_IF_ERROR(deferred_allocation->MakeDeferredAllocation(
            output_location, input_location));
      } else {
        TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocationUser(
            input_location, output_location));
      }
      return Status::OK();
    }

    // Defer copy if there's no allocation target and no tensor allocated.
    // Also defer only bypass/deduce copies for now to avoid regressions.
    // TODO(T54942): Try deduce layout for all copies.
    if (!allocate_now && !input && !input_subshape.IsOpaque() &&
        !input_subshape.IsToken() &&
        (clone_method == CloneMethod_DeduceNewOrderOrPreserveAliases ||
         clone_method == CloneMethod_DeduceNewOrderOrExpandAliases)) {
      VLOG(3) << "Deferring a copy at " << inst->name();
      auto op_dnai = GetDebugNameAndId(op);
      // Allocation function for copy instruction:
      // Find instruction input and create destination tensor according to the
      // desired clone method. Don't do actual copy yet (alocation function will
      // be called for the most layout sensitive instruction and this is just a
      // fallback).
      DeferredAllocateFunction allocate_fn =
          [this, &graph, inst, op, clone_method, tuple_idx, dnai, op_dnai](
              TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
        poplar::program::Sequence seq(dnai);
        auto inputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, /*input=*/0,
            {tuple_idx, tuple_idx + 1}, seq, dnai,
            /*expand_aliasing=*/false);
        CHECK_EQ(inputs.size(), 1);
        CHECK(inputs[0].IsTensor());
        poplar::Tensor input = inputs[0].AsTensor();
        switch (clone_method) {
          case CloneMethod_DeduceNewOrderOrPreserveAliases:
            return graph.clone(
                input.elementType(), input,
                {dnai, absl::StrCat(std::to_string(tuple_idx),
                                    op_dnai.getPathName())},
                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
          case CloneMethod_DeduceNewOrderOrExpandAliases:
            return TensorCloneAndRebalanceAliasing(
                graph, resources_, input,
                {dnai, absl::StrCat(std::to_string(tuple_idx), "/",
                                    op_dnai.getPathName())});

          default:
            return FailedPrecondition("Unexpected clone method: %s",
                                      CloneMethod_Name(clone_method));
        }
      };
      // Postprocess function for copy instruction:
      // Now we have destination tensor, and we want to copy input to it.
      // In case of bypass, just return input tensor.
      DeferredPostProcessFunction postprocess_fn =
          [this, inst, op, tuple_idx, clone_method, dnai,
           op_dnai](const poplar::Tensor& tensor) -> StatusOr<poplar::Tensor> {
        if (clone_method == CloneMethod_Bypass) {
          return tensor;
        }
        poplar::program::Sequence seq(dnai);
        auto inputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, /*input=*/0,
            {tuple_idx, tuple_idx + 1}, seq, op_dnai,
            /*expand_aliasing=*/false);
        CHECK_EQ(inputs.size(), 1);
        CHECK(inputs[0].IsTensor());
        poplar::Tensor input = inputs[0].AsTensor();
        seq.add(poplar::program::Copy(
            input, tensor, false,
            {absl::StrCat(std::to_string(tuple_idx), op_dnai.getPathName())}));
        TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
        return tensor;
      };
      // Create deferred allocation at the specific location. In case of tuple
      // we create multiple allocation/postprocessing functions for each tuple
      // element.
      return deferred_allocation->AddDeferredAllocation(
          allocate_now, TensorLocation{inst, tuple_idx}, std::move(allocate_fn),
          std::move(postprocess_fn));
    }
    // Input was deferred, but it's not deducing/bypassing copy. Allocate input
    // and handle it normally.
    if (!input) {
      auto inputs = FindInstructionInputsInRange(
          tensor_map, resources_, inst, /*input=*/0, {tuple_idx, tuple_idx + 1},
          seq, dnai, /*expand_aliasing=*/false);
      CHECK_EQ(inputs.size(), 1);
      input = inputs[0];
    }

    if (input->IsTensor()) {
      auto tensor_input = input->AsTensor();
      poplar::Tensor out;
      switch (clone_method) {
        case CloneMethod_PreserveOrderAndAliases: {
          out = poputil::duplicate(
              graph, tensor_input, seq, {dnai, std::to_string(tuple_idx)},
              poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
          break;
        }
        case CloneMethod_PreserveOrderUnlessAliases: {
          out = TensorCloneAndRebalanceAliasing(graph, resources_, tensor_input,
                                                {dnai});
          seq.add(poplar::program::Copy(tensor_input, out, false, {dnai}));
          break;
        }
        case CloneMethod_Bypass: {
          out = tensor_input;
          break;
        }
        case CloneMethod_DeduceNewOrderOrPreserveAliases:
        case CloneMethod_DeduceNewOrderOrExpandAliases: {
          // Create a new tensor using "AddTensor" to get a good layout.
          if (allocate_now) {
            TF_ASSIGN_OR_RETURN(out,
                                AddTensor(graph, input_location, input_subshape,
                                          resources_, tensor_map, {dnai}));
            // Copy the original into the new layout.
            seq.add(poplar::program::Copy(tensor_input, out, false, {dnai}));
          } else if (clone_method ==
                     CloneMethod_DeduceNewOrderOrExpandAliases) {
            out = TensorCloneAndRebalanceAliasing(graph, resources_,
                                                  tensor_input, {dnai});
            seq.add(poplar::program::Copy(tensor_input, out, false, {dnai}));
          } else {
            CHECK_EQ(clone_method, CloneMethod_DeduceNewOrderOrPreserveAliases);
            // Fall back to default copy
            out = poputil::duplicate(
                graph, tensor_input, seq, {dnai, std::to_string(tuple_idx)},
                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
          }
          break;
        }
        default:
          return xla::FailedPrecondition(
              "Found invalid clone method for a copy instruction '%s' at input "
              "%d.",
              inst->name(), tuple_idx);
      }
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, out));
    } else if (input->IsOpaque()) {
      TF_CHECK_OK(
          AddOutputOpaque(tensor_map, inst, tuple_idx, inputs[tuple_idx]));
    } else {
      return xla::FailedPrecondition(
          "Found illegal remote buffer as the input to a copy instruction '%s' "
          "at input %d.",
          inst->ToString(), tuple_idx);
    }
    ++tuple_idx;
  }
  return AddSequenceForInstruction(inst, seq);
}

Status DeferredVisitor::HandleConditional(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  // Get inputs preserving any deferred allocations.
  TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));

  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence seq,
      CreateConditionalOp(resources_, inst, inputs, GetOutputShape(inst),
                          tensor_map, debug_name_and_id));
  return AddSequenceForInstruction(inst, seq);
}

Status DeferredVisitor::HandleCall(HloInstruction* inst) {
  return HandleDeferredAllocationCall(inst);
}

Status DeferredVisitor::HandleDeferredAllocationCall(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  if (IsRepeatLoop(inst)) {
    // Get inputs preserving any deferred allocations.
    TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));
    TF_ASSIGN_OR_RETURN(
        auto seq, CreateRepeatOp(resources_, inst, inputs, GetOutputShape(inst),
                                 tensor_map, debug_name_and_id));
    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    return Status::OK();
  } else if (IsPipelineOp(inst)) {
    // Get inputs preserving any deferred allocations.
    TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));

    TF_ASSIGN_OR_RETURN(
        auto seq,
        CreatePipelineOp(resources_, inst, inputs, GetOutputShape(inst),
                         tensor_map, debug_name_and_id));
    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    return Status::OK();
  } else if (IsFunction(inst)) {
    // Get inputs preserving any deferred allocations.
    TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));

    TF_ASSIGN_OR_RETURN(
        auto seq,
        CreateFunctionOp(resources_, inst, inputs, GetOutputShape(inst),
                         tensor_map, debug_name_and_id));
    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    return Status::OK();
  } else {
    return FullVisitor::HandleCall(inst);
  }
}

Status DeferredVisitor::HandleWhile(HloInstruction* inst) {
  return HandleDeferredAllocationWhile(inst);
}

Status DeferredVisitor::HandleDeferredAllocationWhile(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(auto inputs, GetInputsForDeferredRBInstruction(inst));
  TF_ASSIGN_OR_RETURN(
      auto seq, CreateWhileOp(resources_, inst, inputs, GetOutputShape(inst),
                              tensor_map, debug_name_and_id));
  TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
  return Status::OK();
}

Status DeferredVisitor::HandleCustomCall(HloInstruction* inst) {
  if (IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst)) {
    return HandleGradientAccumulatorCreate(inst);
  }
  if (IsPoplarInstruction(PoplarOp::CreateBuffer)(inst)) {
    return HandleCreateBuffer(inst);
  }
  if (IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst)) {
    return HandleRemoteParameterLoad(inst);
  }
  if (IsPoplarInstruction(PoplarOp::BufferLoadSlice)(inst)) {
    return HandleBufferLoadSlice(inst);
  }
  return HandleNonDeferredCustomCall(inst);
}

Status DeferredVisitor::HandleFusion(HloInstruction* inst) {
  if (IsWideConstant(inst)) {
    return HandleDeferredWideConst(inst);
  }
  return HandleNonDeferredFusion(inst);
}

Status DeferredVisitor::HandleNonDeferredCustomCall(HloInstruction* inst) {
  return FullVisitor::HandleCustomCall(inst);
}

Status DeferredVisitor::HandleNonDeferredFusion(HloInstruction* inst) {
  return FullVisitor::HandleFusion(inst);
}

namespace {
Status AddGradientAccumulationZeroing(CompilerResources& res,
                                      const poplar::Tensor& tensor) {
  if (res.gradient_accumulation_zeroing_tensors.empty()) {
    return FailedPrecondition("Cannot zero gradient accumulation buffer.");
  }

  res.gradient_accumulation_zeroing_tensors.top().push_back(tensor);
  return Status::OK();
}

Status AddGradientAccumulationZeroing(
    CompilerResources& res, const poplar::program::Sequence& sequence) {
  if (res.gradient_accumulation_zeroing_remote_buffers.empty()) {
    return FailedPrecondition("Cannot zero gradient accumulation buffer.");
  }

  res.gradient_accumulation_zeroing_remote_buffers.top().push_back(sequence);
  return Status::OK();
}
}  // namespace

Status DeferredVisitor::HandleDeferredWideConst(HloInstruction* inst) {
  auto dnai = GetDebugNameAndId(inst);
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, 0);

  const HloInstruction* root = inst->fused_expression_root();
  const HloInstruction* constant = root->operand(0);
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);
  const Literal& constant_literal = constant->literal();

  TensorLocation output_location = TensorLocation{inst, 0};
  Shape output_shape = GetOutputShape(inst);

  // Allocate the constant first.
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor constant_tensor,
      AddConstantTensor(graph, TensorLocation{constant, 0}, constant->shape(),
                        constant_literal, resources_, tensor_map, dnai));
  // Broadcast the tensor to the right shape.
  TF_ASSIGN_OR_RETURN(poplar::Tensor broadcasted_tensor,
                      BroadcastTensor(constant_tensor, output_shape, {}));

  const bool allocate_now =
      HasTensorAllocationTarget(output_location, resources_);
  // For wide constants, check if they have an allocation target, if so then
  // allocate the tensor with that target and copy the constant to that
  // layout.
  if (allocate_now) {
    // Doing this copy rather than allocating a big constant and calling
    // setInitialValue is a trade off between having a large tensor always
    // live and a copy + a scalar constant always being live.
    TF_ASSIGN_OR_RETURN(poplar::Tensor layout,
                        AddTensor(graph, output_location, output_shape,
                                  resources_, tensor_map, {dnai, "layout"}));
    poplar::program::Sequence seq(dnai);
    seq.add(poplar::program::Copy(broadcasted_tensor, layout, false, dnai));
    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, layout));
  } else {
    // If wide constant does not have an allocation target, defer its
    // allocation. If postprocess function gets original broadcast created by
    // op, just ignore it. If it's any other tensor, fill it with desired
    // value.
    DeferredAllocateFunction allocate_fn =
        [broadcasted_tensor](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      return broadcasted_tensor;
    };

    DeferredPostProcessFunction postprocess_fn =
        [inst, this, broadcasted_tensor,
         dnai](const poplar::Tensor& tensor) -> StatusOr<poplar::Tensor> {
      if (tensor == broadcasted_tensor) {
        // This is the original tensor we allocated, do not fill it.
        return tensor;
      }
      poplar::program::Sequence seq(dnai);
      seq.add(poplar::program::Copy(broadcasted_tensor, tensor));
      TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
      return tensor;
    };

    TF_ASSIGN_OR_RETURN(auto deferred_allocations, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocations->AddDeferredAllocation(
        allocate_now, output_location, allocate_fn, postprocess_fn));
  }
  return Status::OK();
}

Status DeferredVisitor::HandleGradientAccumulatorCreate(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  CHECK(!inst->shape().IsTuple());

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  TensorLocation output_location{inst, 0};
  const HloGradientAccumulatorCreate* create =
      Cast<HloGradientAccumulatorCreate>(inst);

  TF_ASSIGN_OR_RETURN(poplar::Type output_type, PoplarDataType(inst->shape()));

  if (create->IsRemote()) {
    CHECK(inst->shape().IsArray());

    poplar::program::Sequence zeroing_seq({}, {debug_name_and_id, "zeroing"});

    poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, 0);
    const int64 element_count = ShapeUtil::ElementsIn(inst->shape());

    auto info = create->RemoteBufferInfo();
    const bool has_info = info.has_value();
    TF_ASSIGN_OR_RETURN(
        auto output,
        GetOrCreateRemoteBuffer(
            graph, resources_, has_info ? info->name : inst->name(),
            output_type, element_count,
            /*num_repeats=*/1, has_info ? info->num_merged : 1,
            /*is_replica_partitioned=*/false));

    TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, 0, output));

    // Zero the remote buffer for the offset that we own.
    auto remote_buffer = output.AsRemoteBuffer();
    ZeroRemoteBuffer(resources_, graph, remote_buffer,
                     has_info ? info->merge_offset : 0, zeroing_seq,
                     debug_name_and_id);

    // Add the remote buffer to the zeroing stack.
    TF_RETURN_IF_ERROR(AddGradientAccumulationZeroing(resources_, zeroing_seq));
    return Status::OK();
  }

  // Function which is called when allocating this tensor.
  DeferredAllocateFunction allocate_fn;
  const bool allocate_now =
      HasTensorAllocationTarget(output_location, resources_);
  if (inst->operand_count() > 0) {
    allocate_fn =
        [this, inst, debug_name_and_id, output_type](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      poplar::Graph& graph = GetGraph(resources_, inst);

      // Get the layout of the variable passed into the gradient accumulator -
      // if it is a tensor it can be used as a guide for how to allocate the
      // gradient accumulator.
      TensorOrRemoteBufferVector outputs =
          FindInstructionOutputs(tensor_map, resources_, inst->operand(0));
      CHECK_EQ(outputs.size(), 1);
      const auto& variable_tensor = outputs[0];
      absl::optional<poplar::Tensor> tensor_like;
      if (variable_tensor.IsTensor()) {
        tensor_like = variable_tensor.AsTensor();
      }

      poplar::Tensor tensor;
      if (tensor_like && inst->user_count() > 1) {
        // Use the tensor-like layout as the layout for the accumulator.
        tensor = TensorCloneAndRebalanceAliasing(
            graph, resources_, variable_tensor, debug_name_and_id);
      } else {
        // Allocate the accumulator, and if there isn't a layout to use, use
        // the tensor-like.
        TF_ASSIGN_OR_RETURN(tensor,
                            AllocateInput(allocation_location, inst->shape(),
                                          tensor_like, debug_name_and_id));
      }

      // Handle input type different to output type by cloning.
      if (output_type != tensor.elementType()) {
        tensor = graph.clone(output_type, tensor, {debug_name_and_id});
      }

      return tensor;
    };
  } else {
    allocate_fn =
        [this, inst, debug_name_and_id](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor tensor,
          AllocateInput(allocation_location, inst->shape(), debug_name_and_id));
      return tensor;
    };
  }

  // Function which is called after the tensor has been created.
  DeferredPostProcessFunction post_process_fn =
      [this, inst](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
    // Add the tensor to the zeroing stack.
    TF_RETURN_IF_ERROR(AddGradientAccumulationZeroing(resources_, tensor));
    return tensor;
  };

  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
      allocate_now, output_location, std::move(allocate_fn),
      std::move(post_process_fn)));

  return Status::OK();
}

Status DeferredVisitor::HandleCreateBuffer(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  const HloCreateBuffer* create_buffer = Cast<HloCreateBuffer>(inst);
  const Shape& shape = inst->shape();
  CHECK(!shape.IsTuple());

  TensorLocation output_location(inst, 0);
  if (create_buffer->IsRemoteBuffer()) {
    poplar::Graph& graph = GetGraph(resources_, inst);
    TF_ASSIGN_OR_RETURN(poplar::Type element_type, PoplarDataType(shape));
    std::size_t num_repeats = ShapeUtil::GetDimension(shape, 0);
    std::size_t element_count =
        ShapeUtil::ElementsIn(ShapeUtil::DeleteDimension(0, shape));

    auto info = create_buffer->RemoteBufferInfo();
    const bool has_info = info.has_value();
    TF_ASSIGN_OR_RETURN(
        auto output,
        GetOrCreateRemoteBuffer(graph, resources_,
                                has_info ? info->name : inst->name(),
                                element_type, element_count, num_repeats,
                                has_info ? info->num_merged : 1));

    TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, 0, output));
  } else {
    // Allocate now if there is an allocation target.
    const bool allocate_now =
        HasTensorAllocationTarget(output_location, resources_);

    // Function which is called when allocating this tensor.
    DeferredAllocateFunction allocate_fn =
        [this, shape, debug_name_and_id](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor tensor,
          AllocateInput(allocation_location, shape, debug_name_and_id));
      return tensor;
    };

    // Function which is called after the tensor has been created.
    DeferredPostProcessFunction post_process_fn =
        [this, inst](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
      if (IsInPipeline(inst, resources_)) {
        // The create buffer is inside of the pipeline which means it's used
        // as a stash.
        TF_RETURN_IF_ERROR(AddSequenceForInstruction(
            inst,
            poplar::program::Sequence({poplar::program::WriteUndef(tensor)})));
      }
      return tensor;
    };

    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        allocate_now, output_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }

  return Status::OK();
}

namespace {
// TODO(T28772): Work around to make sure remote buffers can be rearranged on
// host.
std::pair<poplar::program::Sequence, poplar::program::Sequence>
AddRemoteBufferLoadCopy(poplar::Graph& graph, CompilerResources& res,
                        poplar::RemoteBuffer remote_buffer,
                        poplar::Tensor destination,
                        const poplar::DebugNameAndId& debug_name_and_id,
                        absl::optional<poplar::Tensor> offset = absl::nullopt) {
  poplar::program::Sequence stream_copy_seq({}, debug_name_and_id);
  poplar::program::Sequence temporary_copy_seq({}, debug_name_and_id);

  const auto& handle = remote_buffer.handle();
  poplar::Tensor layout_tensor;
  if (res.remote_buffer_layouts.contains(handle)) {
    layout_tensor = res.remote_buffer_layouts.at(handle);
  } else {
    layout_tensor = destination;
    res.remote_buffer_layouts[handle] = layout_tensor;
  }

  poplar::Tensor copy_tensor = graph.clone(layout_tensor, {debug_name_and_id});

  if (offset) {
    stream_copy_seq.add(poplar::program::Copy(remote_buffer, copy_tensor,
                                              *offset, {debug_name_and_id}));
  } else {
    stream_copy_seq.add(
        poplar::program::Copy(remote_buffer, copy_tensor, {debug_name_and_id}));
  }
  temporary_copy_seq.add(poplar::program::Copy(copy_tensor.flatten(),
                                               destination.flatten(), false,
                                               {debug_name_and_id}));
  return {stream_copy_seq, temporary_copy_seq};
}
}  // namespace

Status DeferredVisitor::HandleRemoteParameterLoad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  const auto* load_inst = Cast<HloRemoteParameterLoad>(inst);
  const int64 num_inputs = inst->operand_count();

  const auto shapes = FlattenedXlaShape(inst->shape());
  CHECK_EQ(shapes.size(), num_inputs);

  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape shape = shapes[i];
    TensorLocation input_location(inst, i);

    // Function which is called when allocating this tensor.
    DeferredAllocateFunction allocate_fn =
        [this, shape, debug_name_and_id](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor tensor,
          AllocateInput(allocation_location, shape, debug_name_and_id));
      return tensor;
    };

    // Function which is called after the tensor has been created - this
    // function copies the values from the buffer into the tensor.
    DeferredPostProcessFunction post_process_fn =
        [this, load_inst, i, shape,
         debug_name_and_id](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
      poplar::Graph& shard_graph =
          GetGraphWithOutputIndex(resources_, load_inst, i);

      poplar::program::Sequence stream_copy_seq({}, debug_name_and_id);
      poplar::program::Sequence temporary_copy_seq({}, debug_name_and_id);
      if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
        if (UseSyntheticDataInitializer()) {
          // Initialize the tensor to a constant value.
          auto& initializer = DataInitializer::GetSyntheticDataInitializer();
          TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(shape));
          TF_RETURN_IF_ERROR(
              SetInitialTensorValue(shard_graph, tensor, literal));
        }
        stream_copy_seq.add(poplar::program::WriteUndef(tensor));
      } else {
        TensorOrRemoteBufferVector inputs =
            FindInstructionInputs(tensor_map, resources_, load_inst, i,
                                  stream_copy_seq, debug_name_and_id, true);

        CHECK_EQ(inputs.size(), 1);
        const TensorOrRemoteBuffer& input = inputs[0];

        const uint64 load_replication_factor =
            input.IsReplicaPartitioned()
                ? resources_.partition_replication_factor
                : 1;
        CHECK_EQ(load_inst->GetReplicationFactor(i), load_replication_factor)
            << load_inst->ToString();

        if (!input.IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Expected a Poplar RemoteBuffer as operand %d to %s", i,
              GetDebugName(load_inst));
        }

        poplar::RemoteBuffer remote_buffer = input.AsRemoteBuffer();
        auto pair_seq = AddRemoteBufferLoadCopy(
            shard_graph, resources_, remote_buffer, tensor, debug_name_and_id);
        stream_copy_seq.add(pair_seq.first);
        temporary_copy_seq.add(pair_seq.second);
      }

      // Add grouped such that all copies from the same instruction are
      // grouped together in the sequence, allowing Poplar to merge them.
      TF_RETURN_IF_ERROR(
          PrependSequenceGroupedByInstruction(load_inst, stream_copy_seq));
      TF_RETURN_IF_ERROR(
          AppendSequenceGroupedByInstruction(load_inst, temporary_copy_seq));

      return tensor;
    };
    const bool allocate_now =
        HasTensorAllocationTarget(input_location, resources_);

    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        allocate_now, input_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }

  return Status::OK();
}

Status DeferredVisitor::HandleBufferLoadSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  const auto* load_inst = Cast<HloBufferLoadSlice>(inst);
  const int64 num_outputs = load_inst->RemoteBuffers().size();

  const auto shapes = FlattenedXlaShape(inst->shape());
  CHECK_EQ(shapes.size(), num_outputs);

  for (size_t i = 0; i < shapes.size(); i++) {
    const Shape shape = shapes[i];
    TensorLocation input_location(inst, i);

    // Function which is called when allocating this tensor.
    DeferredAllocateFunction allocate_fn =
        [this, shape, debug_name_and_id](
            TensorLocation allocation_location) -> StatusOr<poplar::Tensor> {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor tensor,
          AllocateInput(allocation_location, shape, debug_name_and_id));
      return tensor;
    };

    // Function which is called after the tensor has been created - this
    // function copies the values from the buffer into the tensor.
    DeferredPostProcessFunction post_process_fn =
        [this, load_inst, i, num_outputs, shape,
         debug_name_and_id](poplar::Tensor tensor) -> StatusOr<poplar::Tensor> {
      poplar::Graph& shard_graph =
          GetGraphWithOutputIndex(resources_, load_inst, i);

      poplar::program::Sequence stream_copy_seq({}, debug_name_and_id);
      poplar::program::Sequence temporary_copy_seq({}, debug_name_and_id);
      if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
        if (UseSyntheticDataInitializer()) {
          // Initialize the tensor to a constant value.
          auto& initializer = DataInitializer::GetSyntheticDataInitializer();
          TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(shape));
          TF_RETURN_IF_ERROR(
              SetInitialTensorValue(shard_graph, tensor, literal));
        }
        stream_copy_seq.add(poplar::program::WriteUndef(tensor));
      } else {
        // Get the remote buffer input.
        TensorOrRemoteBufferVector inputs =
            FindInstructionInputs(tensor_map, resources_, load_inst, i,
                                  stream_copy_seq, debug_name_and_id);
        CHECK_EQ(inputs.size(), 1);
        const TensorOrRemoteBuffer& input = inputs[0];

        const uint64 load_replication_factor =
            input.IsReplicaPartitioned()
                ? resources_.partition_replication_factor
                : 1;
        CHECK_EQ(load_inst->GetReplicationFactor(i), load_replication_factor)
            << load_inst->ToString();

        poplar::RemoteBuffer remote_buffer = input.AsRemoteBuffer();

        TF_ASSIGN_OR_RETURN(
            poplar::Tensor offset,
            FindInstructionInput(tensor_map, resources_, load_inst,
                                 num_outputs + i, stream_copy_seq,
                                 debug_name_and_id));

        auto pair_seq =
            AddRemoteBufferLoadCopy(shard_graph, resources_, remote_buffer,
                                    tensor, debug_name_and_id, offset);
        stream_copy_seq.add(pair_seq.first);
        temporary_copy_seq.add(pair_seq.second);
      }

      // Add grouped such that all copies from the same instruction are
      // grouped together in the sequence, allowing Poplar to merge them.
      TF_RETURN_IF_ERROR(
          PrependSequenceGroupedByInstruction(load_inst, stream_copy_seq));
      TF_RETURN_IF_ERROR(
          AppendSequenceGroupedByInstruction(load_inst, temporary_copy_seq));

      return tensor;
    };
    const bool allocate_now =
        HasTensorAllocationTarget(input_location, resources_);

    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        allocate_now, input_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }

  return Status::OK();
}

Status DeferredVisitor::FinishScopedVisit(HloInstruction* inst) {
  // By default allocate all inputs into a callsite.
  // Go through all the unallocated input tensors and allocate all the required
  // ones.
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
  for (auto input_location : deferred_allocation->GetNotAllocatedLocations()) {
    const HloInstruction* inst = input_location.instruction;
    const bool is_required_parameter =
        inst->opcode() == HloOpcode::kParameter &&
        InputIsAllocated(input_location.instruction->parameter_number(),
                         input_location.flattened_output_tuple_index);

    // Force the allocation.
    if (allocate_all_input_tensors_ || is_required_parameter) {
      VLOG(1) << "Allocating input " << inst->ToString() << " index "
              << input_location.flattened_output_tuple_index;
      TF_RETURN_IF_ERROR(deferred_allocation->MakeDeferredAllocation(
          input_location, input_location));
    }
  }

  outputs_ = FindInstructionOutputs(tensor_map, resources_, inst);
  // Delegate.
  TF_RETURN_IF_ERROR(FinishDeferedAllocationVisit(inst));
  resources_.tensor_maps.AddTensorMapForComputation(inst->parent()->name(),
                                                    std::move(tensor_map));
  resources_.deferred_allocation_scopes.pop();
  return Status::OK();
}

Status DeferredVisitor::FinishDeferedAllocationVisit(HloInstruction* inst) {
  return Status::OK();
}

Status DeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst,
    const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return PropagateDeferredAllocations(
      callsite_inst, callsite_inputs,
      std::vector<bool>(callsite_inst->operand_count(), true),
      debug_name_and_id);
}

Status DeferredVisitor::PropagateDeferredAllocationsOperand(
    const HloInstruction* callsite_inst, int64 operand_idx, int64 parameter_idx,
    const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return PropagateDeferredAllocationsOperand(
      callsite_inst, operand_idx, parameter_idx, callsite_input,
      /*add_clone=*/true, debug_name_and_id);
}

Status DeferredVisitor::PropagateDeferredAllocationsOperand(
    const HloInstruction* callsite_inst, int64 operand_idx, int64 parameter_idx,
    const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
    bool add_clone, const poplar::DebugNameAndId& debug_name_and_id) {
  // Note that this is called after finish visit, so tensors are being added to
  // the scope which called the visitor.
  TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());

  const HloInstruction* input_inst = callsite_inst->operand(operand_idx);
  auto shapes = FlattenedXlaShape(input_inst->shape());
  for (size_t i = 0; i < shapes.size(); i++) {
    if (!callsite_input[i]) {
      VLOG(1) << "Propagating allocated tensor at callsite "
              << callsite_inst->name() << " for input (" << input_inst->name()
              << ", " << i << ").";
      poplar::Tensor t = computation_inputs_[parameter_idx][i];

      // Depending on the visitor inplace usage, the tensor might need to be
      // cloned so that it is not clobbered unexpectedly.
      if (add_clone) {
        poplar::Graph& graph =
            GetGraphWithOutputIndex(resources_, input_inst, i);

        t = graph.clone(t, {debug_name_and_id, std::to_string(i)},
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
      }
      TF_RETURN_IF_ERROR(
          deferred_allocation->PostProcessAllocation({input_inst, i}, t));
    }
  }

  return Status::OK();
}

Status DeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst,
    const DeferredArgRBVectors& callsite_inputs, std::vector<bool> add_clone,
    const poplar::DebugNameAndId& debug_name_and_id) {
  for (int64 operand_idx = 0; operand_idx != callsite_inst->operand_count();
       ++operand_idx) {
    TF_RETURN_IF_ERROR(PropagateDeferredAllocationsOperand(
        callsite_inst, operand_idx, operand_idx, callsite_inputs[operand_idx],
        add_clone[operand_idx], debug_name_and_id));
  }

  return Status::OK();
}

StatusOr<poplar::Tensor> DeferredVisitor::AllocateInput(
    TensorLocation allocation_location, const Shape& shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return AllocateInput(allocation_location, shape,
                       absl::optional<poplar::Tensor>(absl::nullopt),
                       debug_name_and_id);
}

StatusOr<poplar::Tensor> DeferredVisitor::AllocateInput(
    TensorLocation allocation_location, const Shape& shape,
    absl::optional<TensorOrRemoteBuffer> tensor_like,
    const poplar::DebugNameAndId& debug_name_and_id) {
  CHECK(!tensor_like || tensor_like->IsTensor());

  if (tensor_like) {
    return AllocateInput(allocation_location, shape, tensor_like->AsTensor(),
                         debug_name_and_id);
  } else {
    return AllocateInput(allocation_location, shape,
                         absl::optional<poplar::Tensor>(absl::nullopt),
                         debug_name_and_id);
  }
}

StatusOr<poplar::Tensor> DeferredVisitor::AllocateInput(
    TensorLocation allocation_location, const Shape& shape,
    absl::optional<poplar::Tensor> tensor_like,
    const poplar::DebugNameAndId& debug_name_and_id) {
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
    TF_ASSIGN_OR_RETURN(out,
                        AddTensor(graph, allocation_location, shape, resources_,
                                  tensor_map, {debug_name_and_id}));
  } else {
    out = TensorCloneAndRebalanceAliasing(graph, resources_, *tensor_like,
                                          debug_name_and_id);
  }

  return out;
}

StatusOr<poplar::Tensor> DeferredVisitor::PostProcessInputTensor(
    poplar::Tensor tensor, TensorLocation input_location, const Shape& shape,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);

  // Each visitor might need to post process the allocation, for example change
  // add copies to a sequence. Use the input location for that information.
  switch (input_location.instruction->opcode()) {
    case HloOpcode::kInfeed: {
      TF_ASSIGN_OR_RETURN(
          tensor, PostProcessInfeedAllocation(input_location, shape, seq,
                                              tensor, debug_name_and_id));
      break;
    }
    case HloOpcode::kParameter: {
      TF_ASSIGN_OR_RETURN(
          tensor, PostProcessParameterAllocation(input_location, shape, seq,
                                                 tensor, debug_name_and_id));
      break;
    }
    default: {
      return xla::FailedPrecondition(
          "Unsupported input allocation for opcode %s.",
          HloOpcodeString(input_location.instruction->opcode()).c_str());
    }
  }

  // Add the sequence where any post processing operations should be added to
  // (for example stream copies).
  TF_RETURN_IF_ERROR(
      AddSequenceForInstruction(input_location.instruction, seq));

  return tensor;
}

StatusOr<poplar::Tensor> DeferredVisitor::PostProcessInfeedAllocation(
    TensorLocation location, const Shape& shape,
    poplar::program::Sequence& sequence, poplar::Tensor tensor,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateInfeed(resources_, location.instruction,
                                   location.flattened_output_tuple_index, shape,
                                   tensor, debug_name_and_id));
  sequence.add(prog);
  return tensor;
}

StatusOr<DeferredAllocations*> DeferredVisitor::GetDeferredAllocations() {
  if (resources_.deferred_allocation_scopes.empty()) {
    return FailedPrecondition("Cannot get the DeferredAllocations.");
  }
  return &resources_.deferred_allocation_scopes.top();
}

bool DeferredVisitor::InputIsUsedInThisComputation(const HloInstruction* inst,
                                                   int64 tuple_index) {
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

  for (auto user : inst->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return true;
    }

    if (user->tuple_index() == tuple_index) {
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

poplar::program::Sequence DeferredVisitor::GetSequence(
    bool copy_execution_counters) {
  poplar::program::Sequence seq({}, dnai_);
  if (copy_execution_counters) {
    TF_CHECK_OK(
        CopyExecutionCountersFromScope(resources_, execution_counters_, seq));
  }
  seq.add(FullVisitor::GetRawSequence());
  return seq;
}

poplar::program::Sequence DeferredVisitor::GetFunctionCall() {
  if (!function_) {
    // Do not copy the execution counters as part of the function - the
    // callsites might be different.
    poplar::program::Sequence func_seq = GetSequence(false);
    function_ = GetMasterGraph(resources_).addFunction(func_seq);
  }

  poplar::program::Sequence seq({}, dnai_);
  TF_CHECK_OK(
      CopyExecutionCountersFromScope(resources_, execution_counters_, seq));
  seq.add(poplar::program::Call(*function_, {dnai_}));
  return seq;
}

InplaceDeferredVisitor::InplaceDeferredVisitor(
    CompilerResources& res, const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id,
    const std::vector<const DeferredVisitor*>& dependent_subcomputations,
    bool reallocate_inputs)
    : InplaceDeferredVisitor(
          res, inputs, description, debug_name_and_id,
          dependent_subcomputations,
          GetReallocateInputsInfo(inputs, reallocate_inputs)) {}

InplaceDeferredVisitor::InplaceDeferredVisitor(
    CompilerResources& res, const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id,
    const std::vector<const DeferredVisitor*>& dependent_subcomputations,
    const ReallocateInputsInfo& reallocate_inputs_info)
    : DeferredVisitor(res, inputs, debug_name_and_id, true,
                      dependent_subcomputations, reallocate_inputs_info),
      description_(description) {}

Status InplaceDeferredVisitor::PropagateDeferredAllocations(
    const HloInstruction* callsite_inst,
    const DeferredArgRBVectors& callsite_inputs,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return DeferredVisitor::PropagateDeferredAllocations(
      callsite_inst, callsite_inputs,
      std::vector<bool>(callsite_inst->operand_count(), false),
      debug_name_and_id);
}

Status InplaceDeferredVisitor::PropagateDeferredAllocationsOperand(
    const HloInstruction* callsite_inst, int64 operand_idx, int64 parameter_idx,
    const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return DeferredVisitor::PropagateDeferredAllocationsOperand(
      callsite_inst, operand_idx, parameter_idx, callsite_input,
      /*add_clone=*/false, debug_name_and_id);
}

StatusOr<poplar::program::Sequence> InplaceDeferredVisitor::GetPreambleCopies(
    const poplar::DebugNameAndId& debug_name_and_id) {
  CHECK_EQ(callsite_inputs_.size(), computation_inputs_.size());
  CHECK_EQ(callsite_inputs_.size(), reallocate_inputs_info_.size());
  poplar::program::Sequence seq({}, debug_name_and_id);
  for (uint64 i = 0; i != callsite_inputs_.size(); ++i) {
    CHECK_EQ(callsite_inputs_[i].size(), computation_inputs_[i].size());
    CHECK_EQ(callsite_inputs_[i].size(), reallocate_inputs_info_[i].size());
    for (uint64 j = 0; j != callsite_inputs_[i].size(); ++j) {
      if (callsite_inputs_[i][j] &&
          *callsite_inputs_[i][j] != computation_inputs_[i][j]) {
        // For inplace vistors, they should only differ if we allow relocation.
        if (!reallocate_inputs_info_[i][j]) {
          return FailedPrecondition("Input should have not been reallocated.");
        }
        VLOG(1) << "Adding a copy for input (" << i << ", " << j << ").";
        if (callsite_inputs_[i][j]->IsTensor()) {
          seq.add(poplar::program::Copy(callsite_inputs_[i][j]->AsTensor(),
                                        computation_inputs_[i][j].AsTensor(),
                                        false, debug_name_and_id));
        } else {
          *callsite_inputs_[i][j] = computation_inputs_[i][j].AsOpaque();
        }
      }
    }
  }
  return seq;
}

Status InplaceDeferredVisitor::HandleParameterTensor(
    TensorLocation input_location, const Shape shape) {
  poplar::DebugNameAndId debug_name_and_id =
      GetDebugNameAndId(input_location.instruction);
  const auto param_num = input_location.instruction->parameter_number();

  // Whether this input has an allocation target.
  const bool has_allocation_target =
      HasTensorAllocationTarget(input_location, resources_);

  auto callsite_tensor =
      callsite_inputs_[param_num][input_location.flattened_output_tuple_index];

  // Function which is called when tensor is allocated for this input.
  DeferredAllocateFunction allocate_fn = MakeParameterAllocationFunction(
      input_location, shape, callsite_tensor, debug_name_and_id);

  // Function which is called when post processing the allocation of this input.
  DeferredPostProcessFunction post_process_fn =
      MakeParameterPostProcessFunction(input_location, param_num, shape,
                                       debug_name_and_id);

  const bool reallocate_input =
      reallocate_inputs_info_[param_num]
                             [input_location.flattened_output_tuple_index];

  if (callsite_tensor &&
      (callsite_tensor->IsRemoteBuffer() || callsite_tensor->IsOpaque())) {
    // Add the remote buffer or opaque to the computation inputs.
    computation_inputs_[param_num]
                       [input_location.flattened_output_tuple_index] =
                           *callsite_tensor;

    TF_CHECK_OK(AddOutput(tensor_map, input_location.instruction,
                          input_location.flattened_output_tuple_index,
                          *callsite_tensor));

    return Status::OK();
  }

  if (callsite_tensor && !reallocate_input) {
    // If a tensor is passed as an input and we are not reallocating inputs then
    // use it and post process it immediately.
    poplar::Tensor output = *callsite_tensor;
    TF_RETURN_IF_ERROR(post_process_fn(output).status());
    TF_CHECK_OK(AddOutputTensor(tensor_map, input_location.instruction,
                                input_location.flattened_output_tuple_index,
                                output));
  } else {
    TF_ASSIGN_OR_RETURN(auto deferred_allocation, GetDeferredAllocations());
    TF_RETURN_IF_ERROR(deferred_allocation->AddDeferredAllocation(
        has_allocation_target, input_location, std::move(allocate_fn),
        std::move(post_process_fn)));
  }

  return Status::OK();
}

StatusOr<TensorOrRemoteBufferVector>
InplaceDeferredVisitor::AddLoopInputOutputAliasingCopies(
    poplar::Graph& graph, const HloComputation* computation,
    const poplar::DebugNameAndId& debug_name_and_id) {
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

  const int64 num_tensors = outputs_.size();
  std::vector<AliasType> alias_type(num_tensors, AliasType::NO_ALIAS_USED);

  // Create a flat version of the loop inputs.
  TensorOrRemoteBufferVector loop_inputs;
  for (int64 i = 0; i < computation_inputs_.size(); ++i) {
    for (int64 k = 0; k < computation_inputs_[i].size(); ++k) {
      loop_inputs.push_back(computation_inputs_[i][k]);
    }
  }

  // Outputs are already flat.
  TensorOrRemoteBufferVector loop_outputs = outputs_;

  // Find all the alias information index by output tensor.
  for (int64 o = 0; o < num_tensors; o++) {
    int64 param_number, param_index;
    std::tie(param_number, param_index) = GetParameterNumberAndFlatIndex(o);

    const bool input_used = InputIsAllocated(param_number, param_index);
    if (input_used) {
      if (loop_inputs[o] == loop_outputs[o]) {
        alias_type[o] = AliasType::IDENTICAL_ALIAS;
      }
      // Check whether a temporary copy is required.
      for (int64 i = 0; i < num_tensors; i++) {
        int64 input_param_number, input_param_index;
        std::tie(input_param_number, input_param_index) =
            GetParameterNumberAndFlatIndex(i);

        if ((alias_type[o] != AliasType::IDENTICAL_ALIAS || i != o) &&
            InputIsAllocated(input_param_number, input_param_index)) {
          if (loop_outputs[o].IsTensor() && loop_inputs[i].IsTensor() &&
              loop_outputs[o].AsTensor().intersectsWith(
                  loop_inputs[i].AsTensor())) {
            alias_type[o] = AliasType::PARTIAL_ALIAS;
          }
        }
      }
    } else {
      // If the input is not used, check that the output at that index does not
      // alias any of the inputs which might have changed during
      // computation.
      alias_type[o] = AliasType::NO_ALIAS_NOT_USED;
      for (int64 i = 0; i < num_tensors; i++) {
        int64 input_param_number, input_param_index;
        std::tie(input_param_number, input_param_index) =
            GetParameterNumberAndFlatIndex(i);
        if (InputIsAllocated(input_param_number, input_param_index)) {
          if (loop_outputs[o].IsTensor() && loop_inputs[i].IsTensor() &&
              loop_outputs[o].AsTensor().intersectsWith(
                  loop_inputs[i].AsTensor())) {
            alias_type[o] = AliasType::PARTIAL_ALIAS_OUTPUT_ONLY;
          }
        }
      }
    }

    // If this input is not inplace, then it has to be identical.
    if (!description_.GetInplaceOperandSet().contains(param_number)) {
      // Input and output can only be different iff the input is not parallel
      // writable.
      if (loop_inputs[o] != loop_outputs[o]) {
        CHECK(loop_inputs[o].IsRemoteBuffer() ||
              !loop_inputs[o].AsTensor().isParallelWriteable());
        CHECK(alias_type[o] == AliasType::NO_ALIAS_NOT_USED ||
              alias_type[o] == AliasType::NO_ALIAS_USED);
      } else {
        CHECK(alias_type[o] == AliasType::IDENTICAL_ALIAS);
      }
      alias_type[o] = AliasType::IDENTICAL_ALIAS;
    }
  }

  // For partial aliasing types, create temporary tensors from outputs in order
  // to remove any aliasing.
  TensorOrRemoteBufferVector unaliased_loop_outputs(loop_outputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::PARTIAL_ALIAS: {
        // Remote buffers can't alias.
        if (unaliased_loop_outputs[i].IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Found disallowed remote buffer aliasing at input %d of loop %s",
              i, computation->name());
        }
        VLOG(1) << "Adding a partial copy in "
                << debug_name_and_id.getPathName() << " for tuple index " << i;
        unaliased_loop_outputs[i] = graph.clone(
            loop_outputs[i], {debug_name_and_id, std::string("bodyout_temp_") +
                                                     std::to_string(i)});
        AddSequenceForAliasingCopy(
            computation->root_instruction(),
            poplar::program::Sequence(
                {poplar::program::Copy(loop_outputs[i].AsTensor(),
                                       unaliased_loop_outputs[i].AsTensor(),
                                       false, {debug_name_and_id})}));
        break;
      }
      default:
        break;
    }
  }

  TensorOrRemoteBufferVector loop_state(loop_inputs);
  for (int64 i = 0; i < num_tensors; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS:
      case AliasType::NO_ALIAS_USED: {
        // Remote buffers can't alias.
        if (loop_state[i].IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Found disallowed remote buffer aliasing at input %d of loop %s",
              i, computation->name());
        }
        VLOG(1) << "Adding a output to input copy in "
                << debug_name_and_id.getPathName() << " for tuple index " << i;
        if (loop_inputs[i].IsOpaque()) {
          // Opaque inputs are just forwarded because they never really alias
          unaliased_loop_outputs[i] = loop_inputs[i].AsOpaque();
        } else {
          // Get the input ready for the next iteration.
          AddSequenceForAliasingCopy(
              computation->root_instruction(),
              poplar::program::Sequence({poplar::program::Copy(
                  unaliased_loop_outputs[i].AsTensor(),
                  loop_inputs[i].AsTensor(), false, {debug_name_and_id})}));
        }
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

void InplaceDeferredVisitor::AddSequenceForAliasingCopy(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  // By default just add the copies as a regular sequence for the instruction.
  TF_CHECK_OK(FullVisitor::AddSequenceForInstruction(inst, seq));
}

std::pair<int64, int64> InplaceDeferredVisitor::GetParameterNumberAndFlatIndex(
    int64 output_flat_index) {
  int64 parameter_number = 0;
  int64 flat_index = output_flat_index;
  while (flat_index >=
         static_cast<int64>(computation_inputs_[parameter_number].size())) {
    flat_index -= computation_inputs_[parameter_number].size();
    parameter_number++;
  }
  return {parameter_number, flat_index};
}

}  // namespace poplarplugin
}  // namespace xla
