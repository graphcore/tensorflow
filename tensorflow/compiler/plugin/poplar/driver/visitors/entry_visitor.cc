/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/ReplicatedStreamMode.hpp>

namespace xla {
namespace poplarplugin {
namespace {
// Helper function which marks all the entry computation inputs as unallocated.
DeferredArgVectors MakeArgVector(const HloComputation* comp) {
  DeferredArgVectors output(comp->num_parameters());
  for (int64 i = 0; i != comp->num_parameters(); ++i) {
    output[i].resize(
        FlattenedXlaShape(comp->parameter_instruction(i)->shape()).size());
  }
  return output;
}
}  // namespace

EntryVisitor::EntryVisitor(CompilerResources& resources,
                           const HloComputation* comp)
    : DeferredVisitor(resources, MakeArgVector(comp), true) {}

StatusOr<poplar::program::Sequence*> EntryVisitor::GetSequenceForInstruction(
    const HloInstruction* inst) {
  // Get the right sequence for stream copies, otherwise fallback to the
  // default.
  switch (inst->opcode()) {
    case HloOpcode::kParameter: {
      const auto& in_info = resources_.annotations.input_output_aliasing_map
                                .GetEntryInputInfos()[inst->parameter_number()];
      return in_info.IsStreaming() ? &sequence : &host_to_device;
    }
    default: { return DeferredVisitor::GetSequenceForInstruction(inst); }
  }
}

StatusOr<poplar::Tensor> EntryVisitor::PostProcessParameterAllocation(
    TensorSource location, const Shape& shape,
    poplar::program::Sequence& stream_copy_seq, poplar::Tensor tensor) {
  auto inst = location.first;
  auto flat_tuple_index = location.second;

  const auto& in_info = resources_.annotations.input_output_aliasing_map
                            .GetEntryInputInfos()[inst->parameter_number()];

  std::vector<Shape> module_shapes;

  const HloModule* module = inst->GetModule();
  const ComputationLayout layout = module->entry_computation_layout();
  if (layout.parameter_count() > inst->parameter_number()) {
    const Shape& mod_shape = layout.parameter_shape(inst->parameter_number());
    module_shapes = FlattenedXlaShape(mod_shape);
  }

  poplar::Graph& graph = GetGraph(resources_, inst);

  if (!UseSyntheticData()) {
    poplar::Tensor tensor_destination = tensor;
    if (!LayoutUtil::IsMonotonicWithDim0Major(
            module_shapes[flat_tuple_index].layout())) {
      // Host tensor needs to be host layout.
      tensor_destination =
          ConvertFromDeviceLayout(module_shapes[flat_tuple_index], tensor);
    }

    // Create a host stream.
    auto fifo = graph.addHostToDeviceFIFO(
        GetInputCopyHandle(inst->parameter_number(), flat_tuple_index),
        tensor_destination.elementType(), tensor_destination.numElements(),
        poplar::ReplicatedStreamMode::BROADCAST);

    stream_copy_seq.add(poplar::program::Copy(
        fifo, tensor_destination,
        !in_info.IsStreaming() || resources_.always_rearrange_copies_on_host));

  } else if (UseSyntheticData() && UseSyntheticDataInitializer()) {
    // Initialize the tensor to a constant value.
    auto& initializer = DataInitializer::GetSyntheticDataInitializer();
    TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(shape));
    TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, tensor, literal));
  }

  // If a the input to the graph is a resource variable which does not change
  // a value, then add a clone/copy to make sure it does not get overwritten
  // between runs
  if (in_info.IsResourceNotModified()) {
    poplar::Tensor non_modified_tensor = tensor;
    poplar::Graph& graph =
        GetGraphWithOutputIndex(resources_, inst, flat_tuple_index);
    tensor = graph.clone(non_modified_tensor,
                         GetDebugName(inst) + ".resource_not_modified_clone");
    sequence.add(poplar::program::Copy(non_modified_tensor, tensor));
  }
  return tensor;
}

Status EntryVisitor::FinishDeferedAllocationVisit(HloInstruction* root) {
  VLOG(1) << "Processing FinishVisit";
  HloComputation* comp = root->parent();
  if (ShapeUtil::IsEmptyTuple(root->shape())) {
    VLOG(1) << "Root instruction shape is empty tuple";
    return Status::OK();
  }

  poplar::Graph& graph = GetGraph(resources_, root);

  auto* layout = comp->parent()->mutable_entry_computation_layout();
  std::vector<Shape> shapes = FlattenedXlaShape(layout->result_shape());

  const auto& entry_outputs =
      resources_.annotations.input_output_aliasing_map.GetEntryOutputInfos();

  const uint64 num_outputs =
      root->shape().IsTuple() ? ShapeUtil::TupleElementCount(root->shape()) : 1;

  CHECK_EQ(num_outputs, entry_outputs.size());
  // Go through all the flat tensor outputs
  // *Reminder* We use depth-first flattening of nested tuples for inputs and
  // outputs
  uint64 from_tensor_index = 0;
  uint64 to_tensor_index = 0;
  // TODO see T5364
  auto out_tensors =
      FindExpandedInstructionOutputs(tensor_map, resources_, root, sequence);
  for (uint64 idx = 0; idx < entry_outputs.size(); idx++) {
    auto& out_info = entry_outputs[idx];
    poplar::program::Sequence& seq =
        out_info.IsStreaming() ? sequence : device_to_host;

    // Flatten the tuple tensor (if required) and iterate over all of them
    const auto sub_shape =
        root->shape().IsTuple()
            ? ShapeUtil::GetTupleElementShape(root->shape(), idx)
            : root->shape();
    to_tensor_index +=
        sub_shape.IsTuple() ? ShapeUtil::TupleElementCount(sub_shape) : 1;
    // all_outputs_flat_tensor_index is the global index into all the flattened
    // output tensors
    // current_output_flat_tensor_index is the local index into all the
    // flattened tensors for for output idx
    for (uint64 all_outputs_flat_tensor_index = from_tensor_index,
                current_output_flat_tensor_index = 0;
         all_outputs_flat_tensor_index < to_tensor_index;
         all_outputs_flat_tensor_index++, current_output_flat_tensor_index++) {
      if (out_info.IsResourceModified()) {
        // Get the mapped input and make sure they are the same tensor,
        // otherwise add a on device copy to make sure location of the resource
        // variable doesn't change between the runs
        // (the alternative is to reload the graph everytime)
        auto in_tensors = FindInstructionOutputs(
            tensor_map, resources_,
            comp->parameter_instruction(out_info.GetInputIndex()));
        if (in_tensors[current_output_flat_tensor_index] !=
            out_tensors[all_outputs_flat_tensor_index]) {
          sequence.add(poplar::program::Copy(
              out_tensors[all_outputs_flat_tensor_index],
              in_tensors[current_output_flat_tensor_index]));
        }
      }

      if (!UseSyntheticData()) {
        poplar::Tensor out =
            ConvertFromDeviceLayout(shapes[all_outputs_flat_tensor_index],
                                    out_tensors[all_outputs_flat_tensor_index]);

        auto fifo = graph.addDeviceToHostFIFO(
            GetOutputCopyHandle(idx, current_output_flat_tensor_index),
            out.elementType(), out.numElements());

        seq.add(poplar::program::Copy(
            out, fifo,
            !out_info.IsStreaming() ||
                resources_.always_rearrange_copies_on_host));
      }
    }
    from_tensor_index = to_tensor_index;
  }

  return Status::OK();
}

const poplar::program::Sequence EntryVisitor::GetHostToDevice() const {
  return host_to_device;
}
const poplar::program::Sequence EntryVisitor::GetDeviceToHost() const {
  return device_to_host;
}

}  // namespace poplarplugin
}  // namespace xla
