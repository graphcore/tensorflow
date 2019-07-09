/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_allocation_visitor.h"

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popsys/CycleCount.hpp>

#include "google/protobuf/util/message_differencer.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

Status DeferredAllocationVisitor::AllocateInput(const HloInstruction* inst,
                                                const int64 flat_tuple_index,
                                                const Shape& shape) {
  poplar::Graph& graph =
      GetGraphWithOutputIndex(resources_, inst, flat_tuple_index);

  auto source = std::make_pair(inst, flat_tuple_index);

  // Do the allocation
  VLOG(2) << "Allocating input tensor for " << inst->name() << ":"
          << flat_tuple_index << " shape " << shape.ToString() << " on shard "
          << GetShardForOutputIndex(inst, flat_tuple_index);
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      AddTensor(graph, source, shape, resources_, tensor_map));

  // Each visitor might need to post process the allocation, for example change
  // the layout of the tensor - this depends on the original input op type.
  const HloInstruction* original_input = inst;
  int64 original_flat_tuple_index = flat_tuple_index;
  // For deferred allocation the original input is the back of the
  // deferred_allocations_path
  if (IsDeferredAllocation(inst, flat_tuple_index)) {
    auto& deferred_allocations_path =
        resources_.annotations.tensor_allocation_map.at(source)
            .deferred_allocations_path;
    original_input = deferred_allocations_path.back().first;
    original_flat_tuple_index = deferred_allocations_path.back().second;
  }

  switch (original_input->opcode()) {
    case HloOpcode::kInfeed: {
      TF_ASSIGN_OR_RETURN(
          out, PostProcessInfeedAllocation(
                   original_input, original_flat_tuple_index, shape, out));
      break;
    }
    case HloOpcode::kParameter: {
      TF_ASSIGN_OR_RETURN(
          out, PostProcessParameterAllocation(
                   original_input, original_flat_tuple_index, shape, out));
      break;
    }
    default: {
      return xla::FailedPrecondition(
          "Unsupported input allocation for opcode %s.",
          HloOpcodeString(original_input->opcode()).c_str());
    }
  }

  // Add the post processed tensor.
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, flat_tuple_index, out));

  // If this was a deferred allocation, we need to make sure that all the ops
  // which we skipped the allocation for have their tensor set to this
  // allocation.
  if (IsDeferredAllocation(inst, flat_tuple_index)) {
    auto& deferred_allocations_path =
        resources_.annotations.tensor_allocation_map.at(source)
            .deferred_allocations_path;
    for (TensorSource deferred : deferred_allocations_path) {
      TF_RETURN_IF_ERROR(
          AddOutputTensor(tensor_map, deferred.first, deferred.second, out));
      original_input = deferred.first;
      original_flat_tuple_index = deferred.second;
    }
  }
  return Status::OK();
}

Status DeferredAllocationVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // Go through all the shapes for inst, don't allocate any tensors which are
  // marked as deferred.
  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
  bool defer_any_allocations = false;
  bool allocate = false;
  // First check if there are any deferred allocation - if not we can call the
  // parent class to deal with it.
  for (int64 i = 0; i < shapes.size(); i++) {
    defer_any_allocations |= IsInDeferredAllocationPath(inst, i);
    allocate |= IsDeferredAllocation(inst, i);
  }

  if (allocate) {
    if (shapes.size() != 1) {
      return xla::FailedPrecondition(
          "Trying to allocate multiple deferred tensors.");
    }
    return AllocateInput(inst, 0, shapes[0]);
  } else if (defer_any_allocations) {
    // Note that the forward allocation finder makes sure that this is inplace -
    // we therefore don't need to worry about copies.
    const int64 offset =
        InsertIntoTuple(inst->operand(0)->shape(), inst->tuple_index(), 0);
    for (int64 i = 0; i < shapes.size(); i++) {
      if (!IsInDeferredAllocationPath(inst, i)) {
        // Get the tensor for this shape and assign it as an output.
        auto range = std::make_pair(offset + i, offset + i + 1);
        auto outputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, 0, range, sequence, false);
        CHECK_EQ(outputs.size(), 1);
        TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, outputs[0]));
      }
    }
    return Status::OK();
  } else {
    return FullVisitor::HandleGetTupleElement(inst);
  }
}

Status DeferredAllocationVisitor::HandleInfeed(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // We currently have no way of ordering infeeds in the same compuation - this
  // can result in unexpected results.
  if (has_infeed_) {
    return xla::FailedPrecondition(
        "Currently calling `get_next()` multiple times on the same "
        "IPUInfeedQueue in the same computation block is not supported.");
  }

  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());
  // We allow the same infeed queue to be dequeued multiple times, however
  // we don't support multiple infeed queues in the same program.
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
        "the `replication_factor` to %d when creating IPUInfeedQueue.",
        resources_.replication_factor, infeed_config.replication_factor(),
        resources_.replication_factor);
  }

  std::vector<Shape> shapes = FlattenedXlaShape(infeed->infeed_shape());
  for (int64 i = 0; i < shapes.size(); i++) {
    if (CanDeferAllocation(inst, i)) {
      VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
              << i << ".";
      DeferAllocation(inst, i);
    } else {
      TF_RETURN_IF_ERROR(AllocateInput(inst, i, shapes[i]));
    }
  }
  has_infeed_ = true;

  FeedInfo info(infeed->name(), infeed_config, infeed->shape());
  resources_.annotations.infeed_infos.push_back(info);

  return Status::OK();
}

StatusOr<poplar::Tensor> DeferredAllocationVisitor::PostProcessInfeedAllocation(
    const HloInstruction* inst, const int64 flat_tuple_index,
    const Shape& shape, poplar::Tensor tensor) {
  const HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);

  poplar::Graph& graph = GetGraph(resources_, inst);

  // Parse the infeed config to find out how much data to prefetch if at all.
  xla::poplarplugin::PoplarFeedConfig infeed_config;
  infeed_config.ParseFromString(infeed->infeed_config());

  poplar::Graph& master_graph = GetMasterGraph(resources_);

  // The amount of data the user has specified to be prefetched on each host
  // sync.
  size_t data_to_prefetch =
      std::max<size_t>(1, infeed_config.data_to_prefetch());

  poplar::program::Sequence& seq =
      resources_.merge_infeed_io_copies ? merged_infeed_sequence : sequence;

  // A functor wrapper to either use synthetic data or copy from the host,
  // depending on the global synthetic flags.
  auto init_synthetic_or_copy = [&](poplar::program::Sequence& seq,
                                    const Shape& data_shape,
                                    poplar::Tensor& tensor_to_update) {
    if (!UseSyntheticData()) {
      auto fifo = graph.addHostToDeviceFIFO(
          GetInfeedCopyHandle(infeed->name(), flat_tuple_index),
          tensor_to_update.elementType(), tensor_to_update.numElements());
      seq.add(poplar::program::Copy(fifo, tensor_to_update, false));
    } else if (UseSyntheticData() && UseSyntheticDataInitializer()) {
      // Initialize the tensor with a synthetic initalizer.
      auto& initializer = DataInitializer::GetSyntheticDataInitializer();
      TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(data_shape));
      TF_RETURN_IF_ERROR(
          SetInitialTensorValue(graph, tensor_to_update, literal));
    }
    // If neither case then we want synthetic data but don't want it initalized
    // so we just return the empty tensor unchanged.
    return Status::OK();
  };

  if (data_to_prefetch != 1) {
    // Extend the old shape to add a new dimension for the batches of memory.
    std::vector<size_t> new_shape = tensor.shape();
    new_shape.insert(new_shape.begin(), data_to_prefetch);

    // Clone the original tensor "data_to_prefetch" so we can concat them into
    // one big tensor to be prefetched.
    std::vector<poplar::Tensor> cloned_tensors(data_to_prefetch);
    for (int i = 0; i < data_to_prefetch; ++i) {
      cloned_tensors[i] = graph.clone(tensor);
    }

    // Concatenate all the cloned tensors then reshape to make sure we are in
    // the shape [data_to_prefetch][original_shape].
    poplar::Tensor pegged_memory =
        poplar::concat(cloned_tensors).reshape(new_shape);

    // We need to have two counters as we have a master counter for the
    // predicate condition in the program::If which *cannot* be replicated (as
    // the control flow must be the same for all replicated graphs) and then a
    // seperate counter for the dynamic slice which is replicated so must have a
    // replicated counter.
    poplar::Tensor counter = master_graph.addVariable(
        poplar::UNSIGNED_INT, {}, poplar::VariableMappingMethod::LINEAR,
        GetDebugName(inst) + "/InfeedCounter_master/" +
            std::to_string(flat_tuple_index));
    master_graph.setInitialValue(counter, data_to_prefetch);

    poplar::Tensor replicated_counter = graph.addVariable(
        poplar::UNSIGNED_INT, {}, poplar::VariableMappingMethod::LINEAR,
        GetDebugName(inst) + "/InfeedCounter_replicated/" +
            std::to_string(flat_tuple_index));
    graph.setInitialValue(replicated_counter, 0);

    // The body for copying from host and zeroing the counter.
    poplar::program::Sequence true_body;

    // If we are using synthetic data, init pegged_memory with it otherwise host
    // copy. Either way we will have a tensor with some number of prefetched
    // batches and we will dynamic slice the actual batch from that. This is to
    // ensure that we can benchmark synthetic vs non-synthetic without chaging
    // the graph too much.
    TF_RETURN_IF_ERROR(init_synthetic_or_copy(
        true_body,
        XlaShapeFromPoplarShape(shape.element_type(), pegged_memory.shape()),
        pegged_memory));

    popops::zero(master_graph, counter, true_body,
                 GetDebugName(inst) + "/InfeedZeroCounter_master/" +
                     std::to_string(flat_tuple_index));

    popops::zero(graph, replicated_counter, true_body,
                 GetDebugName(inst) + "/InfeedZeroCounter_replciated/" +
                     std::to_string(flat_tuple_index));

    // The NOP body.
    poplar::program::Sequence false_body;

    // Check the counter doesn't equal
    poplar::Tensor predicate = popops::map(
        master_graph, pe::Equal(pe::_1, pe::Const(data_to_prefetch)), {counter},
        seq,
        GetDebugName(inst) + "/InfeedCheckCounter/" +
            std::to_string(flat_tuple_index));

    // The main body which contains the control flow for copy from host and
    // the dynamic slice.
    seq.add(poplar::program::If(predicate, true_body, false_body));

    // Create a big switch statement each with a static copy from one index of
    // the prefetched block.
    std::vector<std::pair<std::int32_t, poplar::program::Program>> cases;
    for (std::uint32_t i = 0; i < data_to_prefetch; ++i) {
      cases.push_back(std::make_pair<std::int32_t, poplar::program::Program>(
          static_cast<int32_t>(i),
          poplar::program::Copy(pegged_memory[i], tensor)));
    }

    seq.add(poplar::program::Switch(counter, cases));

    // Increment the counter by one.
    popops::mapInPlace(master_graph, pe::Add(pe::_1, pe::Const(1)), {counter},
                       seq,
                       GetDebugName(inst) + "/InfeedCounterUpdate_Master/" +
                           std::to_string(flat_tuple_index));

    popops::mapInPlace(graph, pe::Add(pe::_1, pe::Const(1)),
                       {replicated_counter}, seq,
                       GetDebugName(inst) + "/InfeedCounterUpdate_Replicated/" +
                           std::to_string(flat_tuple_index));
  } else {
    // Just an normal copy from host->tensor or init tensor with synthetic data.
    TF_RETURN_IF_ERROR(init_synthetic_or_copy(seq, shape, tensor));
  }
  return tensor;
}

bool DeferredAllocationVisitor::CanDeferAllocation(
    const HloInstruction* inst, const int64 flat_tuple_index) {
  auto deferred_allocations = resources_.annotations.deferred_allocations;
  return deferred_allocations[inst->parent()].contains(
      std::make_pair(inst, flat_tuple_index));
}

void DeferredAllocationVisitor::DeferAllocation(const HloInstruction* inst,
                                                const int64 flat_tuple_index) {
  auto& deferred_allocations = resources_.annotations.deferred_allocations;
  auto& tensor_allocation_map = resources_.annotations.tensor_allocation_map;

  // Get the source and target for the allocation.
  auto deferred_allocation_source = deferred_allocations[inst->parent()].at(
      std::make_pair(inst, flat_tuple_index));
  auto& tensor_target = tensor_allocation_map.at(deferred_allocation_source);
  // Add the path so that we don't try to access the tensor which has not yet
  // been allocated.
  instructions_in_deferred_allocation_paths.insert(
      tensor_target.deferred_allocations_path.begin(),
      tensor_target.deferred_allocations_path.end());
  deferred_allocation_sources.insert(deferred_allocation_source);
}

bool DeferredAllocationVisitor::IsInDeferredAllocationPath(
    const HloInstruction* inst, const int64 flat_tuple_index) {
  return instructions_in_deferred_allocation_paths.contains(
      std::make_pair(inst, flat_tuple_index));
}

bool DeferredAllocationVisitor::IsDeferredAllocation(
    const HloInstruction* inst, const int64 flat_tuple_index) {
  return deferred_allocation_sources.contains(
      std::make_pair(inst, flat_tuple_index));
}

}  // namespace poplarplugin
}  // namespace xla
