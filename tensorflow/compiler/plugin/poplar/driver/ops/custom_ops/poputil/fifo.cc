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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"

#include <poplar/DebugContext.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

bool is_powerof2(std::size_t v) { return v && ((v & (v - 1)) == 0); }

uint32 find_powerof2_mask(uint32 v) {
  assert(is_powerof2(v));

  return 0xFFFFFFFF % v;
}

Status AddWriteUndefToFIFOBuffer(
    const HloInstruction* inst, const poplar::Tensor& buffer,
    CompilerResources& res, const poplar::DebugNameAndId& debug_name_and_id) {
  if (IsInPipeline(inst, res)) {
    // We need to write undef the FIFO buffer otherwise it will be marked as
    // always live in Poplar as we never write to it fully in the pipeline.
    if (res.pipelining_write_undef_sequences.empty()) {
      return FailedPrecondition("Cannot WriteUndef a FIFO buffer.");
    }
    poplar::program::Sequence seq({}, debug_name_and_id);
    seq.add(poplar::program::WriteUndef(buffer, {debug_name_and_id}));
    res.pipelining_write_undef_sequences.top().push_back(seq);
  }
  return Status::OK();
}

void IncreaseCounter(poplar::Graph& graph, const poplar::Tensor& counter,
                     int64 depth, poplar::program::Sequence& seq,
                     const poplar::DebugNameAndId& debug_name_and_id) {
  // A slightly faster path if the depth is a power of two
  // counter = (counter + 1) % depth
  if (is_powerof2(depth)) {
    popops::mapInPlace(
        graph,
        popops::expr::BitwiseAnd(
            popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
            popops::expr::Const(find_powerof2_mask(depth))),
        {counter}, seq, {debug_name_and_id, "CounterIncreaseMask"});
  } else {
    popops::mapInPlace(
        graph,
        popops::expr::Rem(
            popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
            popops::expr::Const(depth)),
        {counter}, seq, {debug_name_and_id, "CounterIncreaseMod"});
  }
}

void IncreaseCounters(std::vector<poplar::Graph*>& graphs,
                      const std::vector<poplar::Tensor>& counters, int64 depth,
                      poplar::program::Sequence& seq,
                      const poplar::DebugNameAndId& debug_name_and_id) {
  for (int i = 0; i < graphs.size(); ++i) {
    if (counters[i].numElements() > 0) {
      IncreaseCounter(*graphs[i], counters[i], depth, seq, debug_name_and_id);
    }
  }
}

poplar::Tensor DealiasTensor(poplar::Graph& graph, poplar::Tensor tensor) {
  tensor = tensor.flatten();
  graph.reorderToSimplify(&tensor, {});
  auto intervals = graph.getSortedContiguousRegions(
      tensor, {{0, tensor.numElements()}}, true);
  auto slices = tensor.slices(intervals);

  if (slices.empty()) {
    // Can't concat empty intervals, so return a zero element tensor.
    const size_t size[1] = {0};
    return graph.addVariable(tensor.elementType(), size);
  } else {
    return poplar::concat(slices);
  }
}

std::vector<poplar::Tensor> PartitionTensor(
    const std::vector<poplar::Graph*>& graphs, poplar::Tensor tensor) {
  std::vector<poplar::Tensor> result;

  for (auto& graph : graphs) {
    auto intervals =
        graph->getTileMapping(tensor, true, /* allowExternal = */ true);
    auto slices = tensor.flatten().slices(intervals);
    if (slices.empty()) {
      // Can't concat empty intervals, so push back a zero element tensor.
      const size_t size[1] = {0};
      result.push_back(graph->addVariable(tensor.elementType(), size));
    } else {
      result.push_back(poplar::concat(slices));
    }
  }

  return result;
}

std::vector<poplar::Tensor> CreateCounters(
    const std::vector<poplar::Graph*>& graphs, const HloInstruction* inst,
    CompilerResources& res, const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "CreateCounters");
  std::vector<poplar::Tensor> counters;

  for (auto vgraph : graphs) {
    auto counter =
        vgraph->addVariable(poplar::UNSIGNED_INT, {}, {debug_info, "counter"});
    vgraph->setTileMapping(counter, 0);
    AddZeroTensorToPreamble(res, counter, {debug_info});

    counters.emplace_back(std::move(counter));
  }

  return counters;
}

std::vector<poplar::Graph*> GetGraphs(poplar::Graph& graph,
                                      const HloInstruction* inst,
                                      CompilerResources& res) {
  if (inst->has_sharding() &&
      inst->sharding().GetUniqueDevice() == Devices::All) {
    std::vector<poplar::Graph*> result;

    for (auto& graph : res.shard_compute_graphs) {
      // TODO(T58509) - remove cast
      result.push_back(&((poplar::Graph&)graph));  // NOLINT
    }

    return result;
  }

  return {&graph};
}

class FifoOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "FifoOp");
    auto fifo_inst = Cast<HloFifoInstruction>(inst);
    const size_t fifo_depth = fifo_inst->depth();
    const bool fifo_offload = fifo_inst->offload();

    poplar::program::Sequence seq({}, debug_info);
    const std::string debug_name = GetDebugName(inst);

    // Opaque inputs are compile-time constants, so pass through FIFOs.
    if (inst->operand(0)->shape().IsOpaque()) {
      auto output =
          FindInstructionInputs(tensor_map, res, inst, 0, seq, {debug_info});
      TF_CHECK_OK(AddOutputOpaque(tensor_map, inst, 0, output[0].AsOpaque()));

      return seq;
    }

    TF_ASSIGN_OR_RETURN(TensorVector inputs,
                        FindInstructionInputTensors(tensor_map, res, inst, 0,
                                                    seq, {debug_info}, false));

    // A degenerate case where the fifo is just an identity op.
    if (fifo_depth < 1) {
      for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor output = poputil::duplicate(
            graph, inputs[tuple_idx], seq,
            {debug_info, absl::StrCat("copy/", tuple_idx)},
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
      }
      return seq;
    }

    auto graphs = GetGraphs(graph, inst, res);

    // If the FIFO can only store a single buffer then skip the counter creation
    // and use copies.
    if (fifo_depth == 1) {
      for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor input = inputs[tuple_idx];
        // Create the output with the same mapping as the input.
        poplar::Tensor output =
            graph.clone(input, {debug_info, absl::StrCat("out/", tuple_idx)},
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));

        // Skip empty inputs.
        if (input.numElements() == 0) {
          continue;
        }

        // Flatten inputs and outputs.
        std::vector<poplar::Tensor> input_flat =
            PartitionTensor(graphs, DealiasTensor(graph, input.flatten()));
        std::vector<poplar::Tensor> output_flat =
            PartitionTensor(graphs, DealiasTensor(graph, output.flatten()));

        for (std::size_t partition_idx = 0; partition_idx < graphs.size();
             ++partition_idx) {
          // Skip empty partitions.
          if (input_flat[partition_idx].numElements() == 0) {
            continue;
          }

          if (fifo_offload) {
            poplar::RemoteBuffer buffer =
                graphs[partition_idx]->addRemoteBuffer(
                    absl::StrCat(debug_name, "/buffer/", tuple_idx, "/",
                                 partition_idx),
                    input.elementType(),
                    input_flat[partition_idx].numElements(), 1, true);

            // Copy the content of the buffer to the output.
            seq.add(poplar::program::Copy(buffer, output_flat[partition_idx],
                                          {debug_info}));

            // Copy the input into the buffer.
            seq.add(poplar::program::Copy(input_flat[partition_idx], buffer,
                                          {debug_info}));
          } else {
            // Create a buffer for swapping the values.
            poplar::Tensor buffer = graphs[partition_idx]->clone(
                input_flat[partition_idx],
                {debug_info,
                 absl::StrCat("buffer/", tuple_idx, "/", partition_idx)},
                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
            TF_RETURN_IF_ERROR(
                AddWriteUndefToFIFOBuffer(inst, buffer, res, {debug_info}));

            // Copy the content of the buffer to the output.
            seq.add(poplar::program::Copy(buffer, output_flat[partition_idx],
                                          false, {debug_info}));

            // Copy the input into the buffer.
            seq.add(poplar::program::Copy(input_flat[partition_idx], buffer,
                                          false, {debug_info}));
          }
        }
      }
      return seq;
    }

    // Keep track of where in the buffer we are.
    auto counters = CreateCounters(graphs, inst, res, debug_context);

    for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
      poplar::Tensor input_clone = graph.clone(
          inputs[tuple_idx], {debug_info, absl::StrCat("out/", tuple_idx)},
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, input_clone));

      // Skip empty inputs.
      if (input_clone.numElements() == 0) {
        continue;
      }

      std::vector<poplar::Tensor> input_partitions =
          PartitionTensor(graphs, DealiasTensor(graph, inputs[tuple_idx]));
      std::vector<poplar::Tensor> output_partitions =
          PartitionTensor(graphs, DealiasTensor(graph, input_clone));

      for (size_t partition_idx = 0u; partition_idx < input_partitions.size();
           ++partition_idx) {
        poplar::Tensor input = input_partitions[partition_idx];
        poplar::Tensor output = output_partitions[partition_idx];
        // Flatten the input.
        poplar::Tensor input_flat = input.flatten();
        poplar::Tensor output_flat = output.flatten();

        // Skip empty partitions.
        if (input_flat.numElements() == 0) {
          continue;
        }

        if (fifo_offload) {
          // Create a remote buffer of the given depth.
          poplar::RemoteBuffer buffer = graphs[partition_idx]->addRemoteBuffer(
              absl::StrCat(debug_name, "/buffer/", tuple_idx, "/",
                           partition_idx),
              input.elementType(), input_flat.numElements(), fifo_depth, true);

          // Copy the content of the buffer to the output.
          seq.add(poplar::program::Copy(buffer, output_flat,
                                        counters[partition_idx].reshape({1}),
                                        {debug_info}));

          // Copy the input into the buffer.
          seq.add(poplar::program::Copy(input_flat, buffer,
                                        counters[partition_idx].reshape({1}),
                                        {debug_info}));
        } else {
          // Create a buffer of the given depth and the same mapping as the
          // input.
          poplar::Tensor buffer = popops::createSliceableTensorFromSlice(
              *graphs[partition_idx], input_flat.expand({0}), {0}, {fifo_depth},
              {debug_info,
               absl::StrCat("buffer/", tuple_idx, "/", partition_idx)});
          TF_RETURN_IF_ERROR(
              AddWriteUndefToFIFOBuffer(inst, buffer, res, {debug_info}));

          // Slice the element into the output.
          popops::dynamicSliceWithOutput(
              *graphs[partition_idx], output_flat.expand({0}), buffer,
              counters[partition_idx].reshape({1}), {0}, {1}, seq,
              {debug_info,
               absl::StrCat("pop/", tuple_idx, "/", partition_idx)});

          // Update the buffer with the new value.
          popops::dynamicUpdate(
              *graphs[partition_idx], buffer, input_flat.expand({0}),
              counters[partition_idx].reshape({1}), {0}, {1}, seq,
              {debug_info,
               absl::StrCat("push/", tuple_idx, "/", partition_idx)});
        }
      }
    }

    IncreaseCounters(graphs, counters, fifo_depth, seq, {debug_info});
    return seq;
  }
};
REGISTER_POPLAR_OP(Fifo, FifoOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
