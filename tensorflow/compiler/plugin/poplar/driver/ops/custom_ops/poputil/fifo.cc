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

void IncreaseCounter(poplar::Graph& graph, poplar::Tensor& counter, int64 depth,
                     poplar::program::Sequence& seq,
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

std::map<std::size_t, poplar::Interval> GetInverseIntervalsMap(
    const std::vector<poplar::Interval>& intervals) {
  std::map<std::size_t, poplar::Interval> result;

  std::size_t offset = 0;
  for (size_t i = 0; i != intervals.size(); ++i) {
    const poplar::Interval& interval = intervals[i];
    result[interval.begin()] =
        poplar::Interval(offset, offset + interval.size());
    offset += interval.size();
  }
  return result;
}

// Helper struct for storing information about tensor aliasing when a tensor is
// passed between instructions.
struct TensorAliasingInformation {
  // Indicates whether a tensor has aliasing, if not, other fields are not
  // populated.
  bool has_aliasing;

  // The interval map will store the interval beginning to the interval it
  // aliases.
  std::map<std::size_t, poplar::Interval> interval_map;

  // Inverse map for reconstructing a tensor with aliasing from intervals.
  std::map<std::size_t, poplar::Interval> inverse_map;
};

std::pair<poplar::Tensor, TensorAliasingInformation>
GetAliasingInformationAndDealiesedTensor(poplar::Graph& graph,
                                         const poplar::Tensor& tensor) {
  CHECK_EQ(tensor.rank(), 1);

  TensorAliasingInformation info;

  if (!tensor.containsAliases()) {
    info.has_aliasing = false;
    return std::make_pair(tensor, info);
  }

  // Get the aliasing information.
  std::vector<std::size_t> interval_aliases;
  std::vector<std::vector<poplar::Interval>> sorted_contiguous_intervals =
      graph.getSortedContiguousRegions(tensor, {{0, tensor.numElements()}},
                                       false, &interval_aliases);

  // Flatten the sorted contiguous intervals so that we can easily map it
  // to the aliasing information.
  std::vector<poplar::Interval> flat_intervals;
  // The interval map will store the interval beginning to the interval it
  // aliases.
  std::map<std::size_t, poplar::Interval> interval_map;
  for (auto& intervals : sorted_contiguous_intervals) {
    flat_intervals.insert(flat_intervals.end(), intervals.begin(),
                          intervals.end());
    absl::c_transform(intervals,
                      std::inserter(interval_map, std::begin(interval_map)),
                      [](const poplar::Interval& interval) {
                        return std::make_pair(interval.begin(), interval);
                      });
  }
  CHECK_EQ(interval_aliases.size(), flat_intervals.size());

  // Update the aliasing map and get all the intervals with no aliasing.
  std::vector<poplar::Interval> flat_dealiased_intervals;
  for (size_t i = 0; i != flat_intervals.size(); ++i) {
    poplar::Interval& interval = flat_intervals[i];
    if (interval.begin() == interval_aliases[i]) {
      flat_dealiased_intervals.push_back(interval);
    } else {
      interval_map[interval.begin()] = interval_map.at(interval_aliases[i]);
    }
  }

  // Dealias the input given the intervals.
  poplar::Tensor dealised_tensor =
      poplar::concat(tensor.slices(flat_dealiased_intervals));

  info.has_aliasing = true;
  info.interval_map = interval_map;
  info.inverse_map = GetInverseIntervalsMap(flat_dealiased_intervals);
  return std::make_pair(dealised_tensor, info);
}

poplar::Tensor AddAliasing(const poplar::Tensor& tensor,
                           const TensorAliasingInformation& info) {
  if (!info.has_aliasing) {
    return tensor;
  }

  std::vector<poplar::Tensor> output_regions;
  output_regions.reserve(info.interval_map.size());

  // Since the std::map is sorted by the key which is the beginning of the
  // interval, we can reconstruct the original tensor by iterating over it.
  for (const auto& entry : info.interval_map) {
    const poplar::Interval& aliased_interval = entry.second;
    // Get the output interval for that unaliased interval.
    const poplar::Interval& output_interval =
        info.inverse_map.at(aliased_interval.begin());
    output_regions.push_back(tensor.slice(output_interval));
  }

  // Concatenate the regions and reshape accordingly.
  return poplar::concat(output_regions);
}

class FifoOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
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
        // Flatten inputs and outputs.
        poplar::Tensor input_flat = input.flatten();
        poplar::Tensor output_flat = output.flatten();

        if (input_flat.containsAliases()) {
          // Get the aliasing information.
          std::vector<std::vector<poplar::Interval>> flat_dealiased_intervals =
              graph.getSortedContiguousRegions(
                  input_flat, {{0, input_flat.numElements()}}, true);
          // Dealias inputs and outputs.
          input_flat =
              poplar::concat(input_flat.slices(flat_dealiased_intervals));
          output_flat =
              poplar::concat(output_flat.slices(flat_dealiased_intervals));
        }

        if (fifo_offload) {
          poplar::RemoteBuffer buffer = graph.addRemoteBuffer(
              absl::StrCat(debug_name, "/buffer/", tuple_idx),
              input.elementType(), input_flat.numElements(), 1, true);

          // Copy the content of the buffer to the output.
          seq.add(poplar::program::Copy(buffer, output_flat, {debug_info}));

          // Copy the input into the buffer.
          seq.add(poplar::program::Copy(input_flat, buffer, {debug_info}));
        } else {
          // Create a buffer for swapping the values.
          poplar::Tensor buffer = graph.clone(
              input_flat, {debug_info, absl::StrCat("buffer/", tuple_idx)},
              poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
          TF_RETURN_IF_ERROR(
              AddWriteUndefToFIFOBuffer(inst, buffer, res, {debug_info}));

          // Copy the content of the buffer to the output.
          seq.add(
              poplar::program::Copy(buffer, output_flat, false, {debug_info}));

          // Copy the input into the buffer.
          seq.add(
              poplar::program::Copy(input_flat, buffer, false, {debug_info}));
        }
      }
      return seq;
    }

    // Keep track of where in the buffer we are.
    auto counter =
        graph.addVariable(poplar::UNSIGNED_INT, {}, {debug_info, "counter"});
    graph.setTileMapping(counter, 0);
    AddZeroTensorToPreamble(res, counter, {debug_info});

    for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
      poplar::Tensor input = inputs[tuple_idx];
      // Flatten the input.
      poplar::Tensor input_flat = input.flatten();

      // Remove any aliasing from the tensor to make sure the FIFO is as small
      // as possible.
      TensorAliasingInformation info;
      std::tie(input_flat, info) =
          GetAliasingInformationAndDealiesedTensor(graph, input_flat);

      poplar::Tensor output_flat;
      if (fifo_offload) {
        // Keep the layout of the input for the output.
        output_flat = graph.clone(
            input_flat, {debug_info, absl::StrCat("out/", tuple_idx)});

        // Create a remote buffer of the given depth.
        poplar::RemoteBuffer buffer = graph.addRemoteBuffer(
            absl::StrCat(debug_name, "/buffer/", tuple_idx),
            input.elementType(), input_flat.numElements(), fifo_depth, true);

        // Copy the content of the buffer to the output.
        seq.add(poplar::program::Copy(buffer, output_flat, counter.reshape({1}),
                                      {debug_info}));

        // Copy the input into the buffer.
        seq.add(poplar::program::Copy(input_flat, buffer, counter.reshape({1}),
                                      {debug_info}));
      } else {
        // Create a buffer of the given depth and the same mapping as the input.
        poplar::Tensor buffer = popops::createSliceableTensorFromSlice(
            graph, input_flat.expand({0}), {0}, {fifo_depth},
            {debug_info, absl::StrCat("buffer/", tuple_idx)});
        TF_RETURN_IF_ERROR(
            AddWriteUndefToFIFOBuffer(inst, buffer, res, {debug_info}));

        // Create the output with the same mapping as the input.
        output_flat = popops::dynamicSlice(
                          graph, buffer, counter.reshape({1}), {0}, {1}, seq,
                          {debug_info, absl::StrCat("pop/", tuple_idx)})
                          .squeeze({0});

        // Update the buffer with the new value.
        popops::dynamicUpdate(graph, buffer, input_flat.expand({0}),
                              counter.reshape({1}), {0}, {1}, seq,
                              {debug_info, absl::StrCat("push/", tuple_idx)});
      }

      // Add the aliasing information back in.
      output_flat = AddAliasing(output_flat, info);
      poplar::Tensor output = output_flat.reshape(input.shape());

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
    }

    IncreaseCounter(graph, counter, fifo_depth, seq, {debug_info});
    return seq;
  }
};
REGISTER_POPLAR_OP(Fifo, FifoOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
