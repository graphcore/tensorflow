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

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

bool IsFifoInPipeline(const HloInstruction* inst, CallGraph* call_graph) {
  auto call_sites = call_graph->GetNode(inst->parent()).caller_callsites();
  return call_sites.size() == 1 && IsPipelineOp(call_sites[0].instruction());
}

Status AddWriteUndefToFIFOBuffer(const HloInstruction* inst,
                                 const poplar::Tensor& buffer,
                                 CompilerResources& res) {
  if (IsFifoInPipeline(inst, res.module_call_graph.get())) {
    // We need to write undef the FIFO buffer otherwise it will be marked as
    // always live in Poplar as we never write to it fully in the pipeline.
    if (res.pipelining_write_undef_sequences.empty()) {
      return FailedPrecondition("Cannot WriteUndef a FIFO buffer.");
    }
    poplar::program::Sequence seq;
    seq.add(poplar::program::WriteUndef(buffer));
    res.pipelining_write_undef_sequences.top().push_back(seq);
  }
  return Status::OK();
}

void IncreaseCounter(poplar::Graph& graph, poplar::Tensor& counter, int64 depth,
                     poplar::program::Sequence& seq,
                     const std::string& debug_name) {
  // A slightly faster path if the depth is a power of two
  // counter = (counter + 1) % depth
  if (is_powerof2(depth)) {
    popops::mapInPlace(
        graph,
        popops::expr::BitwiseAnd(
            popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
            popops::expr::Const(find_powerof2_mask(depth))),
        {counter}, seq, debug_name + "/CounterIncreaseMask");
  } else {
    popops::mapInPlace(
        graph,
        popops::expr::Rem(
            popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
            popops::expr::Const(depth)),
        {counter}, seq, debug_name + "/CounterIncreaseMod");
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

  // Intervals for the given tensor.
  std::vector<poplar::Interval> contiguous_intervals;

  // The interval map will store the interval begining to the interval it
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
  // Get the intervals for the flat input.
  std::vector<poplar::Interval> contiguous_intervals =
      tensor.getContiguousRegions();

  // Flatten the sorted contiguous intervals so that we can easily map it
  // to the aliasing information.
  std::vector<poplar::Interval> flat_intervals;
  // The interval map will store the interval begining to the interval it
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
  CHECK_EQ(contiguous_intervals.size(), flat_intervals.size());

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
  info.contiguous_intervals = contiguous_intervals;
  info.interval_map = interval_map;
  info.inverse_map = GetInverseIntervalsMap(flat_dealiased_intervals);
  return std::make_pair(dealised_tensor, info);
}

poplar::Tensor AddAliasing(const poplar::Tensor& tensor,
                           const TensorAliasingInformation& info) {
  if (!info.has_aliasing) {
    return tensor;
  }

  std::vector<poplar::Tensor> output_regions(info.contiguous_intervals.size());
  for (size_t i = 0; i != info.contiguous_intervals.size(); ++i) {
    const poplar::Interval& interval = info.contiguous_intervals.at(i);
    // First lookup the interval map to look through any aliasing.
    const poplar::Interval& aliased_interval =
        info.interval_map.at(interval.begin());
    // Get the output interval for that unaliased interval.
    const poplar::Interval& output_interval =
        info.inverse_map.at(aliased_interval.begin());
    output_regions[i] = tensor.slice(output_interval);
  }

  // Concatenate the regions and reshape accordingly.
  return poplar::concat(output_regions);
}

class FifoOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    auto fifo_inst = Cast<HloFifoInstruction>(inst);
    poplar::program::Sequence seq;
    const std::string debug_name = GetDebugName(inst);

    TensorVector inputs =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

    // A degenerate case where the fifo is just an identity op.
    if (fifo_inst->depth() < 1) {
      for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor output = poputil::duplicate(
            graph, inputs[tuple_idx], seq,
            absl::StrCat(debug_name, "/copy/", tuple_idx),
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
      }
      return seq;
    }
    // If the FIFO can only store a single buffer then skip the counter creation
    // and use copies.
    if (fifo_inst->depth() == 1) {
      for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor input = inputs[tuple_idx];
        // Create the output with the same mapping as the input.
        poplar::Tensor output =
            graph.clone(input, absl::StrCat(debug_name, "/out/", tuple_idx),
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

        // Create a buffer of the given depth and the same mapping as the
        // input.
        poplar::Tensor buffer = graph.clone(
            input_flat, absl::StrCat(debug_name, "/buffer/", tuple_idx),
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_RETURN_IF_ERROR(AddWriteUndefToFIFOBuffer(inst, buffer, res));

        // Copy the content of the buffer to the output.
        seq.add(poplar::program::Copy(buffer, output_flat));

        // Copy the input into the buffer.
        seq.add(poplar::program::Copy(input_flat, buffer));
      }
      return seq;
    }

    // Keep track of where in the buffer we are.
    auto counter =
        graph.addVariable(poplar::UNSIGNED_INT, {}, debug_name + "/counter");
    graph.setTileMapping(counter, 0);
    AddZeroTensorToPreamble(res, counter);

    for (size_t tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
      poplar::Tensor input = inputs[tuple_idx];
      // Flatten the input.
      poplar::Tensor input_flat = input.flatten();

      // Remove any aliasing from the tensor to make sure the FIFO is as small
      // as possible.
      TensorAliasingInformation info;
      std::tie(input_flat, info) =
          GetAliasingInformationAndDealiesedTensor(graph, input_flat);

      // Create a buffer of the given depth and the same mapping as the input.
      const size_t fifo_depth = fifo_inst->depth();
      poplar::Tensor buffer = popops::createSliceableTensorFromSlice(
          graph, input_flat.expand({0}), {0}, {fifo_depth},
          absl::StrCat(debug_name, "/buffer/", tuple_idx));
      TF_RETURN_IF_ERROR(AddWriteUndefToFIFOBuffer(inst, buffer, res));

      // Create the output with the same mapping as the input.
      poplar::Tensor output_flat =
          popops::dynamicSlice(graph, buffer, counter.reshape({1}), {0}, {1},
                               seq,
                               absl::StrCat(debug_name, "/pop/", tuple_idx))
              .squeeze({0});

      // Update the buffer with the new value.
      popops::dynamicUpdate(graph, buffer, input_flat.expand({0}),
                            counter.reshape({1}), {0}, {1}, seq,
                            absl::StrCat(debug_name, "/push/", tuple_idx));

      // Add the aliasing information back in.
      output_flat = AddAliasing(output_flat, info);
      poplar::Tensor output = output_flat.reshape(input.shape());

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
    }

    IncreaseCounter(graph, counter, fifo_inst->depth(), seq, debug_name);
    return seq;
  }
};
REGISTER_POPLAR_OP(Fifo, FifoOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
