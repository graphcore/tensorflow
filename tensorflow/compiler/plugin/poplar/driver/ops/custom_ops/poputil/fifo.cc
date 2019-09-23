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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Util.hpp>

namespace xla {
namespace poplarplugin {
namespace {

bool is_powerof2(std::size_t v) { return v && ((v & (v - 1)) == 0); }

uint32 find_powerof2_mask(uint32 v) {
  assert(is_powerof2(v));

  return 0xFFFFFFFF % v;
}

std::map<std::size_t, poplar::Interval> GetInverseIntervalsMap(
    const std::vector<poplar::Interval>& intervals) {
  std::map<std::size_t, poplar::Interval> result;

  std::size_t offset = 0;
  for (int64 i = 0; i != intervals.size(); ++i) {
    const poplar::Interval& interval = intervals[i];
    result[interval.begin()] =
        poplar::Interval(offset, offset + interval.size());
    offset += interval.size();
  }
  return result;
}

class FifoOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    auto fifo_inst = Cast<HloFifoInstruction>(inst);
    poplar::program::Sequence seq;

    ArgVector inputs =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

    // A degenerate case where the fifo is just an identity op.
    if (fifo_inst->depth() < 1) {
      for (int64 tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor output = poputil::duplicate(
            graph, inputs[tuple_idx], seq,
            absl::StrCat(GetDebugName(inst), "/copy/", tuple_idx),
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
      }
      return seq;
    }
    // If the FIFO can only store a single buffer then skip the counter creation
    // and use copies.
    if (fifo_inst->depth() == 1) {
      for (int64 tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
        poplar::Tensor input = inputs[tuple_idx];
        // Create the output with the same mapping as the input.
        poplar::Tensor output = graph.clone(
            input, absl::StrCat(GetDebugName(inst), "/out/", tuple_idx),
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));
        // Flatten inputs and outputs.
        poplar::Tensor input_flat = input.flatten();
        poplar::Tensor output_flat = output.flatten();
        // Get the aliasing information.
        std::vector<std::vector<poplar::Interval>> flat_dealiased_intervals =
            graph.getSortedContiguousRegions(
                input_flat, {{0, input_flat.numElements()}}, true);
        // Dealias inputs and outputs.
        input_flat =
            poplar::concat(input_flat.slices(flat_dealiased_intervals));
        output_flat =
            poplar::concat(output_flat.slices(flat_dealiased_intervals));

        // Create a buffer of the given depth and the same mapping as the
        // input.
        poplar::Tensor buffer = graph.clone(
            input_flat, absl::StrCat(GetDebugName(inst), "/buffer/", tuple_idx),
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        // Copy the content of the buffer to the output.
        seq.add(poplar::program::Copy(buffer, output_flat));

        // Copy the input into the buffer.
        seq.add(poplar::program::Copy(input_flat, buffer));
      }
      return seq;
    }

    // Keep track of where in the buffer we are.
    auto counter = graph.addVariable(poplar::UNSIGNED_INT, {},
                                     GetDebugName(inst) + "/counter");
    graph.setTileMapping(counter, 0);
    graph.setInitialValue(counter, 0);
    res.zeroed_tensors.push_back(counter);

    for (int64 tuple_idx = 0; tuple_idx < inputs.size(); ++tuple_idx) {
      poplar::Tensor input = inputs[tuple_idx];
      // Flatten the input.
      poplar::Tensor input_flat = input.flatten();

      // Get the aliasing information.
      std::vector<std::size_t> interval_aliases;
      std::vector<std::vector<poplar::Interval>> sorted_contiguous_intervals =
          graph.getSortedContiguousRegions(input_flat,
                                           {{0, input_flat.numElements()}},
                                           false, &interval_aliases);
      // Get the intervals for the flat input.
      std::vector<poplar::Interval> contiguous_intervals =
          input_flat.getContiguousRegions();

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
      for (int64 i = 0; i != flat_intervals.size(); ++i) {
        poplar::Interval& interval = flat_intervals[i];
        if (interval.begin() == interval_aliases[i]) {
          flat_dealiased_intervals.push_back(interval);
        } else {
          interval_map[interval.begin()] = interval_map.at(interval_aliases[i]);
        }
      }

      // Dealias the input given the intervals.
      input_flat = poplar::concat(input_flat.slices(flat_dealiased_intervals));

      // Create a buffer of the given depth and the same mapping as the input.
      std::vector<poplar::Tensor> cloned_tensors(fifo_inst->depth());
      for (int64 i = 0; i != fifo_inst->depth(); ++i) {
        cloned_tensors[i] = graph.clone(
            input_flat.expand({0}),
            absl::StrCat(GetDebugName(inst), "/buffer/", tuple_idx));
      }
      poplar::Tensor buffer = poplar::concat(cloned_tensors);

      // Create the output with the same mapping as the input.
      poplar::Tensor output_flat =
          popops::dynamicSlice(
              graph, buffer, counter.reshape({1}), {0}, {1}, seq,
              absl::StrCat(GetDebugName(inst), "/pop/", tuple_idx))
              .squeeze({0});

      // Reconstruct the output tensor from the intervals.
      // Create an inverse map - note that we only map the intervals with no
      // aliasing.
      std::map<std::size_t, poplar::Interval> inverse_map =
          GetInverseIntervalsMap(flat_dealiased_intervals);

      std::vector<poplar::Tensor> output_regions(contiguous_intervals.size());
      for (int64 i = 0; i != contiguous_intervals.size(); ++i) {
        poplar::Interval& interval = contiguous_intervals[i];
        // First lookup the interval map to look through any aliasing.
        poplar::Interval& aliased_interval = interval_map.at(interval.begin());
        // Get the output interval for that unaliased interval.
        poplar::Interval& output_interval =
            inverse_map.at(aliased_interval.begin());
        output_regions[i] = output_flat.slice(output_interval);
      }

      // Concatenate the regions and reshape accordingly.
      poplar::Tensor output =
          poplar::concat(output_regions).reshape(input.shape());

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));

      // Update the buffer with the new value.
      popops::dynamicUpdate(
          graph, buffer, input_flat.expand({0}), counter.reshape({1}), {0}, {1},
          seq, absl::StrCat(GetDebugName(inst), "/push/", tuple_idx));
    }

    // A slightly faster path if the depth is a power of two
    // counter = (counter + 1) % depth
    if (is_powerof2(fifo_inst->depth())) {
      popops::mapInPlace(
          graph,
          popops::expr::BitwiseAnd(
              popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
              popops::expr::Const(find_powerof2_mask(fifo_inst->depth()))),
          {counter}, seq, GetDebugName(inst) + "/counter_inc_mask");
    } else {
      popops::mapInPlace(
          graph,
          popops::expr::Rem(
              popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
              popops::expr::Const(fifo_inst->depth())),
          {counter}, seq, GetDebugName(inst) + "/counter_inc_mod");
    }
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poputil, Fifo, FifoOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
