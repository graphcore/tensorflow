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
        // Create the output with the same mapping as the input.
        poplar::Tensor output =
            graph.clone(inputs[tuple_idx],
                        absl::StrCat(GetDebugName(inst), "/out/", tuple_idx),
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, tuple_idx, output));

        // Create a buffer of the given depth and the same mapping as the input.
        poplar::Tensor buffer =
            graph.clone(inputs[tuple_idx].expand({0}),
                        absl::StrCat(GetDebugName(inst), "/buffer/", tuple_idx),
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        // Copy the content of the buffer to the output.
        seq.add(poplar::program::Copy(buffer, output));

        // Copy the input into the buffer.
        seq.add(poplar::program::Copy(inputs[tuple_idx], buffer));
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
      // Create a buffer of the given depth and the same mapping as the input.
      std::vector<poplar::Tensor> cloned_tensors(fifo_inst->depth());
      for (int64 i = 0; i != fifo_inst->depth(); ++i) {
        cloned_tensors[i] =
            graph.clone(inputs[tuple_idx].expand({0}),
                        absl::StrCat(GetDebugName(inst), "/buffer/", tuple_idx),
                        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
      }
      poplar::Tensor buffer = poplar::concat(cloned_tensors);
      // Create the output with the same mapping as the input.
      poplar::Tensor output = popops::dynamicSlice(
          graph, buffer, counter.reshape({1}), {0}, {1}, seq,
          absl::StrCat(GetDebugName(inst), "/pop/", tuple_idx));
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, tuple_idx, output.squeeze({0})));

      popops::dynamicUpdate(
          graph, buffer, inputs[tuple_idx].expand({0}), counter.reshape({1}),
          {0}, {1}, seq, absl::StrCat(GetDebugName(inst), "/push/", tuple_idx));
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
