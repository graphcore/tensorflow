/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sequence_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/SequenceSlice.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class SequenceSliceOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "SequenceSliceOp");

    DriverProgramSequence seq(graph, debug_info);

    auto seq_slice_inst = Cast<HloSequenceSliceInstruction>(inst);

    TF_ASSIGN_OR_RETURN(auto input,
                        GetInputTensor(tensor_map, res, inst, seq, debug_info));

    TF_ASSIGN_OR_RETURN(auto num_elems, GetNumElemsTensor(tensor_map, res, inst,
                                                          seq, debug_info));
    auto num_elems_ui =
        popops::cast(graph, num_elems, poplar::UNSIGNED_INT, seq, debug_info);

    TF_ASSIGN_OR_RETURN(
        auto src_offsets,
        GetSrcOffsetsTensor(tensor_map, res, inst, seq, debug_info));
    auto src_offsets_ui =
        popops::cast(graph, src_offsets, poplar::UNSIGNED_INT, seq, debug_info);

    TF_ASSIGN_OR_RETURN(
        auto dst_offsets,
        GetDstOffsetsTensor(tensor_map, res, inst, seq, debug_info));
    auto dst_offsets_ui =
        popops::cast(graph, dst_offsets, poplar::UNSIGNED_INT, seq, debug_info);

    TF_ASSIGN_OR_RETURN(auto output, GetOutputTensor(graph, tensor_map, res,
                                                     inst, seq, debug_info));

    popops::sequenceSlice(graph, input, output, num_elems_ui, src_offsets_ui,
                          dst_offsets_ui, seq_slice_inst->ZeroUnused(), seq,
                          {debug_info, "SequenceSlicePoplar"});

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }

 protected:
  virtual StatusOr<DriverTensor> GetOutputTensor(
      DriverGraph& graph, TensorMap& tensor_map, CompilerResources& res,
      const HloInstruction* inst, DriverProgramSequence& seq,
      PoplarOpDefDebugInfo& debug_info) {
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, debug_info));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    return inputs[0][0];
  }

  virtual StatusOr<poplar::Tensor> GetInputTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) {
    return FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info});
  }

  virtual StatusOr<poplar::Tensor> GetNumElemsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) {
    return FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info});
  }

  virtual StatusOr<poplar::Tensor> GetSrcOffsetsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) {
    return FindInstructionInput(tensor_map, res, inst, 3, seq, {debug_info});
  }

  virtual StatusOr<poplar::Tensor> GetDstOffsetsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) {
    return FindInstructionInput(tensor_map, res, inst, 4, seq, {debug_info});
  }
};
REGISTER_POPLAR_OP(SequenceSlice, SequenceSliceOp);

class SequenceSliceUnpackOp : public SequenceSliceOp {
 protected:
  StatusOr<DriverTensor> GetOutputTensor(
      DriverGraph& graph, TensorMap& tensor_map, CompilerResources& res,
      const HloInstruction* inst, DriverProgramSequence& seq,
      PoplarOpDefDebugInfo& debug_info) override {
    return AddTensor(graph, TensorLocation{inst, 0}, inst->shape(), res,
                     tensor_map, {debug_info, "output"});
  }

  StatusOr<poplar::Tensor> GetInputTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) override {
    return FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info});
  }

  StatusOr<poplar::Tensor> GetNumElemsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) override {
    return FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info});
  }

  StatusOr<poplar::Tensor> GetSrcOffsetsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) override {
    return FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info});
  }

  StatusOr<poplar::Tensor> GetDstOffsetsTensor(
      TensorMap& tensor_map, CompilerResources& res, const HloInstruction* inst,
      DriverProgramSequence& seq, PoplarOpDefDebugInfo& debug_info) override {
    return FindInstructionInput(tensor_map, res, inst, 3, seq, {debug_info});
  }
};
REGISTER_POPLAR_OP(SequenceSliceUnpack, SequenceSliceUnpackOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
