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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/histogram.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <popops/Cast.hpp>
#include <popops/GatherStatistics.hpp>

namespace xla {
namespace poplarplugin {
namespace {

/*
 * Histogram.
 */
class HistogramOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "HistogramOp");

    DriverProgramSequence seq(debug_info);

    auto hist_inst = Cast<HloHistogramInstruction>(inst);

    TF_ASSIGN_OR_RETURN(auto input, FindInstructionInput(tensor_map, res, inst,
                                                         0, seq, {debug_info}));

    TF_ASSIGN_OR_RETURN(
        auto levels,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info}));

    poplar::OptionFlags options;
    options.set("useFloatArithmetic", "true");
    auto output =
        popops::histogram(graph, input, levels, hist_inst->AbsoluteOfInput(),
                          seq, {debug_info, "HistogramPoplar"}, options);

    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, hist_inst, 0, DriverTensor(output)));
    return seq;
  }
};
REGISTER_POPLAR_OP(Histogram, HistogramOp);

/*
 * Histogram update.
 */
class HistogramUpdateOp : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "HistogramUpdateOp");

    DriverProgramSequence seq(debug_info);

    auto hist_upd_inst = Cast<HloHistogramUpdateInstruction>(inst);

    TF_ASSIGN_OR_RETURN(
        TensorVectors inplace_inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, debug_info));
    CHECK_EQ(inplace_inputs.size(), 1);
    CHECK_EQ(inplace_inputs[0].size(), 1);
    auto hist = inplace_inputs[0][0];

    TF_ASSIGN_OR_RETURN(auto input, FindInstructionInput(tensor_map, res, inst,
                                                         1, seq, {debug_info}));

    TF_ASSIGN_OR_RETURN(
        auto levels,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info}));

    popops::histogram(graph, input, hist, /*updateOutput=*/true, levels,
                      hist_upd_inst->AbsoluteOfInput(), seq,
                      {debug_info, "HistogramUpdatePoplar"});

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, hist_upd_inst, 0, hist));
    return seq;
  }
};
REGISTER_POPLAR_OP(HistogramUpdate, HistogramUpdateOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
