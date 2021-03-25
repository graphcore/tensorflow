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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/ctc_loss.h"

#include <poplar/DebugContext.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/CTCLoss.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class CTCLossOpBase : public PoplarOpDef {
 protected:
  virtual const std::string ClassName() const = 0;

  virtual std::pair<poplar::Tensor, poplar::Tensor> CalcLossAndGradient(
      poplar::Graph&, const poplar::Type&, const poplar::Tensor&,
      const poplar::Tensor&, const poplar::Tensor&, const poplar::Tensor&,
      poplar::program::Sequence&, const int64, const popnn::ctc::Plan&,
      const poplar::DebugInfo&) const = 0;

 public:
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    // Create the control program.
    poplar::program::Sequence seq({}, debug_info);
    const HloCTCLossInstructionBase* ctc_inst =
        Cast<HloCTCLossInstructionBase>(inst);

    // Get the inputs.
    TF_ASSIGN_OR_RETURN(poplar::Tensor data,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor labels,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor data_lengths,
                        FindInstructionInput(tensor_map, res, inst, 2, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor label_lengths,
                        FindInstructionInput(tensor_map, res, inst, 3, seq,
                                             {debug_info}, false));

    int64 blank_index = ctc_inst->blank_index();
    TF_ASSIGN_OR_RETURN(poplar::Type output_type,
                        PoplarDataType(ctc_inst->out_dtype()));

    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));

    poplar::Tensor loss, grad;
    std::tie(loss, grad) = CalcLossAndGradient(
        graph, output_type, data, labels.reinterpret(poplar::UNSIGNED_INT),
        data_lengths.reinterpret(poplar::UNSIGNED_INT),
        label_lengths.reinterpret(poplar::UNSIGNED_INT), seq, blank_index,
        *plan, debug_info);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, loss));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, grad));

    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    const HloInstruction* inst = tensor_target.tgt;
    const HloCTCLossInstructionBase* ctc_inst =
        Cast<HloCTCLossInstructionBase>(inst);
    const int64 input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));

    const Shape& allocation_shape = inst->operand(input_index)->shape();
    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(allocation_shape));

    auto data_shape = inst->operand(0)->shape();
    auto labels_shape = inst->operand(1)->shape();
    const int64 batch_size = ShapeUtil::GetDimension(data_shape, 1);

    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(poplar::Type in_dtype,
                            PoplarDataType(ctc_inst->in_dtype()));
        if (in_dtype != dtype) {
          return xla::FailedPrecondition(
              "dtype of the data input tensor (%s) does not match the "
              "specified "
              "input dtype (%s)",
              dtype.toString(), in_dtype.toString());
        }
        const int64 max_time = ShapeUtil::GetDimension(data_shape, 0);
        const int64 num_classes = ShapeUtil::GetDimension(data_shape, 2);
        return popnn::ctc::createDataInput(graph, dtype, batch_size, max_time,
                                           num_classes, *plan,
                                           {debug_info, "data"});
      }
      case 1: {
        const int64 max_label_length = ShapeUtil::GetDimension(labels_shape, 1);
        return popnn::ctc::createLabelsInput(graph, dtype, batch_size,
                                             max_label_length, *plan,
                                             {debug_info, "labels"});
      }
      default: {
        return FailedPrecondition(
            "Invalid allocation index %d for instruction ", input_index,
            inst->ToString());
      }
    }
  }
};

class CTCLossWithLogitsOp : public CTCLossOpBase {
  const std::string ClassName() const override { return "CTCLossWithLogitsOp"; }

  std::pair<poplar::Tensor, poplar::Tensor> CalcLossAndGradient(
      poplar::Graph& graph, const poplar::Type& out_type,
      const poplar::Tensor& logits, const poplar::Tensor& labels,
      const poplar::Tensor& data_lengths, const poplar::Tensor& label_lengths,
      poplar::program::Sequence& prog, const int64 blank_class,
      const popnn::ctc::Plan& plan,
      const poplar::DebugInfo& debug_info) const override {
    return popnn::ctc::calcLossAndGradientLogits(
        graph, out_type, logits, labels, data_lengths, label_lengths, prog,
        blank_class, plan, debug_info);
  }
};

REGISTER_POPLAR_OP(CTCLossWithLogits, CTCLossWithLogitsOp);

class CTCLossWithLogProbsOp : public CTCLossOpBase {
  const std::string ClassName() const override { return "CTCLossOp"; }

  std::pair<poplar::Tensor, poplar::Tensor> CalcLossAndGradient(
      poplar::Graph& graph, const poplar::Type& out_type,
      const poplar::Tensor& log_probs, const poplar::Tensor& labels,
      const poplar::Tensor& data_lengths, const poplar::Tensor& label_lengths,
      poplar::program::Sequence& prog, const int64 blank_class,
      const popnn::ctc::Plan& plan,
      const poplar::DebugInfo& debug_info) const override {
    return popnn::ctc::calcLossAndGradientLogProbabilities(
        graph, out_type, log_probs, labels, data_lengths, label_lengths, prog,
        blank_class, plan, debug_info);
  }
};

REGISTER_POPLAR_OP(CTCLossWithLogProbs, CTCLossWithLogProbsOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
