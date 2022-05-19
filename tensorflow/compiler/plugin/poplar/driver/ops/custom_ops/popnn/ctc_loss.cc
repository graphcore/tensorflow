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
#include <popnn/CTCInference.hpp>
#include <popnn/CTCLoss.hpp>
#include <popops/Cast.hpp>

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
      DriverProgramSequence&, const int64_t, const popnn::ctc::Plan&,
      const poplar::DebugInfo&) const = 0;

 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    // Create the control program.
    DriverProgramSequence seq(graph, debug_info);
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

    int64_t blank_index = ctc_inst->blank_index();
    TF_ASSIGN_OR_RETURN(poplar::Type output_type,
                        PoplarDataType(ctc_inst->out_dtype()));

    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));

    poplar::Tensor loss, grad;
    std::tie(loss, grad) = CalcLossAndGradient(
        graph, output_type, data, labels.reinterpret(poplar::UNSIGNED_INT),
        data_lengths.reinterpret(poplar::UNSIGNED_INT),
        label_lengths.reinterpret(poplar::UNSIGNED_INT), seq, blank_index,
        *plan, debug_info);

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(loss, graph)));
    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 1, DriverTensor(grad, graph)));

    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    const HloInstruction* inst = tensor_target.tgt;
    const HloCTCLossInstructionBase* ctc_inst =
        Cast<HloCTCLossInstructionBase>(inst);
    const int64_t input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));

    const Shape& allocation_shape = inst->operand(input_index)->shape();
    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(allocation_shape));

    auto data_shape = inst->operand(0)->shape();
    auto labels_shape = inst->operand(1)->shape();
    const int64_t batch_size = ShapeUtil::GetDimension(data_shape, 1);

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
        const int64_t max_time = ShapeUtil::GetDimension(data_shape, 0);
        const int64_t num_classes = ShapeUtil::GetDimension(data_shape, 2);
        return popnn::ctc::createDataInput(graph, dtype, batch_size, max_time,
                                           num_classes, *plan,
                                           {debug_info, "data"});
      }
      case 1: {
        const int64_t max_label_length =
            ShapeUtil::GetDimension(labels_shape, 1);
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
      DriverProgramSequence& prog, const int64_t blank_class,
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
      DriverProgramSequence& prog, const int64_t blank_class,
      const popnn::ctc::Plan& plan,
      const poplar::DebugInfo& debug_info) const override {
    return popnn::ctc::calcLossAndGradientLogProbabilities(
        graph, out_type, log_probs, labels, data_lengths, label_lengths, prog,
        blank_class, plan, debug_info);
  }
};

REGISTER_POPLAR_OP(CTCLossWithLogProbs, CTCLossWithLogProbsOp);

class CTCBeamSearchOpBase : public PoplarOpDef {
 public:
  struct BeamSearchReturns {
    poplar::Tensor label_probabilities;
    poplar::Tensor label_lengths;
    poplar::Tensor decoded_labels;
  };

 private:
  virtual const std::string ClassName() const = 0;

  virtual BeamSearchReturns PerformBeamSearch(
      poplar::Graph& graph, const poplar::Tensor& data,
      const poplar::Tensor& dataLengths, DriverProgramSequence& prog,
      int64_t blankClass, int64_t beamwidth, int64_t topPaths,
      const popnn::ctc::Plan& plan,
      const poplar::DebugNameAndId& debug_name_and_id) const = 0;

 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    DriverProgramSequence seq(graph, debug_info);
    const auto* ctc_inst = Cast<HloCTCInferenceInstructionBase>(inst);

    // Retreive intputs, attributes and outputs from instruction
    TF_ASSIGN_OR_RETURN(poplar::Tensor data,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info, "data"}, false));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor data_lengths,
        FindInstructionInput(tensor_map, res, inst, 1, seq,
                             {debug_info, "data_lengths"}, false));

    TF_ASSIGN_OR_RETURN(poplar::Type in_type,
                        PoplarDataType(ctc_inst->in_dtype()));

    int64_t blank_index = ctc_inst->blank_index();
    int64_t beam_width = ctc_inst->beam_width();
    int64_t top_paths = ctc_inst->top_paths();

    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));

    auto outputs =
        PerformBeamSearch(graph, data, data_lengths, seq, blank_index,
                          beam_width, top_paths, *plan, debug_info);

    TF_CHECK_OK(AddOutputTensor(
        tensor_map, inst, 0, DriverTensor(outputs.label_probabilities, graph)));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1,
                                DriverTensor(outputs.label_lengths, graph)));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2,
                                DriverTensor(outputs.decoded_labels, graph)));
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, ClassName());
    const HloInstruction* inst = tensor_target.tgt;
    const HloCTCInferenceInstructionBase* ctc_inst =
        Cast<HloCTCInferenceInstructionBase>(inst);
    const int64_t input_index = tensor_target.input_index;
    if (input_index != 0) {
      return xla::FailedPrecondition(
          "Invalid allocation index %d for instruction", input_index,
          inst->ToString());
    }
    const Shape& allocation_shape = inst->operand(0)->shape();
    TF_ASSIGN_OR_RETURN(const popnn::ctc::Plan* plan, GetCTCPlan(res, inst));
    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(allocation_shape));
    TF_ASSIGN_OR_RETURN(poplar::Type in_dtype,
                        PoplarDataType(ctc_inst->in_dtype()));
    const int64_t batch_size = ShapeUtil::GetDimension(allocation_shape, 1);
    const int64_t max_time = ShapeUtil::GetDimension(allocation_shape, 0);
    const int64_t num_classes = ShapeUtil::GetDimension(allocation_shape, 2);
    if (in_dtype != dtype) {
      return xla::FailedPrecondition(
          "dtype of the data input tensor (%s) "
          "does not match the specified "
          "input dtype (%s)",
          dtype.toString(), in_dtype.toString());
    }
    return popnn::ctc_infer::createDataInput(graph, dtype, batch_size, max_time,
                                             num_classes, *plan,
                                             {debug_info, "data"});
  }
};

class CTCBeamSearchWithLogitsOp : public CTCBeamSearchOpBase {
  const std::string ClassName() const override { return "CTCLossWithLogitsOp"; }

  CTCBeamSearchOpBase::BeamSearchReturns PerformBeamSearch(
      poplar::Graph& graph, const poplar::Tensor& data,
      const poplar::Tensor& data_lengths, DriverProgramSequence& prog,
      int64_t blank_class, int64_t beam_width, int64_t top_paths,
      const popnn::ctc::Plan& plan,
      const poplar::DebugNameAndId& debug_name_and_id) const override {
    poplar::Tensor input_lengths =
        data_lengths.elementType() == poplar::UNSIGNED_INT
            ? data_lengths
            : popops::cast(graph, data_lengths, poplar::UNSIGNED_INT, prog,
                           debug_name_and_id);

    CTCBeamSearchOpBase::BeamSearchReturns result;
    std::tie(result.label_probabilities, result.label_lengths,
             result.decoded_labels) =
        popnn::ctc_infer::beamSearchDecoderLogits(
            graph, data, input_lengths, prog, blank_class, beam_width,
            top_paths, plan, debug_name_and_id);

    // Op def expects signed integer type tensors
    result.label_lengths = popops::cast(graph, result.label_lengths,
                                        poplar::INT, prog, debug_name_and_id);
    result.decoded_labels = popops::cast(graph, result.decoded_labels,
                                         poplar::INT, prog, debug_name_and_id);
    return result;
  }
};

REGISTER_POPLAR_OP(CTCBeamSearchWithLogits, CTCBeamSearchWithLogitsOp);

class CTCBeamSearchWithLogProbsOp : public CTCBeamSearchOpBase {
  const std::string ClassName() const override {
    return "CTCBeamSearchWithLogProbsOp";
  }

  CTCBeamSearchOpBase::BeamSearchReturns PerformBeamSearch(
      poplar::Graph& graph, const poplar::Tensor& data,
      const poplar::Tensor& data_lengths, DriverProgramSequence& prog,
      int64_t blank_class, int64_t beam_width, int64_t top_paths,
      const popnn::ctc::Plan& plan,
      const poplar::DebugNameAndId& debug_name_and_id) const override {
    CTCBeamSearchOpBase::BeamSearchReturns result;

    poplar::Tensor input_lengths =
        data_lengths.elementType() == poplar::UNSIGNED_INT
            ? data_lengths
            : popops::cast(graph, data_lengths, poplar::UNSIGNED_INT, prog,
                           debug_name_and_id);

    std::tie(result.label_probabilities, result.label_lengths,
             result.decoded_labels) =
        popnn::ctc_infer::beamSearchDecoderLogProbabilities(
            graph, data, input_lengths, prog, blank_class, beam_width,
            top_paths, plan, debug_name_and_id);

    // Op def expects signed integer type tensors
    result.label_lengths = popops::cast(graph, result.label_lengths,
                                        poplar::INT, prog, debug_name_and_id);
    result.decoded_labels = popops::cast(graph, result.decoded_labels,
                                         poplar::INT, prog, debug_name_and_id);
    return result;
  }
};

REGISTER_POPLAR_OP(CTCBeamSearchWithLogProbs, CTCBeamSearchWithLogProbsOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
