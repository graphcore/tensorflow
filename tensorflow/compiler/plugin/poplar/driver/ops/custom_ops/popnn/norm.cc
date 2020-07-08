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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/GraphFunction.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace pe = popops::expr;
namespace pg = poputil::graphfn;

namespace xla {
namespace poplarplugin {
namespace {

enum class NormType {
  BatchNorm,
  GroupNorm,
};

struct NormOptions {
  NormType type;
  uint32 feature_index;
  absl::optional<uint32> num_groups;
  float epsilon;
  poplar::OptionFlags flags;
};

StatusOr<NormOptions> GetNormOptions(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormGrad: {
      const HloBatchNormInstruction* norm_inst =
          Cast<HloBatchNormInstruction>(inst);

      auto optional_feature_index =
          convert_scalar<uint32>(norm_inst->feature_index());
      if (!optional_feature_index) {
        return xla::FailedPrecondition(
            "Norm - Feature index cannot be interpreted as an unsigned "
            "integer.");
      }
      const auto feature_index = *optional_feature_index;

      return NormOptions{NormType::BatchNorm, feature_index, absl::nullopt,
                         norm_inst->epsilon()};
    }
    default: {
      auto norm_inst = Cast<HloNormInstruction>(inst);

      auto optional_feature_index =
          convert_scalar<uint32>(norm_inst->feature_index());
      if (!optional_feature_index) {
        return xla::FailedPrecondition(
            "Norm - Feature index cannot be interpreted as an unsigned "
            "integer.");
      }
      const auto feature_index = *optional_feature_index;

      auto optional_num_groups =
          convert_scalar<uint32>(norm_inst->num_groups());
      if (!optional_num_groups) {
        return xla::FailedPrecondition(
            "Norm - Num groups cannot be interpreted as an unsigned integer.");
      }
      const auto num_groups = *optional_num_groups;

      // Build a poplar::OptionFlags instance and indicate if channel
      // grouping is to be used.
      poplar::OptionFlags flags;
      const auto* inst_gn = Cast<HloGroupNormBaseInstruction>(inst);
      const auto scg = inst_gn->strided_channel_grouping() ? "true" : "false";
      flags.set("groupNormStridedChannelGrouping", scg);

      return NormOptions{NormType::GroupNorm, feature_index, num_groups,
                         norm_inst->epsilon(), flags};
    }
  }
}

poplar::Tensor ShuffleNormInputToPoplar(const poplar::Tensor& input,
                                        uint32 feature_index) {
  return input.dimShufflePartial({feature_index}, {1});
}

poplar::Tensor ShuffleNormOutputToTensorflow(const poplar::Tensor& output,
                                             uint32 feature_index) {
  return output.dimShufflePartial({1}, {feature_index});
}

poplar::Tensor ConvertVarianceToInvStdDev(poplar::Graph& graph,
                                          const poplar::Tensor& variance,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  poplar::Tensor inv_sd = graph.clone(variance);
  seq.add(poplar::program::Copy(variance, inv_sd));

  popops::mapInPlace(graph, pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon)),
                     {inv_sd}, seq, debug_name + "/VarToInvStdDev");
  return inv_sd;
}

poplar::Tensor ConvertInvStdDevToVariance(poplar::Graph& graph,
                                          const poplar::Tensor& inv_sd,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  poplar::Tensor variance = graph.clone(inv_sd);
  seq.add(poplar::program::Copy(inv_sd, variance));

  popops::mapInPlace(graph, pe::InvStdDevToVariance(pe::_1, pe::Const(epsilon)),
                     {variance}, seq, debug_name + "/InvStdDevToVar");
  return variance;
}

poplar::Tensor BatchNormalise(
    poplar::Graph& graph, const poplar::Tensor& operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& inv_sd,
    poplar::program::Sequence& seq, const std::string& debug_name) {
  poplar::Tensor multiplicand =
      popops::map(graph, pe::Mul(pe::_1, pe::_2), {scale, inv_sd}, seq,
                  debug_name + "/Multiplicand");
  poplar::Tensor addend =
      popops::map(graph, pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                  {offset, multiplicand, mean}, seq, debug_name + "/Addend");
  return popnn::bn::batchNormalise(graph, operand, multiplicand, addend, seq,
                                   debug_name);
}

StatusOr<poplar::Tensor> AddNormScaleTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& debug_name,
    const HloInstruction* layout, uint64 layout_output_idx,
    uint32 feature_index, const TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(TensorVector outputs,
                      FindInstructionOutputTensors(tensor_map, res, layout));

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto shuffled = ShuffleNormInputToPoplar(acts, feature_index);
  return poplin::createNormGamma(graph, shuffled);
}

StatusOr<poplar::Tensor> AddNormOffsetTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& debug_name,
    const HloInstruction* layout, uint64 layout_output_idx,
    uint32 feature_index, const TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(TensorVector outputs,
                      FindInstructionOutputTensors(tensor_map, res, layout));

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto shuffled = ShuffleNormInputToPoplar(acts, feature_index);
  return poplin::createNormBeta(graph, shuffled);
}

class NormInferenceAndTrainingOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    absl::optional<const HloInstruction*> layout = tensor_target.layout;
    absl::optional<int64> layout_output_idx = tensor_target.layout_output_idx;
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));

    switch (input_index) {
      case 1: {
        return AddNormScaleTensor(graph, res, name, *layout, *layout_output_idx,
                                  norm_opts.feature_index, tensor_map);
      }
      case 2: {
        return AddNormOffsetTensor(graph, res, name, *layout,
                                   *layout_output_idx, norm_opts.feature_index,
                                   tensor_map);
      }
      default: {
        return xla::FailedPrecondition(
            "NormInferenceTraining op %s should not be allocating on index "
            "%lld.",
            inst->name().c_str(), input_index);
      }
    }
  }
};

class NormInferenceOp : public NormInferenceAndTrainingOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq;

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_operand,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_scale,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_offset,
                        FindInstructionInput(tensor_map, res, inst, 2, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_mean,
                        FindInstructionInput(tensor_map, res, inst, 3, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_variance_or_inv_std_dev,
                        FindInstructionInput(tensor_map, res, inst, 4, seq,
                                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor out = graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(out, 0);
      TF_ASSIGN_OR_RETURN(out,
                          BroadcastTensor(out, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
      return seq;
    }

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, debug_prefix, norm_opts](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor operand = args[0];
      poplar::Tensor scale = args[1];
      poplar::Tensor offset = args[2];
      poplar::Tensor mean = args[3];
      poplar::Tensor variance_or_inv_std_dev = args[4];
      // Move the channels.
      operand = ShuffleNormInputToPoplar(operand, norm_opts.feature_index);

      switch (norm_opts.type) {
        case NormType::BatchNorm: {
          // For batch norm variance_or_inv_std_dev is variance, so we need to
          // convert it.
          poplar::Tensor inv_sd =
              ConvertVarianceToInvStdDev(graph, variance_or_inv_std_dev,
                                         norm_opts.epsilon, prog, debug_prefix);
          args[5] = BatchNormalise(graph, operand, scale, offset, mean, inv_sd,
                                   prog, debug_prefix);
          break;
        }
        case NormType::GroupNorm: {
          // For group norm variance_or_inv_std_dev is inv_std_dev, so we
          // don't need to convert it.
          args[5] =
              popnn::gn::groupNormalise(graph, operand, scale, offset, mean,
                                        variance_or_inv_std_dev, prog,
                                        debug_prefix, norm_opts.flags)
                  .first;
          break;
        }
      }
      args[5] = ShuffleNormOutputToTensorflow(args[5], norm_opts.feature_index);
    };

    poplar::Tensor output;
    std::vector<poplar::Tensor> args = {arg_operand,
                                        arg_scale,
                                        arg_offset,
                                        arg_mean,
                                        arg_variance_or_inv_std_dev,
                                        output};
    pg::Signature signature = {
        pg::input(arg_operand, "operand"),
        pg::input(arg_scale, "scale"),
        pg::input(arg_offset, "offset"),
        pg::input(arg_mean, "mean"),
        pg::input(arg_variance_or_inv_std_dev, "variance_or_inv_std_dev"),
        pg::created("output")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args, {}, {{1, 0}, {2, 0}}));

    output = args[5];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};
REGISTER_POPLAR_OP(GroupNormInference, NormInferenceOp);
REGISTER_HLO_OP(kBatchNormInference, NormInferenceOp);

class NormTrainingOp : public NormInferenceAndTrainingOp {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq;

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_operand,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_scale,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_offset,
                        FindInstructionInput(tensor_map, res, inst, 2, seq,
                                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor out = graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(out, 0);
      TF_ASSIGN_OR_RETURN(out,
                          BroadcastTensor(out, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
      poplar::Tensor mean =
          graph.addConstant(arg_operand.elementType(), {1}, NAN);
      graph.setTileMapping(mean, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
      poplar::Tensor variance_or_inv_std_dev =
          graph.addConstant(arg_operand.elementType(), {1}, NAN);
      graph.setTileMapping(variance_or_inv_std_dev, 0);
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));
      return seq;
    }

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, debug_prefix, norm_opts,
                 use_stable_statistics = res.use_stable_norm_statistics](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor operand = args[0];
      poplar::Tensor scale = args[1];
      poplar::Tensor offset = args[2];

      // Move the channels.
      operand = ShuffleNormInputToPoplar(operand, norm_opts.feature_index);

      switch (norm_opts.type) {
        case NormType::BatchNorm: {
          poplar::Tensor inv_sd;
          std::tie(args[4], inv_sd) = popnn::bn::batchNormStatistics(
              graph, operand, norm_opts.epsilon, prog,
              /*unbiasedVarEstimate=*/false, use_stable_statistics,
              poplar::FLOAT, debug_prefix);

          args[3] = BatchNormalise(graph, operand, scale, offset, args[4],
                                   inv_sd, prog, debug_prefix);
          // For batch norm variance_or_inv_std_dev is variance, so we need to
          // convert it.
          args[5] = ConvertInvStdDevToVariance(graph, inv_sd, norm_opts.epsilon,
                                               prog, debug_prefix);
          break;
        }
        case NormType::GroupNorm: {
          // For group norm variance_or_inv_std_dev is inv_std_dev, so we
          // don't need to convert it.
          std::tie(args[4], args[5]) = popnn::gn::groupNormStatistics(
              graph, operand, norm_opts.epsilon, prog, *norm_opts.num_groups,
              /*unbiasedVarEstimate=*/false, use_stable_statistics,
              poplar::FLOAT, debug_prefix, norm_opts.flags);

          args[3] = popnn::gn::groupNormalise(graph, operand, scale, offset,
                                              args[4], args[5], prog,
                                              debug_prefix, norm_opts.flags)
                        .first;
          break;
        }
      }
      args[3] = ShuffleNormOutputToTensorflow(args[3], norm_opts.feature_index);
    };

    poplar::Tensor output, mean, variance_or_inv_std_dev;
    std::vector<poplar::Tensor> args = {arg_operand, arg_scale,
                                        arg_offset,  output,
                                        mean,        variance_or_inv_std_dev};
    pg::Signature signature = {pg::input(arg_operand, "operand"),
                               pg::input(arg_scale, "scale"),
                               pg::input(arg_offset, "offset"),
                               pg::created("output"),
                               pg::created("mean"),
                               pg::created("variance_or_inv_std_dev")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args, {}, {{1, 0}, {2, 0}}));

    output = args[3];
    mean = args[4];
    variance_or_inv_std_dev = args[5];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));

    return seq;
  }
};
REGISTER_POPLAR_OP(GroupNormTraining, NormTrainingOp);
REGISTER_HLO_OP(kBatchNormTraining, NormTrainingOp);

class NormGradOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq;

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_operand,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_scale,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_mean,
                        FindInstructionInput(tensor_map, res, inst, 2, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_variance_or_inv_std_dev,
                        FindInstructionInput(tensor_map, res, inst, 3, seq,
                                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_grad_output,
                        FindInstructionInput(tensor_map, res, inst, 4, seq,
                                             /*expand_aliasing*/ false));
    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor operand_grad =
          graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(operand_grad, 0);
      TF_ASSIGN_OR_RETURN(
          operand_grad,
          BroadcastTensor(operand_grad, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
      poplar::Tensor scale_grad =
          graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(scale_grad, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
      poplar::Tensor offset_grad =
          graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(offset_grad, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
      return seq;
    }

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, debug_prefix, norm_opts](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor operand = args[0];
      poplar::Tensor scale = args[1];
      poplar::Tensor mean = args[2];
      poplar::Tensor variance_or_inv_std_dev = args[3];
      poplar::Tensor grad_output = args[4];

      // Move the channels.
      operand = ShuffleNormInputToPoplar(operand, norm_opts.feature_index);
      grad_output =
          ShuffleNormInputToPoplar(grad_output, norm_opts.feature_index);

      switch (norm_opts.type) {
        case NormType::BatchNorm: {
          // For batch norm variance_or_inv_std_dev is variance, so we need to
          // convert it.
          poplar::Tensor inv_sd =
              ConvertVarianceToInvStdDev(graph, variance_or_inv_std_dev,
                                         norm_opts.epsilon, prog, debug_prefix);
          poplar::Tensor operand_whitened =
              popnn::bn::batchNormWhiten(graph, operand, mean, inv_sd, prog,
                                         debug_prefix + "/WhitenedActs");

          // Compute the grad for the operand.
          args[5] = popnn::bn::batchNormGradients(
              graph, operand_whitened, grad_output, inv_sd, scale, prog,
              poplar::FLOAT, debug_prefix + "/OperandGrad");
          // Compute the grads for the scale and offset.
          std::tie(args[6], args[7]) = popnn::bn::batchNormParamGradients(
              graph, operand_whitened, grad_output, prog, poplar::FLOAT,
              debug_prefix + "/ScaleOffsetGrads");
          break;
        }
        case NormType::GroupNorm: {
          // For group norm variance_or_inv_std_dev is inv_std_dev, so we
          // don't need to convert it.
          poplar::Tensor operand_whitened = popnn::gn::groupNormWhiten(
              graph, operand, mean, variance_or_inv_std_dev, prog,
              debug_prefix + "/WhitenedActs", norm_opts.flags);

          // Compute the grad for the operand.
          args[5] = popnn::gn::groupNormGradients(
              graph, operand_whitened, grad_output, variance_or_inv_std_dev,
              scale, prog, poplar::FLOAT, debug_prefix + "/OperandGrad",
              norm_opts.flags);
          // Compute the grads for the scale and offset.
          std::tie(args[6], args[7]) = popnn::gn::groupNormParamGradients(
              graph, operand_whitened, grad_output, prog, poplar::FLOAT,
              debug_prefix + "/ScaleOffsetGrads", norm_opts.flags);
          break;
        }
      }
      args[5] = ShuffleNormOutputToTensorflow(args[5], norm_opts.feature_index);
    };

    poplar::Tensor operand_grad, scale_grad, offset_grad;
    std::vector<poplar::Tensor> args = {
        arg_operand,     arg_scale,    arg_mean,   arg_variance_or_inv_std_dev,
        arg_grad_output, operand_grad, scale_grad, offset_grad};
    pg::Signature signature = {
        pg::input(arg_operand, "operand"),
        pg::input(arg_scale, "scale"),
        pg::input(arg_mean, "mean"),
        pg::input(arg_variance_or_inv_std_dev, "variance_or_inv_std_dev"),
        pg::input(arg_grad_output, "grad_output"),
        pg::created("operand_grad"),
        pg::created("scale_grad"),
        pg::created("offset_grad")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));

    operand_grad = args[5];
    scale_grad = args[6];
    offset_grad = args[7];
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
    return seq;
  }
};
REGISTER_POPLAR_OP(GroupNormGrad, NormGradOp);
REGISTER_HLO_OP(kBatchNormGrad, NormGradOp);

class NormStatisticsOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq;

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_operand,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor mean =
          graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(mean, 0);
      TF_ASSIGN_OR_RETURN(mean,
                          BroadcastTensor(mean, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
      poplar::Tensor variance_or_inv_std_dev =
          graph.addConstant(arg_operand.elementType(), {1}, 0);
      graph.setTileMapping(variance_or_inv_std_dev, 0);
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
      return seq;
    }

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, debug_prefix, norm_opts,
                 use_stable_statistics = res.use_stable_norm_statistics](
                    std::vector<poplar::Tensor>& args,
                    poplar::program::Sequence& prog) {
      poplar::Tensor operand = args[0];
      // Move the channels.
      operand = ShuffleNormInputToPoplar(operand, norm_opts.feature_index);

      switch (norm_opts.type) {
        case NormType::BatchNorm: {
          poplar::Tensor inv_sd;
          std::tie(args[1], inv_sd) = popnn::bn::batchNormStatistics(
              graph, operand, norm_opts.epsilon, prog,
              /*unbiasedVarEstimate=*/false, use_stable_statistics,
              poplar::FLOAT, debug_prefix);
          // For batch norm variance_or_inv_std_dev is variance, so we need to
          // convert it.
          args[2] = ConvertInvStdDevToVariance(graph, inv_sd, norm_opts.epsilon,
                                               prog, debug_prefix);
          break;
        }
        case NormType::GroupNorm: {
          // For group norm variance_or_inv_std_dev is inv_std_dev, so we
          // don't need to convert it.
          std::tie(args[1], args[2]) = popnn::gn::groupNormStatistics(
              graph, operand, norm_opts.epsilon, prog, *norm_opts.num_groups,
              /*unbiasedVarEstimate=*/false, use_stable_statistics,
              poplar::FLOAT, debug_prefix, norm_opts.flags);
          break;
        }
      }
    };
    poplar::Tensor mean, variance_or_inv_std_dev;
    std::vector<poplar::Tensor> args = {arg_operand, mean,
                                        variance_or_inv_std_dev};
    pg::Signature signature = {pg::input(arg_operand, "operand"),
                               pg::created("mean"),
                               pg::created("variance_or_inv_std_dev")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));
    mean = args[1];
    variance_or_inv_std_dev = args[2];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
    return seq;
  }
};
REGISTER_POPLAR_OP(GroupNormStatistics, NormStatisticsOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
