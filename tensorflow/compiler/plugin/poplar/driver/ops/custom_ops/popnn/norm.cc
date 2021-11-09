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

#include <gcl/Collectives.hpp>
#include <poplar/DebugContext.hpp>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
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
      if (IsPoplarInstruction(PoplarOp::GroupNormInference, inst) ||
          IsPoplarInstruction(PoplarOp::GroupNormTraining, inst) ||
          IsPoplarInstruction(PoplarOp::GroupNormGrad, inst) ||
          IsPoplarInstruction(PoplarOp::GroupNormStatistics, inst)) {
        const auto* inst_gn = Cast<HloGroupNormBaseInstruction>(inst);

        auto optional_feature_index =
            convert_scalar<uint32>(inst_gn->feature_index());
        if (!optional_feature_index) {
          return xla::FailedPrecondition(
              "Norm - Feature index cannot be interpreted as an unsigned "
              "integer.");
        }
        const auto feature_index = *optional_feature_index;

        auto optional_num_groups =
            convert_scalar<uint32>(inst_gn->num_groups());
        if (!optional_num_groups) {
          return xla::FailedPrecondition(
              "Norm - Num groups cannot be interpreted as an unsigned "
              "integer.");
        }
        const auto num_groups = *optional_num_groups;

        // Build a poplar::OptionFlags instance and indicate if channel
        // grouping is to be used.
        poplar::OptionFlags flags;
        const auto scg = inst_gn->strided_channel_grouping() ? "true" : "false";
        flags.set("groupNormStridedChannelGrouping", scg);

        return NormOptions{NormType::GroupNorm, feature_index, num_groups,
                           inst_gn->epsilon(), flags};

      } else {
        CHECK(IsPoplarInstruction(PoplarOp::BatchNormStatistics, inst));
        const auto* norm_inst = Cast<HloBatchNormStatsInstruction>(inst);

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

poplar::Tensor ConvertVarianceToInvStdDev(
    poplar::Graph& graph, const poplar::Tensor& variance, const float epsilon,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return popops::map(graph, pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon)),
                     {variance}, seq, {debug_name_and_id, "VarToInvStdDev"});
}

poplar::Tensor ConvertInvStdDevToVariance(
    poplar::Graph& graph, const poplar::Tensor& inv_sd, const float epsilon,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id) {
  return popops::map(graph, pe::InvStdDevToVariance(pe::_1, pe::Const(epsilon)),
                     {inv_sd}, seq, {debug_name_and_id, "InvStdDevToVar"});
}

poplar::Tensor BatchNormalise(poplar::Graph& graph,
                              const poplar::Tensor& operand,
                              const poplar::Tensor& scale,
                              const poplar::Tensor& offset,
                              const poplar::Tensor& mean,
                              const poplar::Tensor& inv_sd,
                              poplar::program::Sequence& seq,
                              const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Tensor multiplicand =
      popops::map(graph, pe::Mul(pe::_1, pe::_2), {scale, inv_sd}, seq,
                  {debug_name_and_id, "Multiplicand"});
  poplar::Tensor addend = popops::map(
      graph, pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
      {offset, multiplicand, mean}, seq, {debug_name_and_id, "Addend"});
  return popnn::bn::batchNormalise(graph, operand, multiplicand, addend, seq,
                                   {debug_name_and_id});
}

StatusOr<poplar::Tensor> AddNormScaleTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& debug_name,
    const HloInstruction* layout, uint64 layout_output_idx,
    uint32 feature_index, const TensorMap& tensor_map,
    const HloInstruction* inst,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(TensorVector outputs,
                      FindInstructionOutputTensors(tensor_map, res, layout));

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto shuffled = ShuffleNormInputToPoplar(acts, feature_index);

  // `gamma` is appended to the name by the createNorm function
  return poplin::createNormGamma(graph, shuffled, {debug_name_and_id});
}

StatusOr<poplar::Tensor> AddNormOffsetTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& debug_name,
    const HloInstruction* layout, uint64 layout_output_idx,
    uint32 feature_index, const TensorMap& tensor_map,
    const HloInstruction* inst,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(TensorVector outputs,
                      FindInstructionOutputTensors(tensor_map, res, layout));

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto shuffled = ShuffleNormInputToPoplar(acts, feature_index);

  // `beta` is appended to the name by the createNorm function
  return poplin::createNormBeta(graph, shuffled, {debug_name_and_id});
}

int64 CalculateNormBatchSize(const poplar::Tensor& t,
                             int64 replica_group_size) {
  CHECK_GT(t.rank(), 1);
  return t.dim(0) * replica_group_size;
}

poplin::DistributedNormReduceCallback GetDistributedNormReduceCallback(
    const std::string& debug_name_ref) {
  const std::string debug_name = debug_name_ref;
  return
      [debug_name](
          poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
          poplar::program::Sequence& prog, unsigned replica_group_size,
          const poplar::DebugContext& debug_context,
          const poplar::OptionFlags& options) -> std::vector<poplar::Tensor> {
        PoplarOpDefDebugInfo debug_info(debug_context, debug_name);

        // Use multi-tensor allReduce to reduce them all at the same time even
        // if they have different types.
        return gcl::allReduceCrossReplica(
            graph, inputs, popops::CollectiveOperator::ADD, prog,
            {gcl::CommGroupType::CONSECUTIVE, replica_group_size}, {debug_info},
            options);
      };
}

class NormInferenceAndTrainingOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "NormInferenceAndTrainingOp");
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;
    absl::optional<const HloInstruction*> layout = tensor_target.layout;
    absl::optional<int64> layout_output_idx = tensor_target.layout_output_idx;
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));

    switch (input_index) {
      case 1: {
        return AddNormScaleTensor(graph, res, name, *layout, *layout_output_idx,
                                  norm_opts.feature_index, tensor_map, inst,
                                  {debug_info});
      }
      case 2: {
        return AddNormOffsetTensor(graph, res, name, *layout,
                                   *layout_output_idx, norm_opts.feature_index,
                                   tensor_map, inst, {debug_info});
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
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "NormInferenceOp");
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_operand,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_scale,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_offset,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_mean,
        FindInstructionInput(tensor_map, res, inst, 3, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_variance_or_inv_std_dev,
        FindInstructionInput(tensor_map, res, inst, 4, seq, {debug_info},
                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor out = graph.addConstant(arg_operand.elementType(), {1}, 0,
                                             {debug_info, "out"});
      graph.setTileMapping(out, 0);
      TF_ASSIGN_OR_RETURN(out,
                          BroadcastTensor(out, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
      return seq;
    }

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, debug_name_and_id, norm_opts](
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
          poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
              graph, variance_or_inv_std_dev, norm_opts.epsilon, prog,
              {debug_name_and_id});
          args[5] = BatchNormalise(graph, operand, scale, offset, mean, inv_sd,
                                   prog, {debug_name_and_id});
          break;
        }
        case NormType::GroupNorm: {
          // For group norm variance_or_inv_std_dev is inv_std_dev, so we
          // don't need to convert it.
          args[5] =
              popnn::gn::groupNormalise(graph, operand, scale, offset, mean,
                                        variance_or_inv_std_dev, prog,
                                        {debug_name_and_id}, norm_opts.flags)
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
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    PoplarOpDefDebugInfo debug_info(debug_context, "NormTrainingOp");
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_operand,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_scale,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_offset,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info},
                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor out = graph.addConstant(arg_operand.elementType(), {1}, 0,
                                             {debug_info, "out"});
      graph.setTileMapping(out, 0);
      TF_ASSIGN_OR_RETURN(out,
                          BroadcastTensor(out, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
      poplar::Tensor mean = graph.addConstant(arg_operand.elementType(), {1},
                                              NAN, {debug_info, "mean"});
      graph.setTileMapping(mean, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
      poplar::Tensor variance_or_inv_std_dev =
          graph.addConstant(arg_operand.elementType(), {1}, NAN,
                            {debug_info, "varianceOrInvStdDev"});
      graph.setTileMapping(variance_or_inv_std_dev, 0);
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));
      return seq;
    }

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto& main_graph = GetMasterGraph(res);
    auto func =
        [&main_graph, &graph, debug_name_and_id, norm_opts,
         use_stable_statistics = res.use_stable_norm_statistics,
         replica_group_size =
             res.experimental_distributed_batch_norm_replica_group_size](
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
              if (replica_group_size > 1) {
                std::tie(args[4], inv_sd) =
                    popnn::bn::distributedBatchNormStatistics(
                        main_graph, operand, norm_opts.epsilon, prog,
                        /*unbiasedVarEstimate=*/false,
                        GetDistributedNormReduceCallback(
                            "DistributedBatchNormStatistics"),
                        CalculateNormBatchSize(operand, replica_group_size),
                        use_stable_statistics, poplar::FLOAT,
                        {debug_name_and_id});
              } else {
                std::tie(args[4], inv_sd) = popnn::bn::batchNormStatistics(
                    graph, operand, norm_opts.epsilon, prog,
                    /*unbiasedVarEstimate=*/false, use_stable_statistics,
                    poplar::FLOAT, {debug_name_and_id});
              }

              args[3] = BatchNormalise(graph, operand, scale, offset, args[4],
                                       inv_sd, prog, {debug_name_and_id});
              // For batch norm variance_or_inv_std_dev is variance, so we need
              // to convert it.
              args[5] = ConvertInvStdDevToVariance(
                  graph, inv_sd, norm_opts.epsilon, prog, {debug_name_and_id});
              break;
            }
            case NormType::GroupNorm: {
              // For group norm variance_or_inv_std_dev is inv_std_dev, so we
              // don't need to convert it.
              std::tie(args[4], args[5]) = popnn::gn::groupNormStatistics(
                  graph, operand, norm_opts.epsilon, prog,
                  *norm_opts.num_groups,
                  /*unbiasedVarEstimate=*/false, use_stable_statistics,
                  poplar::FLOAT, {debug_name_and_id}, norm_opts.flags);

              args[3] = popnn::gn::groupNormalise(
                            graph, operand, scale, offset, args[4], args[5],
                            prog, {debug_name_and_id}, norm_opts.flags)
                            .first;
              break;
            }
          }
          args[3] =
              ShuffleNormOutputToTensorflow(args[3], norm_opts.feature_index);
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
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    PoplarOpDefDebugInfo debug_info(debug_context, "NormGradOp");
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_operand,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_scale,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_mean,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_variance_or_inv_std_dev,
        FindInstructionInput(tensor_map, res, inst, 3, seq, {debug_info},
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_grad_output,
        FindInstructionInput(tensor_map, res, inst, 4, seq, {debug_info},
                             /*expand_aliasing*/ false));
    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor operand_grad = graph.addConstant(
          arg_operand.elementType(), {1}, 0, {debug_info, "operandGrad"});
      graph.setTileMapping(operand_grad, 0);
      TF_ASSIGN_OR_RETURN(
          operand_grad,
          BroadcastTensor(operand_grad, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
      poplar::Tensor scale_grad = graph.addConstant(
          arg_operand.elementType(), {1}, 0, {debug_info, "scaleGrad"});
      graph.setTileMapping(scale_grad, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
      poplar::Tensor offset_grad = graph.addConstant(
          arg_operand.elementType(), {1}, 0, {debug_info, "offsetGrad"});
      graph.setTileMapping(offset_grad, 0);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
      return seq;
    }

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto& main_graph = GetMasterGraph(res);
    auto func =
        [&main_graph, &graph, debug_name_and_id, norm_opts,
         replica_group_size =
             res.experimental_distributed_batch_norm_replica_group_size](
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
              // For batch norm variance_or_inv_std_dev is variance, so we need
              // to convert it.
              poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
                  graph, variance_or_inv_std_dev, norm_opts.epsilon, prog,
                  {debug_name_and_id});
              poplar::Tensor operand_whitened = popnn::bn::batchNormWhiten(
                  graph, operand, mean, inv_sd, prog,
                  {debug_name_and_id, "WhitenedActs"});

              // Compute the grad for the operand.
              if (replica_group_size > 1) {
                args[5] = popnn::bn::distributedBatchNormGradients(
                    main_graph, operand_whitened, grad_output, inv_sd, scale,
                    prog,
                    GetDistributedNormReduceCallback(
                        "DistributedBatchNormGradients"),
                    CalculateNormBatchSize(operand, replica_group_size),
                    poplar::FLOAT, {debug_name_and_id, "OperandGrad"});
              } else {
                args[5] = popnn::bn::batchNormGradients(
                    graph, operand_whitened, grad_output, inv_sd, scale, prog,
                    poplar::FLOAT, {debug_name_and_id, "OperandGrad"});
              }
              // Compute the grads for the scale and offset.
              std::tie(args[6], args[7]) = popnn::bn::batchNormParamGradients(
                  graph, operand_whitened, grad_output, prog, poplar::FLOAT,
                  {debug_name_and_id, "ScaleOffsetGrads"});
              break;
            }
            case NormType::GroupNorm: {
              // For group norm variance_or_inv_std_dev is inv_std_dev, so we
              // don't need to convert it.
              poplar::Tensor operand_whitened = popnn::gn::groupNormWhiten(
                  graph, operand, mean, variance_or_inv_std_dev, prog,
                  {debug_name_and_id, "WhitenedActs"}, norm_opts.flags);

              // Compute the grad for the operand.
              args[5] = popnn::gn::groupNormGradients(
                  graph, operand_whitened, grad_output, variance_or_inv_std_dev,
                  scale, prog, poplar::FLOAT,
                  {debug_name_and_id, "OperandGrad"}, norm_opts.flags);
              // Compute the grads for the scale and offset.
              std::tie(args[6], args[7]) = popnn::gn::groupNormParamGradients(
                  graph, operand_whitened, grad_output, prog, poplar::FLOAT,
                  {debug_name_and_id, "ScaleOffsetGrads"}, norm_opts.flags);
              break;
            }
          }
          args[5] =
              ShuffleNormOutputToTensorflow(args[5], norm_opts.feature_index);
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
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    TF_ASSIGN_OR_RETURN(const NormOptions norm_opts, GetNormOptions(inst));
    PoplarOpDefDebugInfo debug_info(debug_context, "NormStatisticsOp");
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_operand,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info},
                             /*expand_aliasing*/ false));

    // Special case - zero sized array
    if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
      poplar::Tensor mean = graph.addConstant(arg_operand.elementType(), {1}, 0,
                                              {debug_info, "mean"});
      graph.setTileMapping(mean, 0);
      TF_ASSIGN_OR_RETURN(mean,
                          BroadcastTensor(mean, inst->operand(0)->shape(), {}));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
      poplar::Tensor variance_or_inv_std_dev =
          graph.addConstant(arg_operand.elementType(), {1}, 0,
                            {debug_info, "varianceOrInvStdDev"});
      graph.setTileMapping(variance_or_inv_std_dev, 0);
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
      return seq;
    }

    auto& main_graph = GetMasterGraph(res);
    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func =
        [&main_graph, &graph, debug_name_and_id, norm_opts,
         use_stable_statistics = res.use_stable_norm_statistics,
         replica_group_size =
             res.experimental_distributed_batch_norm_replica_group_size](
            std::vector<poplar::Tensor>& args,
            poplar::program::Sequence& prog) {
          poplar::Tensor operand = args[0];
          // Move the channels.
          operand = ShuffleNormInputToPoplar(operand, norm_opts.feature_index);

          switch (norm_opts.type) {
            case NormType::BatchNorm: {
              poplar::Tensor inv_sd;

              if (replica_group_size > 1) {
                std::tie(args[1], inv_sd) =
                    popnn::bn::distributedBatchNormStatistics(
                        main_graph, operand, norm_opts.epsilon, prog,
                        /*unbiasedVarEstimate=*/false,
                        GetDistributedNormReduceCallback(
                            "DistributedBatchNormStatistics"),
                        CalculateNormBatchSize(operand, replica_group_size),
                        use_stable_statistics, poplar::FLOAT,
                        {debug_name_and_id});
              } else {
                std::tie(args[1], inv_sd) = popnn::bn::batchNormStatistics(
                    graph, operand, norm_opts.epsilon, prog,
                    /*unbiasedVarEstimate=*/false, use_stable_statistics,
                    poplar::FLOAT, {debug_name_and_id});
              }

              // For batch norm variance_or_inv_std_dev is variance, so we need
              // to convert it.
              args[2] = ConvertInvStdDevToVariance(
                  graph, inv_sd, norm_opts.epsilon, prog, {debug_name_and_id});
              break;
            }
            case NormType::GroupNorm: {
              // For group norm variance_or_inv_std_dev is inv_std_dev, so we
              // don't need to convert it.
              std::tie(args[1], args[2]) = popnn::gn::groupNormStatistics(
                  graph, operand, norm_opts.epsilon, prog,
                  *norm_opts.num_groups,
                  /*unbiasedVarEstimate=*/false, use_stable_statistics,
                  poplar::FLOAT, {debug_name_and_id}, norm_opts.flags);
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
REGISTER_POPLAR_OP(BatchNormStatistics, NormStatisticsOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
