/* Copyright 2018-2019 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <poplar/Tensor.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {
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
}  // namespace

poplar::Tensor ShuffleNormInputToPoplar(const poplar::Tensor& input,
                                        const unsigned feature_dimension) {
  return input.dimShufflePartial({feature_dimension}, {1});
}

poplar::Tensor ShuffleNormOutputToTensorflow(const poplar::Tensor& output,
                                             const unsigned feature_dimension) {
  return output.dimShufflePartial({1}, {feature_dimension});
}

StatusOr<poplar::program::Program> CreateBatchNormInf(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormInstruction* batch_inf_inst =
      Cast<HloBatchNormInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  const auto epsilon = batch_inf_inst->epsilon();
  const unsigned dimension = batch_inf_inst->feature_index();

  return CreateNormInference(NormType::BatchNorm, graph, res, inst, epsilon,
                             dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormInference(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance_or_inv_std_dev,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(out, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }

  using namespace poputil::graphfn;
  const std::string debug_prefix = GetDebugName(inst);
  auto func = [&graph, debug_prefix, norm_type, epsilon, optional_num_groups](
                  std::vector<poplar::Tensor>& args,
                  poplar::program::Sequence& prog) {
    switch (norm_type) {
      case NormType::BatchNorm: {
        // For batch norm variance_or_inv_std_dev is variance, so we need to
        // convert it.
        poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
            graph, args[4], epsilon, prog, debug_prefix);
        args[5] = BatchNormalise(graph, args[0], args[1], args[2], args[3],
                                 inv_sd, prog, debug_prefix);
        break;
      }
      case NormType::GroupNorm: {
        // For group norm variance_or_inv_std_dev is inv_std_dev, so we
        // don't need to convert it.
        args[5] =
            popnn::gn::groupNormalise(graph, args[0], args[1], args[2], args[3],
                                      args[4], prog, debug_prefix)
                .first;
        break;
      }
    }
  };

  // Move the channels.
  operand = ShuffleNormInputToPoplar(operand, feature_dimension);
  poplar::Tensor output;
  std::vector<poplar::Tensor> args = {
      operand, scale, offset, mean, variance_or_inv_std_dev, output};
  Signature signature = {
      input(operand, "operand"),
      input(scale, "scale"),
      input(offset, "offset"),
      input(mean, "mean"),
      input(variance_or_inv_std_dev, "variance_or_inv_std_dev"),
      created("output")};

  TF_RETURN_IF_ERROR(
      res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));

  output = ShuffleNormOutputToTensorflow(args.back(), feature_dimension);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormTraining(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormTrainingInstruction* batch_train_inst =
      Cast<HloBatchNormTrainingInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);
  const auto epsilon = batch_train_inst->epsilon();
  const unsigned dimension = batch_train_inst->feature_index();
  return CreateNormTraining(NormType::BatchNorm, graph, res, inst, epsilon,
                            dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormTraining(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(out, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    poplar::Tensor mean = graph.addConstant(operand.elementType(), {1}, NAN);
    graph.setTileMapping(mean, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
    poplar::Tensor variance_or_inv_std_dev =
        graph.addConstant(operand.elementType(), {1}, NAN);
    graph.setTileMapping(variance_or_inv_std_dev, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));
    return seq;
  }

  using namespace poputil::graphfn;
  const std::string debug_prefix = GetDebugName(inst);
  auto func = [&graph, debug_prefix, norm_type, epsilon, optional_num_groups](
                  std::vector<poplar::Tensor>& args,
                  poplar::program::Sequence& prog) {
    switch (norm_type) {
      case NormType::BatchNorm: {
        poplar::Tensor inv_sd;
        std::tie(args[4], inv_sd) = popnn::bn::batchNormStatistics(
            graph, args[0], epsilon, prog, false, poplar::FLOAT, debug_prefix);

        args[3] = BatchNormalise(graph, args[0], args[1], args[2], args[4],
                                 inv_sd, prog, debug_prefix);
        // For batch norm variance_or_inv_std_dev is variance, so we need to
        // convert it.
        args[5] = ConvertInvStdDevToVariance(graph, inv_sd, epsilon, prog,
                                             debug_prefix);
        break;
      }
      case NormType::GroupNorm: {
        // For group norm variance_or_inv_std_dev is inv_std_dev, so we
        // don't need to convert it.
        std::tie(args[4], args[5]) = popnn::gn::groupNormStatistics(
            graph, args[0], epsilon, prog, *optional_num_groups, false,
            poplar::FLOAT, debug_prefix);

        args[3] =
            popnn::gn::groupNormalise(graph, args[0], args[1], args[2], args[4],
                                      args[5], prog, debug_prefix)
                .first;
        break;
      }
    }
  };

  // Move the channels.
  operand = ShuffleNormInputToPoplar(operand, feature_dimension);

  poplar::Tensor output, mean, variance_or_inv_std_dev;
  std::vector<poplar::Tensor> args = {operand, scale, offset,
                                      output,  mean,  variance_or_inv_std_dev};
  Signature signature = {
      input(operand, "operand"), input(scale, "scale"),
      input(offset, "offset"),   created("output"),
      created("mean"),           created("variance_or_inv_std_dev")};

  TF_RETURN_IF_ERROR(
      res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));

  output = ShuffleNormOutputToTensorflow(args[3], feature_dimension);
  mean = args[4];
  variance_or_inv_std_dev = args[5];

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormGradInstruction* batch_grad_inst =
      Cast<HloBatchNormGradInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  const auto epsilon = batch_grad_inst->epsilon();
  const unsigned dimension = batch_grad_inst->feature_index();
  return CreateNormGrad(NormType::BatchNorm, graph, res, inst, epsilon,
                        dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormGrad(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance_or_inv_std_dev,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor grad_output,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));
  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor operand_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(operand_grad, 0);
    TF_ASSIGN_OR_RETURN(
        operand_grad,
        BroadcastTensor(operand_grad, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
    poplar::Tensor scale_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(scale_grad, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
    poplar::Tensor offset_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(offset_grad, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
    return seq;
  }

  using namespace poputil::graphfn;
  const std::string debug_prefix = GetDebugName(inst);
  auto func = [&graph, debug_prefix, norm_type, epsilon, optional_num_groups](
                  std::vector<poplar::Tensor>& args,
                  poplar::program::Sequence& prog) {
    switch (norm_type) {
      case NormType::BatchNorm: {
        // For batch norm variance_or_inv_std_dev is variance, so we need to
        // convert it.
        poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
            graph, args[2], epsilon, prog, debug_prefix);
        poplar::Tensor operand_whitened =
            popnn::bn::batchNormWhiten(graph, args[0], args[1], inv_sd, prog,
                                       debug_prefix + "/WhitenedActs");

        // Compute the grad for the operand.
        args[5] = popnn::bn::batchNormGradients(
            graph, operand_whitened, args[4], inv_sd, args[3], prog,
            poplar::FLOAT, debug_prefix + "/OperandGrad");
        // Compute the grads for the scale and offset.
        std::tie(args[6], args[7]) = popnn::bn::batchNormParamGradients(
            graph, operand_whitened, args[4], prog, poplar::FLOAT,
            debug_prefix + "/ScaleOffsetGrads");
        break;
      }
      case NormType::GroupNorm: {
        // For group norm variance_or_inv_std_dev is inv_std_dev, so we
        // don't need to convert it.
        poplar::Tensor operand_whitened =
            popnn::gn::groupNormWhiten(graph, args[0], args[1], args[2], prog,
                                       debug_prefix + "/WhitenedActs");

        // Compute the grad for the operand.
        args[5] = popnn::gn::groupNormGradients(
            graph, operand_whitened, args[4], args[2], args[3], prog,
            poplar::FLOAT, debug_prefix + "/OperandGrad");
        // Compute the grads for the scale and offset.
        std::tie(args[6], args[7]) = popnn::gn::groupNormParamGradients(
            graph, operand_whitened, args[4], prog, poplar::FLOAT,
            debug_prefix + "/ScaleOffsetGrads");
        break;
      }
    }
  };

  // Move the channels.
  operand = ShuffleNormInputToPoplar(operand, feature_dimension);
  grad_output = ShuffleNormInputToPoplar(grad_output, feature_dimension);

  poplar::Tensor operand_grad, scale_grad, offset_grad;
  std::vector<poplar::Tensor> args = {
      operand,    mean,        variance_or_inv_std_dev,
      scale,      grad_output, operand_grad,
      scale_grad, offset_grad};
  Signature signature = {
      input(operand, "operand"),
      input(mean, "mean"),
      input(variance_or_inv_std_dev, "variance_or_inv_std_dev"),
      input(scale, "scale"),
      input(grad_output, "grad_output"),
      created("operand_grad"),
      created("scale_grad"),
      created("offset_grad")};

  TF_RETURN_IF_ERROR(
      res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));

  operand_grad = ShuffleNormOutputToTensorflow(args[5], feature_dimension);
  scale_grad = args[6];
  offset_grad = args[7];
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
  return seq;
}

StatusOr<poplar::program::Program> CreateNormStatistics(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor mean = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(mean, 0);
    TF_ASSIGN_OR_RETURN(mean,
                        BroadcastTensor(mean, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
    poplar::Tensor variance_or_inv_std_dev =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(variance_or_inv_std_dev, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
    return seq;
  }

  using namespace poputil::graphfn;
  const std::string debug_prefix = GetDebugName(inst);
  auto func = [&graph, debug_prefix, norm_type, epsilon, optional_num_groups](
                  std::vector<poplar::Tensor>& args,
                  poplar::program::Sequence& prog) {
    switch (norm_type) {
      case NormType::BatchNorm: {
        poplar::Tensor inv_sd;
        std::tie(args[1], inv_sd) = popnn::bn::batchNormStatistics(
            graph, args[0], epsilon, prog, false, poplar::FLOAT, debug_prefix);
        // For batch norm variance_or_inv_std_dev is variance, so we need to
        // convert it.
        args[2] = ConvertInvStdDevToVariance(graph, inv_sd, epsilon, prog,
                                             debug_prefix);
        break;
      }
      case NormType::GroupNorm: {
        // For group norm variance_or_inv_std_dev is inv_std_dev, so we
        // don't need to convert it.
        std::tie(args[1], args[2]) = popnn::gn::groupNormStatistics(
            graph, args[0], epsilon, prog, *optional_num_groups, false,
            poplar::FLOAT, debug_prefix);
        break;
      }
    }
  };

  // Move the channels.
  operand = ShuffleNormInputToPoplar(operand, feature_dimension);
  poplar::Tensor mean, variance_or_inv_std_dev;
  std::vector<poplar::Tensor> args = {operand, mean, variance_or_inv_std_dev};
  Signature signature = {input(operand, "operand"), created("mean"),
                         created("variance_or_inv_std_dev")};

  TF_RETURN_IF_ERROR(
      res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));
  mean = args[1];
  variance_or_inv_std_dev = args[2];

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
