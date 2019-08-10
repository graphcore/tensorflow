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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gru.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Gru.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Util.hpp>

namespace xla {
namespace poplarplugin {
namespace {
static const size_t basic_gru_cell_num_units = 3;

StatusOr<popnn::gru::GruParams> GetGruParameters(const HloInstruction* inst) {
  auto gru_inst = Cast<HloRNNInstruction>(inst);

  const auto input_shape = inst->operand(0)->shape();
  const auto time_steps = input_shape.dimensions(0);
  const auto batch_size = input_shape.dimensions(1);
  auto optional_input_size = convert_scalar<uint32>(input_shape.dimensions(2));
  if (!optional_input_size) {
    return xla::FailedPrecondition(
        "GRU - Input size cannot be interpreted as an unsigned integer.");
  }
  const auto input_size = *optional_input_size;

  auto optional_num_channels = convert_scalar<uint32>(gru_inst->num_channels());
  if (!optional_num_channels) {
    return xla::FailedPrecondition(
        "GRU - Num Channels cannot be interpreted as an unsigned integer.");
  }
  const auto num_channels = *optional_num_channels;

  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(input_shape));
  popnn::gru::GruParams gru_params(type, batch_size, time_steps,
                                   {input_size, num_channels});

  gru_params.calcInputGradients = gru_inst->is_training();
  return gru_params;
}

StatusOr<poplar::OptionFlags> GetGruOpts(const HloInstruction* inst) {
  auto gru_inst = Cast<HloRNNInstruction>(inst);

  poplar::OptionFlags gru_opts;
  bool is_training = gru_inst->is_training();
  if (!is_training) {
    gru_opts.set({{"inferenceOnly", "true"}});
  }

  // Get the partial type.
  xla::PrimitiveType partials_xla_type = gru_inst->partials_type();
  TF_ASSIGN_OR_RETURN(poplar::Type partials_poplar_type,
                      PoplarDataType(partials_xla_type));
  gru_opts.set({{"partialsType", partials_poplar_type.toString()}});
  return gru_opts;
}

poplar::Tensor UnflattenWeight(const poplar::Tensor& t) {
  return t
      .reshape({t.dim(0), basic_gru_cell_num_units,
                t.dim(1) / basic_gru_cell_num_units})
      .dimShuffle({1, 0, 2});
}

// The kernel is stored as:
// [input_size + output_size, basic_gru_cell_num_units * output_size] tensor.
// This extracts the input and output weights.
std::pair<poplar::Tensor, poplar::Tensor> UnpackGruKernel(
    poplar::Tensor kernel, const size_t input_size, const size_t output_size) {
  poplar::Tensor inputWeights = UnflattenWeight(kernel.slice(0, input_size));
  poplar::Tensor outputWeights =
      UnflattenWeight(kernel.slice(input_size, input_size + output_size));
  return {inputWeights, outputWeights};
}

poplar::Tensor FlattenWeight(const poplar::Tensor& t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// Reverse of UnpackGruKernel.
poplar::Tensor PackGruKernel(poplar::Tensor input_weights,
                             poplar::Tensor output_weights) {
  return poplar::concat(FlattenWeight(input_weights),
                        FlattenWeight(output_weights));
}

class GRULayerFwdOp : public PoplibsOpDef {
  StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                     CompilerResources& res,
                                     const std::string& name,
                                     const TensorTarget& tensor_target,
                                     const TensorMap& tensor_map) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(popnn::gru::GruParams gru_params,
                        GetGruParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags gru_opts, GetGruOpts(inst));
    switch (input_index) {
      case 0: {
        // Allocate GRU input tensor.
        return popnn::gru::createInput(graph, gru_params, name, gru_opts,
                                       &res.dot_cache);
      }
      case 1: {
        // Allocate initial state tensor.
        return popnn::gru::createInitialState(graph, gru_params, name, gru_opts,
                                              &res.dot_cache);
      }
      case 2: {
        // Allocate GRU weights kernel.
        poplar::Tensor input_weights;
        poplar::Tensor output_weights;
        std::tie(input_weights, output_weights) =
            popnn::gru::createWeightsKernel(graph, gru_params, name, gru_opts,
                                            &res.dot_cache);
        return PackGruKernel(input_weights, output_weights);
      }
      case 3: {
        // Allocate GRU weights biases.
        return popnn::gru::createWeightsBiases(graph, gru_params, name,
                                               gru_opts, &res.dot_cache);
      }
      default: {
        return xla::FailedPrecondition(
            "Trying to allocate GRULayerFwdOp tensor for an index out of "
            "range "
            "%d.",
            input_index);
      }
    }
  }

  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_input_seq,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_input_state,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_kernel,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_biases,
                        FindInstructionInput(tensor_map, res, inst, 3, seq));

    TF_ASSIGN_OR_RETURN(popnn::gru::GruParams gru_params,
                        GetGruParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags gru_opts, GetGruOpts(inst));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    auto gru_inst = Cast<HloRNNInstruction>(inst);
    bool is_training = gru_inst->is_training();

    using namespace poputil::graphfn;
    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, &res, gru_params, gru_opts, is_training, input_size,
                 output_size, debug_prefix](std::vector<poplar::Tensor>& args,
                                            poplar::program::Sequence& prog) {
      poplar::Tensor input_seq = args[0];
      poplar::Tensor input_state = args[1];
      poplar::Tensor kernel = args[2];
      poplar::Tensor biases = args[3];

      popnn::gru::GruWeights weights;
      std::tie(weights.inputWeights, weights.outputWeights) =
          UnpackGruKernel(kernel, input_size, output_size);
      weights.biases = biases;

      auto intermediates_ptr = is_training ? &args[6] : nullptr;
      args[4] = popnn::gru::gruFwd(graph, gru_params, input_state, input_seq,
                                   weights, intermediates_ptr, prog,
                                   debug_prefix, gru_opts, &res.dot_cache);
      args[5] = poputil::duplicate(graph, args[4][gru_params.timeSteps - 1],
                                   prog, debug_prefix + "/outputHState");
    };

    poplar::Tensor output, output_state, intermediates;
    std::vector<poplar::Tensor> args = {arg_input_seq, arg_input_state,
                                        arg_kernel,    arg_biases,
                                        output,        output_state};
    Signature signature = {input(arg_input_seq, "input_seq"),
                           input(arg_input_state, "input_state"),
                           input(arg_kernel, "kernel"),
                           input(arg_biases, "biases"),
                           created("output"),
                           created("output_state")};
    if (is_training) {
      args.push_back(intermediates);
      signature.push_back(created("intermediates"));
    }

    TF_RETURN_IF_ERROR(
        res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));

    output = args[4];
    output_state = args[5];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, output_state));
    if (is_training) {
      intermediates = args[6];
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, intermediates));
    }
    return seq;
  }
};
REGISTER_POPLIBS_OP(Popnn, GRULayerFwd, GRULayerFwdOp);

class GRULayerBwdOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_input_seq,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_input_state,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_kernel,
                        FindInstructionInput(tensor_map, res, inst, 2, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_biases,
                        FindInstructionInput(tensor_map, res, inst, 3, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_output,
                        FindInstructionInput(tensor_map, res, inst, 4, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_output_state,
                        FindInstructionInput(tensor_map, res, inst, 5, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_intermediates,
                        FindInstructionInput(tensor_map, res, inst, 6, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_output_backprop,
                        FindInstructionInput(tensor_map, res, inst, 7, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor arg_output_state_backprop,
                        FindInstructionInput(tensor_map, res, inst, 8, seq));

    TF_ASSIGN_OR_RETURN(popnn::gru::GruParams gru_params,
                        GetGruParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags gru_opts, GetGruOpts(inst));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    using namespace poputil::graphfn;
    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, &res, gru_params, gru_opts, input_size, output_size,
                 debug_prefix](std::vector<poplar::Tensor>& args,
                               poplar::program::Sequence& prog) {
      poplar::Tensor input_seq = args[0];
      poplar::Tensor input_state = args[1];
      poplar::Tensor kernel = args[2];
      poplar::Tensor biases = args[3];
      poplar::Tensor output = args[4];
      poplar::Tensor output_state = args[5];
      poplar::Tensor intermediates = args[6];
      poplar::Tensor output_backprop = args[7];
      poplar::Tensor output_state_backprop = args[8];

      popnn::gru::GruWeights weights;
      std::tie(weights.inputWeights, weights.outputWeights) =
          UnpackGruKernel(kernel, input_size, output_size);
      weights.biases = biases;

      popops::addInPlace(graph, output_backprop[output_backprop.dim(0) - 1],
                         output_state_backprop, prog,
                         debug_prefix + "/outputGradient");

      popnn::gru::GruWeights weights_backprop;
      args[10] = popnn::gru::gruBwdWithWU(
          graph, gru_params, prog, input_state, intermediates, weights,
          input_seq, output, output_backprop, &args[9], weights_backprop,
          debug_prefix, gru_opts, &res.dot_cache);
      args[11] = PackGruKernel(weights_backprop.inputWeights,
                               weights_backprop.outputWeights);
      args[12] = weights_backprop.biases;
    };

    poplar::Tensor input_backprop, input_state_backprop, kernel_backprop,
        biases_backprop;
    std::vector<poplar::Tensor> args = {
        arg_input_seq,     arg_input_state,      arg_kernel,
        arg_biases,        arg_output,           arg_output_state,
        arg_intermediates, arg_output_backprop,  arg_output_state_backprop,
        input_backprop,    input_state_backprop, kernel_backprop,
        biases_backprop};
    Signature signature = {
        input(arg_input_seq, "input_seq"),
        input(arg_input_state, "input_state"),
        input(arg_kernel, "kernel"),
        input(arg_biases, "biases"),
        input(arg_output, "output"),
        input(arg_output_state, "output_state"),
        input(arg_intermediates, "intermediates"),
        input(arg_output_backprop, "output_backprop"),
        input(arg_output_state_backprop, "output_state_backprop"),
        created("input_backprop"),
        created("input_state_backprop"),
        created("kernel_backprop"),
        created("biases_backprop")};

    TF_RETURN_IF_ERROR(
        res.graph_cache.ExecuteCached(inst, graph, seq, func, signature, args));

    input_backprop = args[9];
    input_state_backprop = args[10];
    kernel_backprop = args[11];
    biases_backprop = args[12];
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, input_state_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, kernel_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 3, biases_backprop));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Popnn, GRULayerBwd, GRULayerBwdOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
