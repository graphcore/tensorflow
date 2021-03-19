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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/lstm.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/rnn_util.h"
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

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Util.hpp>

namespace xla {
namespace poplarplugin {
namespace {
static const size_t basic_lstm_cell_num_units = 4;

poplar::Tensor UnflattenWeight(const poplar::Tensor& t) {
  return t
      .reshape({t.dim(0), basic_lstm_cell_num_units,
                t.dim(1) / basic_lstm_cell_num_units})
      .dimShuffle({1, 0, 2});
}

// The kernel is stored as:
// [input_size + output_size, basic_lstm_cell_num_units * output_size] tensor.
// This extracts the input and output weights.
std::pair<poplar::Tensor, poplar::Tensor> UnpackLstmKernel(
    poplar::Tensor kernel, const size_t input_size, const size_t output_size) {
  poplar::Tensor inputWeights = UnflattenWeight(kernel.slice(0, input_size));
  poplar::Tensor outputWeights =
      UnflattenWeight(kernel.slice(input_size, input_size + output_size));
  return {inputWeights, outputWeights};
}

poplar::Tensor FlattenWeight(const poplar::Tensor& t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// Reverse of UnpackLstmKernel
poplar::Tensor PackLstmKernel(poplar::Tensor input_weights,
                              poplar::Tensor output_weights) {
  return poplar::concat(FlattenWeight(input_weights),
                        FlattenWeight(output_weights));
}

class LstmLayerFwdOp : public PoplarOpDef {
  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "LstmLayerFwdOp");
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;

    TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                        GetLstmParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts, GetLstmOpts(inst, res));
    switch (input_index) {
      case 0: {
        // Allocate LSTM input tensor
        return popnn::lstm::createInput(graph, lstm_params, {debug_info},
                                        lstm_opts, &res.matmul_cache);
      }
      case 1: {
        // Allocate initial output (h) tensor
        return popnn::lstm::createInitialOutput(
            graph, lstm_params, {debug_info}, lstm_opts, &res.matmul_cache);
      }
      case 2: {
        // Allocate initial cell state (c) tensor
        return popnn::lstm::createInitialCellState(
            graph, lstm_params, {debug_info}, lstm_opts, &res.matmul_cache);
      }
      case 3: {
        // Allocate LSTM weights kernel
        poplar::Tensor input_weights;
        poplar::Tensor output_weights;
        std::tie(input_weights, output_weights) =
            popnn::lstm::createWeightsKernel(graph, lstm_params, {debug_info},
                                             lstm_opts, &res.matmul_cache);
        return PackLstmKernel(input_weights, output_weights);
      }
      case 4: {
        // Allocate LSTM weights biases
        return popnn::lstm::createWeightsBiases(
            graph, lstm_params, {debug_info}, lstm_opts, &res.matmul_cache);
      }
      default: {
        return xla::FailedPrecondition(
            "Trying to allocate LstmLayerFwdOp tensor for an index out of "
            "range "
            "%d.",
            input_index);
      }
    }
  }

  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "LstmLayerFwdOp");
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_seq,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_h_state,
        FindInstructionInput(tensor_map, res, inst, 1, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_c_state,
        FindInstructionInput(tensor_map, res, inst, 2, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_kernel,
        FindInstructionInput(tensor_map, res, inst, 3, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_biases,
        FindInstructionInput(tensor_map, res, inst, 4, seq, debug_info,
                             /*expand_aliasing*/ false));

    TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                        GetLstmParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts, GetLstmOpts(inst, res));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    auto lstm_inst = Cast<HloRNNInstruction>(inst);
    bool is_training = lstm_inst->is_training();

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, lstm_params, lstm_opts, is_training, input_size,
                 output_size,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) {
      poplar::Tensor input_seq = args[0];
      poplar::Tensor input_h_state = args[1];
      poplar::Tensor input_c_state = args[2];
      poplar::Tensor kernel = args[3];
      poplar::Tensor biases = args[4];

      popnn::lstm::LstmWeights weights;
      std::tie(weights.inputWeights, weights.outputWeights) =
          UnpackLstmKernel(kernel, input_size, output_size);
      weights.biases = biases;
      popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

      auto intermediates_ptr = is_training ? &args[8] : nullptr;
      std::tie(args[5], args[7]) = popnn::lstm::lstmFwd(
          graph, lstm_params, init_state, input_seq, weights, intermediates_ptr,
          prog, {debug_name_and_id}, lstm_opts, &res.matmul_cache);
      args[6] =
          poputil::duplicate(graph, args[5][lstm_params.rnn.timeSteps - 1],
                             prog, {debug_name_and_id, "outputHState"});
    };

    poplar::Tensor output, output_h_state, output_c_state, intermediates;
    std::vector<poplar::Tensor> args = {arg_input_seq,     arg_input_h_state,
                                        arg_input_c_state, arg_kernel,
                                        arg_biases,        output,
                                        output_h_state,    output_c_state};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(arg_input_seq, "input_seq"),
        poputil::graphfn::input(arg_input_h_state, "input_h_state"),
        poputil::graphfn::input(arg_input_c_state, "input_c_state"),
        poputil::graphfn::input(arg_kernel, "kernel"),
        poputil::graphfn::input(arg_biases, "biases"),
        poputil::graphfn::created("output"),
        poputil::graphfn::created("output_h_state"),
        poputil::graphfn::created("output_c_state")};
    if (is_training) {
      args.push_back(intermediates);
      signature.push_back(poputil::graphfn::created("intermediates"));
    }

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args,
        lstm_inst->AllocatingIndices(), lstm_inst->LayoutDependencies()));

    output = args[5];
    output_h_state = args[6];
    output_c_state = args[7];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, output_h_state));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, output_c_state));
    if (is_training) {
      intermediates = args[8];
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 3, intermediates));
    }
    return seq;
  }
};
REGISTER_POPLAR_OP(LstmLayerFwd, LstmLayerFwdOp);

class LstmLayerBwdOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "LstmLayerBwdOp");
    poplar::program::Sequence seq({}, debug_info);

    // Do not expand aliasing when creating a cached op - the input will be
    // reallocated if required.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_seq,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_h_state,
        FindInstructionInput(tensor_map, res, inst, 1, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_input_c_state,
        FindInstructionInput(tensor_map, res, inst, 2, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_kernel,
        FindInstructionInput(tensor_map, res, inst, 3, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_biases,
        FindInstructionInput(tensor_map, res, inst, 4, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output,
        FindInstructionInput(tensor_map, res, inst, 5, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output_h_state,
        FindInstructionInput(tensor_map, res, inst, 6, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output_c_state,
        FindInstructionInput(tensor_map, res, inst, 7, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_intermediates,
        FindInstructionInput(tensor_map, res, inst, 8, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output_backprop,
        FindInstructionInput(tensor_map, res, inst, 9, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output_h_state_backprop,
        FindInstructionInput(tensor_map, res, inst, 10, seq, debug_info,
                             /*expand_aliasing*/ false));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor arg_output_c_state_backprop,
        FindInstructionInput(tensor_map, res, inst, 11, seq, debug_info,
                             /*expand_aliasing*/ false));

    TF_ASSIGN_OR_RETURN(popnn::lstm::LstmParams lstm_params,
                        GetLstmParameters(inst));
    TF_ASSIGN_OR_RETURN(poplar::OptionFlags lstm_opts, GetLstmOpts(inst, res));

    auto input_size = ShapeUtil::GetDimension(inst->operand(0)->shape(), 2);
    auto output_size = ShapeUtil::GetDimension(inst->operand(1)->shape(), 1);

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    auto func = [&graph, &res, lstm_params, lstm_opts, input_size, output_size,
                 debug_name_and_id](std::vector<poplar::Tensor>& args,
                                    poplar::program::Sequence& prog) {
      poplar::Tensor input_seq = args[0];
      poplar::Tensor input_h_state = args[1];
      poplar::Tensor input_c_state = args[2];
      poplar::Tensor kernel = args[3];
      poplar::Tensor biases = args[4];
      poplar::Tensor output = args[5];
      poplar::Tensor output_h_state = args[6];
      poplar::Tensor output_c_state = args[7];
      poplar::Tensor intermediates = args[8];
      poplar::Tensor output_backprop = args[9];
      poplar::Tensor output_h_state_backprop = args[10];
      poplar::Tensor output_c_state_backprop = args[11];

      popnn::lstm::LstmWeights weights;
      std::tie(weights.inputWeights, weights.outputWeights) =
          UnpackLstmKernel(kernel, input_size, output_size);
      weights.biases = biases;

      popnn::lstm::LstmState init_state = {input_h_state, input_c_state};

      popops::addInPlace(graph, output_backprop[output_backprop.dim(0) - 1],
                         output_h_state_backprop, prog,
                         {debug_name_and_id, "outputGradient"});

      popnn::lstm::LstmWeights weights_backprop;
      popnn::lstm::LstmState init_state_backprop = popnn::lstm::lstmBwdWithWU(
          graph, lstm_params, prog, init_state, intermediates, weights,
          input_seq, output, output_backprop, &output_c_state_backprop,
          &args[12], weights_backprop, {debug_name_and_id}, lstm_opts,
          &res.matmul_cache);
      args[13] = init_state_backprop.output;
      args[14] = init_state_backprop.cellState;
      args[15] = PackLstmKernel(weights_backprop.inputWeights,
                                weights_backprop.outputWeights);
      args[16] = weights_backprop.biases;
    };

    poplar::Tensor input_backprop, input_h_state_backprop,
        input_c_state_backprop, kernel_backprop, biases_backprop;
    std::vector<poplar::Tensor> args = {arg_input_seq,
                                        arg_input_h_state,
                                        arg_input_c_state,
                                        arg_kernel,
                                        arg_biases,
                                        arg_output,
                                        arg_output_h_state,
                                        arg_output_c_state,
                                        arg_intermediates,
                                        arg_output_backprop,
                                        arg_output_h_state_backprop,
                                        arg_output_c_state_backprop,
                                        input_backprop,
                                        input_h_state_backprop,
                                        input_c_state_backprop,
                                        kernel_backprop,
                                        biases_backprop};
    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(arg_input_seq, "input_seq"),
        poputil::graphfn::input(arg_input_h_state, "input_h_state"),
        poputil::graphfn::input(arg_input_c_state, "input_c_state"),
        poputil::graphfn::input(arg_kernel, "kernel"),
        poputil::graphfn::input(arg_biases, "biases"),
        poputil::graphfn::input(arg_output, "output"),
        poputil::graphfn::input(arg_output_h_state, "output_h_state"),
        poputil::graphfn::input(arg_output_c_state, "output_c_state"),
        poputil::graphfn::input(arg_intermediates, "intermediates"),
        poputil::graphfn::input(arg_output_backprop, "output_backprop"),
        poputil::graphfn::input(arg_output_h_state_backprop,
                                "output_h_state_backprop"),
        poputil::graphfn::input(arg_output_c_state_backprop,
                                "output_c_state_backprop"),
        poputil::graphfn::created("input_backprop"),
        poputil::graphfn::created("input_h_state_backprop"),
        poputil::graphfn::created("input_c_state_backprop"),
        poputil::graphfn::created("kernel_backprop"),
        poputil::graphfn::created("biases_backprop")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(inst, graph, res, seq,
                                                     func, signature, args));

    input_backprop = args[12];
    input_h_state_backprop = args[13];
    input_c_state_backprop = args[14];
    kernel_backprop = args[15];
    biases_backprop = args[16];
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, input_h_state_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, input_c_state_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 3, kernel_backprop));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 4, biases_backprop));
    return seq;
  }
};
REGISTER_POPLAR_OP(LstmLayerBwd, LstmLayerBwdOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
