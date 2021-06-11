/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/rnn_util.h"

#include <string>

#include <popnn/GruDef.hpp>
#include <popnn/LstmDef.hpp>
#include <popnn/NonLinearityDef.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/gru.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/lstm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

StatusOr<popnn::NonLinearityType> convertToNonLinearityType(
    rnn_helper::ActivationType activation) {
  switch (activation) {
    case rnn_helper::ActivationType::SOFTMAX: {
      return popnn::NonLinearityType::SOFTMAX;
    }

    case rnn_helper::ActivationType::RELU: {
      return popnn::NonLinearityType::RELU;
    }

    case rnn_helper::ActivationType::TANH: {
      return popnn::NonLinearityType::TANH;
    }

    case rnn_helper::ActivationType::SIGMOID: {
      return popnn::NonLinearityType::SIGMOID;
    }

    case rnn_helper::ActivationType::HARD_SIGMOID: {
      return popnn::NonLinearityType::HARD_SIGMOID;
    }

    default: { return InvalidArgument("Invalid activation type"); }
  }
}

StatusOr<popnn::lstm::LstmParams> GetLstmParameters(
    const HloInstruction* inst) {
  auto lstm_inst = Cast<HloRNNInstruction>(inst);

  const auto input_shape = inst->operand(0)->shape();
  const auto time_steps = input_shape.dimensions(0);
  const auto batch_size = input_shape.dimensions(1);
  auto optional_input_size = convert_scalar<uint32>(input_shape.dimensions(2));
  if (!optional_input_size) {
    return xla::FailedPrecondition(
        "LSTM - Input size cannot be interpreted as an unsigned integer.");
  }
  const auto input_size = *optional_input_size;

  auto optional_num_channels =
      convert_scalar<uint32>(lstm_inst->num_channels());
  if (!optional_num_channels) {
    return xla::FailedPrecondition(
        "LSTM - Num Channels cannot be interpreted as an unsigned integer.");
  }
  const auto num_channels = *optional_num_channels;

  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(input_shape));
  popnn::lstm::LstmParams lstm_params(type, batch_size, time_steps,
                                      {input_size, num_channels});

  TF_ASSIGN_OR_RETURN(popnn::NonLinearityType activation,
                      convertToNonLinearityType(lstm_inst->activation()));
  TF_ASSIGN_OR_RETURN(
      popnn::NonLinearityType recurrent_activation,
      convertToNonLinearityType(lstm_inst->recurrent_activation()));

  lstm_params.calcInputGradients = lstm_inst->is_training();
  lstm_params.cellOrder = {
      BASIC_LSTM_CELL_INPUT_GATE, BASIC_LSTM_CELL_FORGET_GATE,
      BASIC_LSTM_CELL_CANDIDATE, BASIC_LSTM_CELL_OUTPUT_GATE};
  lstm_params.activation = activation;
  lstm_params.recurrentActivation = recurrent_activation;
  lstm_params.outputFullSequence = lstm_inst->output_full_sequence();
  return lstm_params;
}

StatusOr<poplar::OptionFlags> GetLstmOpts(const HloInstruction* inst,
                                          const CompilerResources& res) {
  auto lstm_inst = Cast<HloRNNInstruction>(inst);

  // Initialize options from matmul options
  poplar::OptionFlags lstm_opts = res.default_matmul_options;
  bool is_training = lstm_inst->is_training();
  if (!is_training) {
    lstm_opts.set({{"inferenceOnly", "true"}});
  }

  // Get the partial type
  xla::PrimitiveType partials_xla_type = lstm_inst->partials_type();
  TF_ASSIGN_OR_RETURN(poplar::Type partials_poplar_type,
                      PoplarDataType(partials_xla_type));
  lstm_opts.set({{"partialsType", partials_poplar_type.toString()}});
  return lstm_opts;
}

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

  TF_ASSIGN_OR_RETURN(popnn::NonLinearityType activation,
                      convertToNonLinearityType(gru_inst->activation()));
  TF_ASSIGN_OR_RETURN(
      popnn::NonLinearityType recurrent_activation,
      convertToNonLinearityType(gru_inst->recurrent_activation()));

  gru_params.calcInputGradients = gru_inst->is_training();
  gru_params.cellOrder = {BASIC_GRU_CELL_UPDATE_GATE, BASIC_GRU_CELL_RESET_GATE,
                          BASIC_GRU_CELL_CANDIDATE};
  gru_params.activation = activation;
  gru_params.recurrentActivation = recurrent_activation;

  gru_params.outputFullSequence = gru_inst->output_full_sequence();

  const HloGRUInstructionCommon* gru_common_inst;
  if (IsPoplarInstruction(PoplarOp::GRULayerFwd)(inst)) {
    gru_common_inst = Cast<HloGRUFwdInstruction>(inst);
  } else if (IsPoplarInstruction(PoplarOp::GRULayerBwd)(inst)) {
    gru_common_inst = Cast<HloGRUBwdInstruction>(inst);
  } else if (IsPoplarInstruction(PoplarOp::DynamicGRULayerFwd)(inst)) {
    gru_common_inst = Cast<HloDynamicGRUFwdInstruction>(inst);
  } else if (IsPoplarInstruction(PoplarOp::DynamicGRULayerBwd)(inst)) {
    gru_common_inst = Cast<HloDynamicGRUBwdInstruction>(inst);
  } else if (IsPoplarInstruction(PoplarOp::AUGRULayerFwd)(inst)) {
    gru_common_inst = Cast<HloAUGRUFwdInstruction>(inst);
  } else if (IsPoplarInstruction(PoplarOp::AUGRULayerBwd)(inst)) {
    gru_common_inst = Cast<HloAUGRUBwdInstruction>(inst);
  }
  gru_params.resetAfter = gru_common_inst->reset_after();

  return gru_params;
}

StatusOr<poplar::OptionFlags> GetGruOpts(const HloInstruction* inst,
                                         const CompilerResources& res) {
  auto gru_inst = Cast<HloRNNInstruction>(inst);

  poplar::OptionFlags gru_opts = res.default_matmul_options;
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

}  // namespace poplarplugin
}  // namespace xla
