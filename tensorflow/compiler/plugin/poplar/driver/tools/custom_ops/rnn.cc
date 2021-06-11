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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {
using ActivationType = rnn_helper::ActivationType;

namespace rnn_helper {

namespace {

StatusOr<ActivationType> strToActivationType(const std::string name) {
  if (name == "softmax") {
    return ActivationType::SOFTMAX;
  } else if (name == "relu") {
    return ActivationType::RELU;
  } else if (name == "tanh") {
    return ActivationType::TANH;
  } else if (name == "sigmoid") {
    return ActivationType::SIGMOID;
  } else if (name == "hard_sigmoid") {
    return ActivationType::HARD_SIGMOID;
  } else {
    return InvalidArgument("Invalid activation type");
  }
}
}  // namespace

RNNAttributes::RNNAttributes(int32 num_channels, bool is_training,
                             xla::PrimitiveType partials_xla_type,
                             ActivationType activation,
                             ActivationType recurrent_activation,
                             bool output_full_sequence)
    : num_channels(num_channels),
      is_training(is_training),
      partials_xla_type(partials_xla_type),
      activation(activation),
      recurrent_activation(recurrent_activation),
      output_full_sequence(output_full_sequence) {}
// Helper for parsing the attribute map when converting the custom call
// instruction.
StatusOr<RNNAttributes> RNNAttributes::Parse(
    const HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));

  TF_ASSIGN_OR_RETURN(tensorflow::DataType partials_dtype,
                      attribute_map.GetAttributeAsTFDataType("partials_dtype"));

  TF_ASSIGN_OR_RETURN(std::string activation_string,
                      attribute_map.GetAttributeAsString("activation"));

  TF_ASSIGN_OR_RETURN(
      std::string recurrent_activation_string,
      attribute_map.GetAttributeAsString("recurrent_activation"));

  TF_ASSIGN_OR_RETURN(bool output_full_sequence,
                      attribute_map.GetAttributeAsBool("output_full_sequence"));

  TF_ASSIGN_OR_RETURN(ActivationType activation,
                      strToActivationType(activation_string));

  TF_ASSIGN_OR_RETURN(ActivationType recurrent_activation,
                      strToActivationType(recurrent_activation_string));

  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_dtype, &partials_xla_type));
  return RNNAttributes(num_channels, is_training, partials_xla_type, activation,
                       recurrent_activation, output_full_sequence);
}
}  // namespace rnn_helper

bool HloRNNInstruction::is_training() const { return is_training_; }
ActivationType HloRNNInstruction::activation() const { return activation_; }
ActivationType HloRNNInstruction::recurrent_activation() const {
  return recurrent_activation_;
}
int32 HloRNNInstruction::num_channels() const { return num_channels_; }
xla::PrimitiveType HloRNNInstruction::partials_type() const {
  return partials_type_;
}
bool HloRNNInstruction::output_full_sequence() const {
  return output_full_sequence_;
}

std::vector<std::string> HloRNNInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("is_training=" + std::to_string(is_training_));
  attributes.push_back("num_channels=" + std::to_string(num_channels_));
  attributes.push_back("partials_type=" +
                       xla::PrimitiveType_Name(partials_type_));
  attributes.push_back("output_full_sequence=" +
                       std::to_string(output_full_sequence_));

  return attributes;
}

absl::flat_hash_map<int64, int64> HloRNNFwdInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloRNNFwdInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloRNNFwdInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloRNNFwdInstruction::IsPopOpsElementwise() const { return false; }

absl::flat_hash_set<int64> HloRNNBwdInstruction::AllocatingIndices() const {
  return {};
}

bool HloRNNBwdInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64> HloRNNBwdInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloRNNBwdInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloRNNBwdInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

bool HloRNNBwdInstruction::IsPopOpsElementwise() const { return false; }

}  // namespace poplarplugin
}  // namespace xla
