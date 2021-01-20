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
namespace rnn_helper {
RNNAttributes::RNNAttributes(int32 num_channels, bool is_training,
                             xla::PrimitiveType partials_xla_type)
    : num_channels(num_channels),
      is_training(is_training),
      partials_xla_type(partials_xla_type) {}
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

  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_dtype, &partials_xla_type));
  return RNNAttributes(num_channels, is_training, partials_xla_type);
}
}  // namespace rnn_helper

bool HloRNNInstruction::is_training() const { return is_training_; }
int32 HloRNNInstruction::num_channels() const { return num_channels_; }
xla::PrimitiveType HloRNNInstruction::partials_type() const {
  return partials_type_;
}

std::vector<std::string> HloRNNInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("is_training=" + std::to_string(is_training_));
  attributes.push_back("num_channels=" + std::to_string(num_channels_));
  attributes.push_back("partials_type=" +
                       xla::PrimitiveType_Name(partials_type_));

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
