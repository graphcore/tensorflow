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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/print_tensor.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloPrintTensor::HloPrintTensor(HloInstruction* input,
                               const std::string& tensor_name)
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {input},
                           PoplarOp::PrintTensor),
      tensor_name_(tensor_name) {
  set_custom_call_has_side_effect(true);
}

absl::flat_hash_set<int64_t> HloPrintTensor::AllocatingIndices() const {
  return {};
}

bool HloPrintTensor::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t> HloPrintTensor::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloPrintTensor::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloPrintTensor::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloPrintTensor::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloPrintTensor::AllowNonInplaceLowering() const { return false; }

bool HloPrintTensor::IsPopOpsElementwise() const { return false; }

const std::string& HloPrintTensor::TensorName() const { return tensor_name_; }

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateHloPrintTensor(
    HloInstruction* input, const std::string& tensor_name) {
  return absl::make_unique<HloPrintTensor>(input, tensor_name);
}

std::unique_ptr<HloInstruction> HloPrintTensor::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloPrintTensor(operands[0], tensor_name_);
}

std::vector<std::string> HloPrintTensor::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("tensor_name=", tensor_name_));
  return attributes;
}

namespace {

static HloPoplarInstructionFactory print_tensor_factory(
    PoplarOp::PrintTensor,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      auto attribute = attribute_map.GetAttributeAsString("tensor_name");
      std::string tensor_name = attribute.ok() ? attribute.ValueOrDie() : "";
      return CreateHloPrintTensor(call->mutable_operand(0), tensor_name);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
