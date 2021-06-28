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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/codelet_expression_op.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloCodeletExpressionOpInstruction::HloCodeletExpressionOpInstruction(
    const xla::Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::string& source)
    : HloPoplarInstruction(shape, operands, PoplarOp::CodeletExpressionOp),
      source_(source) {}

const std::string& HloCodeletExpressionOpInstruction::source() const {
  return source_;
}

absl::flat_hash_set<int64>
HloCodeletExpressionOpInstruction::AllocatingIndices() const {
  return {};
}

bool HloCodeletExpressionOpInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloCodeletExpressionOpInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCodeletExpressionOpInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloCodeletExpressionOpInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloCodeletExpressionOpInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCodeletExpressionOpInstruction::IsPopOpsElementwise() const {
  return true;
}

std::vector<std::string>
HloCodeletExpressionOpInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("source=", source_));

  return attributes;
}

std::unique_ptr<HloInstruction>
HloCodeletExpressionOpInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloCodeletExpressionOpInstruction>(
      shape, new_operands, source_);
}
std::unique_ptr<HloInstruction> CreateCodeletExpressionOp(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::string& source) {
  return absl::make_unique<HloCodeletExpressionOpInstruction>(shape, operands,
                                                              source);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloCodeletExpressionOpInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(std::string source,
                      attribute_map.GetAttributeAsString("source"));

  return CreateCodeletExpressionOp(call->shape(), call->operands(), source);
}

static HloPoplarInstructionFactory codelet_expression_op_factory(
    PoplarOp::CodeletExpressionOp,
    HloCodeletExpressionOpInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
