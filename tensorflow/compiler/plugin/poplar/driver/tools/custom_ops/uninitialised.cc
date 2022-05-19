/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/uninitialised.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloUninitialisedInstruction::HloUninitialisedInstruction(const Shape& shape,
                                                         int64_t& identifier)
    : HloPoplarInstruction(shape, {}, PoplarOp::Uninitialised, identifier),
      identifier_(identifier++) {}

HloPoplarUseDescriptions HloUninitialisedInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloUninitialisedInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloUninitialisedInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

std::unique_ptr<HloInstruction> CreateUninitialisedInstruction(
    const Shape& shape, int64_t& identifier) {
  return absl::make_unique<HloUninitialisedInstruction>(shape, identifier);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>> HloUnitialisedInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  TF_ASSIGN_OR_RETURN(int64_t identifier,
                      attribute_map.GetAttributeAsInt("identifier"));
  return CreateUninitialisedInstruction(call->shape(), identifier);
}

static HloPoplarInstructionFactory unintialised_factory(
    PoplarOp::Uninitialised, HloUnitialisedInstructionFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
