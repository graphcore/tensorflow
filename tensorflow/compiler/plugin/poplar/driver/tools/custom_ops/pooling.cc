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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace poplarplugin {

HloPoolingInstruction::HloPoolingInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands, PoplarOp op,
    xla::Window window)
    : HloPoplarInstruction(shape, operands, op, window_util::ToString(window)),
      window_(window) {}

const xla::Window& HloPoolingInstruction::window() const { return window_; }

absl::flat_hash_set<int64_t> HloPoolingInstruction::AllocatingIndices() const {
  return {};
}

bool HloPoolingInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64_t, int64_t>
HloPoolingInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloPoolingInstruction::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloPoolingInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloPoolingInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloPoolingInstruction::AllowNonInplaceLowering() const { return false; }

bool HloPoolingInstruction::IsPopOpsElementwise() const { return false; }

std::vector<std::string>
HloPoolingInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back(absl::StrCat("window=", window_util::ToString(window_)));

  return attributes;
}

HloMaxPoolInstruction::HloMaxPoolInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             xla::Window window)
    : HloPoolingInstruction(shape, {operand}, PoplarOp::MaxPool, window) {}

const HloInstruction* HloMaxPoolInstruction::to_reduce() const {
  return operand(0);
}

HloInstruction* HloMaxPoolInstruction::mutable_to_reduce() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction> HloMaxPoolInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateMaxPool(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateMaxPool(const Shape& shape,
                                              HloInstruction* operand,
                                              xla::Window window) {
  return absl::make_unique<HloMaxPoolInstruction>(shape, operand, window);
}

HloAvgPoolInstruction::HloAvgPoolInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             xla::Window window)
    : HloPoolingInstruction(shape, {operand}, PoplarOp::AvgPool, window) {}

const HloInstruction* HloAvgPoolInstruction::to_reduce() const {
  return operand(0);
}

HloInstruction* HloAvgPoolInstruction::mutable_to_reduce() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction> HloAvgPoolInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateAvgPool(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateAvgPool(const Shape& shape,
                                              HloInstruction* operand,
                                              xla::Window window) {
  return absl::make_unique<HloAvgPoolInstruction>(shape, operand, window);
}

HloMaxPoolGradInstruction::HloMaxPoolGradInstruction(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* output_grad, xla::Window window)
    : HloPoolingInstruction(shape, {input, output, output_grad},
                            PoplarOp::MaxPoolGrad, window) {}

const HloInstruction* HloMaxPoolGradInstruction::input() const {
  return operand(0);
}
const HloInstruction* HloMaxPoolGradInstruction::output() const {
  return operand(1);
}
const HloInstruction* HloMaxPoolGradInstruction::output_grad() const {
  return operand(2);
}

HloInstruction* HloMaxPoolGradInstruction::mutable_input() {
  return mutable_operand(0);
}
HloInstruction* HloMaxPoolGradInstruction::mutable_output() {
  return mutable_operand(1);
}
HloInstruction* HloMaxPoolGradInstruction::mutable_output_grad() {
  return mutable_operand(2);
}

std::unique_ptr<HloInstruction>
HloMaxPoolGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateMaxPoolGrad(shape, operands[0], operands[1], operands[2],
                           window());
}

std::unique_ptr<HloInstruction> CreateMaxPoolGrad(const Shape& shape,
                                                  HloInstruction* input,
                                                  HloInstruction* output,
                                                  HloInstruction* output_grad,
                                                  xla::Window window) {
  return absl::make_unique<HloMaxPoolGradInstruction>(shape, input, output,
                                                      output_grad, window);
}

HloAvgPoolGradInstruction::HloAvgPoolGradInstruction(
    const Shape& shape, HloInstruction* output_grad, xla::Window window)
    : HloPoolingInstruction(shape, {output_grad}, PoplarOp::AvgPoolGrad,
                            window) {}

const HloInstruction* HloAvgPoolGradInstruction::output_grad() const {
  return operand(0);
}
HloInstruction* HloAvgPoolGradInstruction::mutable_output_grad() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction>
HloAvgPoolGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateAvgPoolGrad(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateAvgPoolGrad(const Shape& shape,
                                                  HloInstruction* output_grad,
                                                  xla::Window window) {
  return absl::make_unique<HloAvgPoolGradInstruction>(shape, output_grad,
                                                      window);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> MaxPoolFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateMaxPool(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory max_pool_factory(PoplarOp::MaxPool,
                                                    MaxPoolFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> AvgPoolFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateAvgPool(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory avg_pool_factory(PoplarOp::AvgPool,
                                                    AvgPoolFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> MaxPoolGradFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateMaxPoolGrad(call->shape(), call->mutable_operand(0),
                           call->mutable_operand(1), call->mutable_operand(2),
                           window);
}

static HloPoplarInstructionFactory max_pool_grad_factory(
    PoplarOp::MaxPoolGrad, MaxPoolGradFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> AvgPoolGradFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateAvgPoolGrad(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory avg_pool_grad_factory(
    PoplarOp::AvgPoolGrad, AvgPoolGradFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
