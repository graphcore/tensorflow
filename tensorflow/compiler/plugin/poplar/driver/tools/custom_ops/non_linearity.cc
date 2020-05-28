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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/non_linearity.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Relu
std::unique_ptr<HloInstruction> HloReluInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReluInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateRelu(HloInstruction* const operand) {
  return absl::make_unique<HloReluInstruction>(operand);
}

// Gelu
std::unique_ptr<HloInstruction> HloGeluInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGeluInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateGelu(HloInstruction* const operand) {
  return absl::make_unique<HloGeluInstruction>(operand);
}

// Sigmoid
std::unique_ptr<HloInstruction> HloSigmoidInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSigmoidInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateSigmoid(HloInstruction* const operand) {
  return absl::make_unique<HloSigmoidInstruction>(operand);
}

// Softmax
std::unique_ptr<HloInstruction> HloSoftmaxInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSoftmaxInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateSoftmax(HloInstruction* const operand) {
  return absl::make_unique<HloSoftmaxInstruction>(operand);
}

// ReluGrad
std::unique_ptr<HloInstruction>
HloReluGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReluGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateReluGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return absl::make_unique<HloReluGradInstruction>(out, grad);
}

// GeluGrad
std::unique_ptr<HloInstruction>
HloGeluGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGeluGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateGeluGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return absl::make_unique<HloGeluGradInstruction>(out, grad);
}

// SigmoidGrad
std::unique_ptr<HloInstruction>
HloSigmoidGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSigmoidGradInstruction>(new_operands[0],
                                                      new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateSigmoidGrad(HloInstruction* const out,
                                                  HloInstruction* const grad) {
  return absl::make_unique<HloSigmoidGradInstruction>(out, grad);
}

// TanhGrad
std::unique_ptr<HloInstruction>
HloTanhGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloTanhGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateTanhGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return absl::make_unique<HloTanhGradInstruction>(out, grad);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>> HloReluInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRelu(call->mutable_operand(0));
}

static HloPoplarInstructionFactory relu_factory(PoplarOp::Relu,
                                                HloReluInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGeluInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateGelu(call->mutable_operand(0));
}

static HloPoplarInstructionFactory gelu_factory(PoplarOp::Gelu,
                                                HloGeluInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloSigmoidInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateSigmoid(call->mutable_operand(0));
}

static HloPoplarInstructionFactory sigmoid_factory(
    PoplarOp::Sigmoid, HloSigmoidInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloSoftmaxInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateSoftmax(call->mutable_operand(0));
}

static HloPoplarInstructionFactory softmax_factory(
    PoplarOp::Softmax, HloSoftmaxInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloReluGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateReluGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory relu_grad_factory(
    PoplarOp::ReluGrad, HloReluGradInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloGeluGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateGeluGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory gelu_grad_factory(
    PoplarOp::GeluGrad, HloGeluGradInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloSigmoidGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateSigmoidGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory sigmoid_grad_factory(
    PoplarOp::SigmoidGrad, HloSigmoidGradInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloTanhGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateTanhGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory tanh_grad_factory(
    PoplarOp::TanhGrad, HloTanhGradInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
