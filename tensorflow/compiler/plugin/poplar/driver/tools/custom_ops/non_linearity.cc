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

namespace {
template <class NonLinearityInstruction>
std::unique_ptr<HloInstruction> CreateNonLinearity(
    HloInstruction* const operand) {
  return absl::make_unique<NonLinearityInstruction>(operand);
}

template <class NonLinearityGradInstruction>
std::unique_ptr<HloInstruction> CreateNonLinearityGrad(
    HloInstruction* const out, HloInstruction* const grad) {
  return absl::make_unique<NonLinearityGradInstruction>(out, grad);
}
}  // namespace
// Relu
std::unique_ptr<HloInstruction> CreateRelu(HloInstruction* const operand) {
  return CreateNonLinearity<HloReluInstruction>(operand);
}

// Gelu
std::unique_ptr<HloInstruction> CreateGelu(HloInstruction* const operand) {
  return CreateNonLinearity<HloGeluInstruction>(operand);
}

// Sigmoid
std::unique_ptr<HloInstruction> CreateSigmoid(HloInstruction* const operand) {
  return CreateNonLinearity<HloSigmoidInstruction>(operand);
}

// HardSigmoid
std::unique_ptr<HloInstruction> CreateHardSigmoid(
    HloInstruction* const operand) {
  return CreateNonLinearity<HloHardSigmoidInstruction>(operand);
}

// Swish
std::unique_ptr<HloInstruction> CreateSwish(HloInstruction* const operand) {
  return CreateNonLinearity<HloSwishInstruction>(operand);
}

// ReluGrad
std::unique_ptr<HloInstruction> CreateReluGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloReluGradInstruction>(out, grad);
}

// GeluGrad
std::unique_ptr<HloInstruction> CreateGeluGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloGeluGradInstruction>(out, grad);
}

// SigmoidGrad
std::unique_ptr<HloInstruction> CreateSigmoidGrad(HloInstruction* const out,
                                                  HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloSigmoidGradInstruction>(out, grad);
}

// HardSigmoidGrad
std::unique_ptr<HloInstruction> CreateHardSigmoidGrad(
    HloInstruction* const out, HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloHardSigmoidGradInstruction>(out, grad);
}

// TanhGrad
std::unique_ptr<HloInstruction> CreateTanhGrad(HloInstruction* const out,
                                               HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloTanhGradInstruction>(out, grad);
}

// SwishGrad
std::unique_ptr<HloInstruction> CreateSwishGrad(HloInstruction* const out,
                                                HloInstruction* const grad) {
  return CreateNonLinearityGrad<HloSwishGradInstruction>(out, grad);
}

namespace {

template <class NonLinearityInstruction>
StatusOr<std::unique_ptr<HloInstruction>> HloNonLinearityFactory(
    HloCustomCallInstruction* call) {
  return CreateNonLinearity<NonLinearityInstruction>(call->mutable_operand(0));
}

static HloPoplarInstructionFactory relu_factory(
    PoplarOp::Relu, HloNonLinearityFactory<HloReluInstruction>);

static HloPoplarInstructionFactory gelu_factory(
    PoplarOp::Gelu, HloNonLinearityFactory<HloGeluInstruction>);

static HloPoplarInstructionFactory sigmoid_factory(
    PoplarOp::Sigmoid, HloNonLinearityFactory<HloSigmoidInstruction>);

static HloPoplarInstructionFactory hard_sigmoid_factory(
    PoplarOp::HardSigmoid, HloNonLinearityFactory<HloHardSigmoidInstruction>);

static HloPoplarInstructionFactory swish_factory(
    PoplarOp::Swish, HloNonLinearityFactory<HloSwishInstruction>);

template <class NonLinearityGradInstruction>
StatusOr<std::unique_ptr<HloInstruction>> HloNonLinearityGradFactory(
    HloCustomCallInstruction* call) {
  return CreateNonLinearityGrad<NonLinearityGradInstruction>(
      call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory relu_grad_factory(
    PoplarOp::ReluGrad, HloNonLinearityGradFactory<HloReluGradInstruction>);

static HloPoplarInstructionFactory gelu_grad_factory(
    PoplarOp::GeluGrad, HloNonLinearityGradFactory<HloGeluGradInstruction>);

static HloPoplarInstructionFactory sigmoid_grad_factory(
    PoplarOp::SigmoidGrad,
    HloNonLinearityGradFactory<HloSigmoidGradInstruction>);

static HloPoplarInstructionFactory hard_sigmoid_grad_factory(
    PoplarOp::HardSigmoidGrad,
    HloNonLinearityGradFactory<HloHardSigmoidGradInstruction>);

static HloPoplarInstructionFactory tanh_grad_factory(
    PoplarOp::TanhGrad, HloNonLinearityGradFactory<HloTanhGradInstruction>);

static HloPoplarInstructionFactory swish_grad_factory(
    PoplarOp::SwishGrad, HloNonLinearityGradFactory<HloSwishGradInstruction>);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
