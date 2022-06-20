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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NON_LINEARITY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NON_LINEARITY_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

template <PoplarOp Op>
class HloNonLinearity : public HloPoplarInstruction {
 public:
  explicit HloNonLinearity(HloInstruction* const operand)
      : HloPoplarInstruction(operand->shape(), {operand}, Op) {}

  absl::flat_hash_set<int64_t> AllocatingIndices() const override { return {}; }
  bool AllocatingOutput() const override { return false; }
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }

  HloPoplarUseDescriptions GetUseDescriptions() const override {
    return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
  }

  HloPoplarBufferDescriptions GetBufferDescriptions() const override {
    return BufferDescriptionsNoAllocations();
  }

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const {
    return FindConsumersExtensionResults();
  }

  bool AllowNonInplaceLowering() const override { return true; }

  bool IsPopOpsElementwise() const override { return true; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override {
    return absl::make_unique<HloNonLinearity<Op>>(new_operands[0]);
  }
};

// Relu
using HloReluInstruction = HloNonLinearity<PoplarOp::Relu>;
std::unique_ptr<HloInstruction> CreateRelu(HloInstruction* const operand);

// Gelu
using HloGeluInstruction = HloNonLinearity<PoplarOp::Gelu>;
std::unique_ptr<HloInstruction> CreateGelu(HloInstruction* const operand);

// Sigmoid
using HloSigmoidInstruction = HloNonLinearity<PoplarOp::Sigmoid>;
std::unique_ptr<HloInstruction> CreateSigmoid(HloInstruction* const operand);

// HardSigmoid
using HloHardSigmoidInstruction = HloNonLinearity<PoplarOp::HardSigmoid>;
std::unique_ptr<HloInstruction> CreateHardSigmoid(
    HloInstruction* const operand);

// Swish
using HloSwishInstruction = HloNonLinearity<PoplarOp::Swish>;
std::unique_ptr<HloInstruction> CreateSwish(HloInstruction* const operand);

// Softmax
using HloSoftmaxInstruction = HloNonLinearity<PoplarOp::Softmax>;
std::unique_ptr<HloInstruction> CreateSoftmax(HloInstruction* const operand);

// Stable Softmax
using HloStableSoftmaxInstruction = HloNonLinearity<PoplarOp::StableSoftmax>;
std::unique_ptr<HloInstruction> CreateStableSoftmax(
    HloInstruction* const operand);

template <PoplarOp Op>
class HloNonLinearityGrad : public HloPoplarInstruction {
 public:
  HloNonLinearityGrad(HloInstruction* const out, HloInstruction* const grad)
      : HloPoplarInstruction(out->shape(), {out, grad}, Op) {}

  absl::flat_hash_set<int64_t> AllocatingIndices() const override { return {}; }
  bool AllocatingOutput() const override { return false; }

  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override {
    return {};
  }
  HloPoplarUseDescriptions GetUseDescriptions() const {
    return UseDescriptionsNoInputOutputAlias();
  }
  HloPoplarBufferDescriptions GetBufferDescriptions() const {
    return BufferDescriptionsAllocatesAllOutputs(this);
  }

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const {
    return FindConsumersExtensionResults();
  }

  bool AllowNonInplaceLowering() const override { return false; }
  bool IsPopOpsElementwise() const override { return true; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const {
    return absl::make_unique<HloNonLinearityGrad<Op>>(new_operands[0],
                                                      new_operands[1]);
  }
};

// ReluGrad
using HloReluGradInstruction = HloNonLinearityGrad<PoplarOp::ReluGrad>;
std::unique_ptr<HloInstruction> CreateReluGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

// GeluGrad
using HloGeluGradInstruction = HloNonLinearityGrad<PoplarOp::GeluGrad>;
std::unique_ptr<HloInstruction> CreateGeluGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

// SigmoidGrad
using HloSigmoidGradInstruction = HloNonLinearityGrad<PoplarOp::SigmoidGrad>;
std::unique_ptr<HloInstruction> CreateSigmoidGrad(HloInstruction* const out,
                                                  HloInstruction* const grad);

// HardSigmoidGrad
using HloHardSigmoidGradInstruction =
    HloNonLinearityGrad<PoplarOp::HardSigmoidGrad>;
std::unique_ptr<HloInstruction> CreateHardSigmoidGrad(
    HloInstruction* const out, HloInstruction* const grad);

// TanhGrad
using HloTanhGradInstruction = HloNonLinearityGrad<PoplarOp::TanhGrad>;
std::unique_ptr<HloInstruction> CreateTanhGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

// SwishGrad
using HloSwishGradInstruction = HloNonLinearityGrad<PoplarOp::SwishGrad>;
std::unique_ptr<HloInstruction> CreateSwishGrad(HloInstruction* const out,
                                                HloInstruction* const grad);
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NON_LINEARITY_H_
