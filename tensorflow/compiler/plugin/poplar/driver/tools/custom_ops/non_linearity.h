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
class HloNonLinearityBase : public HloPoplarInstruction {
 public:
  explicit HloNonLinearityBase(HloInstruction* const operand)
      : HloPoplarInstruction(operand->shape(), {operand}, Op) {}

  absl::flat_hash_set<int64> AllocatingIndices() const override { return {}; }
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
    return {};
  }

  HloPoplarUseDescriptions GetUseDescriptions() const override {
    return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
  }

  HloPoplarBufferDescriptions GetBufferDescriptions() const override {
    return BufferDescriptionsNoAllocations();
  }

  bool IsPopOpsElementwise() const override { return true; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }
};

// Relu
class HloReluInstruction : public HloNonLinearityBase<PoplarOp::Relu> {
 public:
  using HloNonLinearityBase::HloNonLinearityBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateRelu(HloInstruction* const operand);

// Gelu
class HloGeluInstruction : public HloNonLinearityBase<PoplarOp::Gelu> {
 public:
  using HloNonLinearityBase::HloNonLinearityBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateGelu(HloInstruction* const operand);

// Sigmoid
class HloSigmoidInstruction : public HloNonLinearityBase<PoplarOp::Sigmoid> {
 public:
  using HloNonLinearityBase::HloNonLinearityBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateSigmoid(HloInstruction* const operand);

template <PoplarOp Op>
class HloNonLinearityGradBase : public HloPoplarInstruction {
 public:
  HloNonLinearityGradBase(HloInstruction* const out, HloInstruction* const grad)
      : HloPoplarInstruction(out->shape(), {out, grad}, Op) {}

  absl::flat_hash_set<int64> AllocatingIndices() const override { return {}; }
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override {
    return {};
  }
  HloPoplarUseDescriptions GetUseDescriptions() const {
    return UseDescriptionsNoInputOutputAlias();
  }
  HloPoplarBufferDescriptions GetBufferDescriptions() const {
    return BufferDescriptionsAllocatesAllOutputs(this);
  }

  bool IsPopOpsElementwise() const override { return true; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }
};

// ReluGrad
class HloReluGradInstruction
    : public HloNonLinearityGradBase<PoplarOp::ReluGrad> {
 public:
  using HloNonLinearityGradBase::HloNonLinearityGradBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateReluGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

// GeluGrad
class HloGeluGradInstruction
    : public HloNonLinearityGradBase<PoplarOp::GeluGrad> {
 public:
  using HloNonLinearityGradBase::HloNonLinearityGradBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateGeluGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

// SigmoidGrad
class HloSigmoidGradInstruction
    : public HloNonLinearityGradBase<PoplarOp::SigmoidGrad> {
 public:
  using HloNonLinearityGradBase::HloNonLinearityGradBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateSigmoidGrad(HloInstruction* const out,
                                                  HloInstruction* const grad);

// TanhGrad
class HloTanhGradInstruction
    : public HloNonLinearityGradBase<PoplarOp::TanhGrad> {
 public:
  using HloNonLinearityGradBase::HloNonLinearityGradBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateTanhGrad(HloInstruction* const out,
                                               HloInstruction* const grad);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_NON_LINEARITY_H_
