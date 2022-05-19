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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ELEMENTWISE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ELEMENTWISE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

// Base class for unary elementwise operations which are not represented in XLA
// natively.
template <PoplarOp Op>
class HloElementwiseUnaryBase : public HloPoplarInstruction {
 public:
  explicit HloElementwiseUnaryBase(HloInstruction* const operand)
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
      FindConsumersExtensionParams params) const override {
    return FindConsumersExtensionResults::DoNotFindConsumers();
  }

  bool AllowNonInplaceLowering() const override { return true; }

  bool IsPopOpsElementwise() const override { return true; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override {
    return {};
  }
};

// Square
class HloSquareInstruction : public HloElementwiseUnaryBase<PoplarOp::Square> {
 public:
  using HloElementwiseUnaryBase::HloElementwiseUnaryBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateSquare(HloInstruction* const operand);

// Inverse
class HloInverseInstruction
    : public HloElementwiseUnaryBase<PoplarOp::Inverse> {
 public:
  using HloElementwiseUnaryBase::HloElementwiseUnaryBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateInverse(HloInstruction* const operand);

// Erf
class HloErfInstruction : public HloElementwiseUnaryBase<PoplarOp::Erf> {
 public:
  using HloElementwiseUnaryBase::HloElementwiseUnaryBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateErf(HloInstruction* const operand);

// GeluErf
class HloGeluErfInstruction
    : public HloElementwiseUnaryBase<PoplarOp::GeluErf> {
 public:
  using HloElementwiseUnaryBase::HloElementwiseUnaryBase;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};
std::unique_ptr<HloInstruction> CreateGeluErf(HloInstruction* const operand);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ELEMENTWISE_H_
