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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloStatefulGradientAccumulate : public HloPoplarInstruction {
 public:
  explicit HloStatefulGradientAccumulate(
      absl::Span<HloInstruction* const> operands, int32 num_mini_batches,
      PoplarOp op = PoplarOp::StatefulGradientAccumulate);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

  // The number of mini batches which will be accumulated.
  int32 MiniBatchesToAccumulate() const { return num_mini_batches_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

 protected:
  int32 num_mini_batches_;
};

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulation(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

class HloStatefulGradientAccumulateAndAllReduce
    : public HloStatefulGradientAccumulate {
 public:
  explicit HloStatefulGradientAccumulateAndAllReduce(
      absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulateAndAllReduce(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

class HloStatefulGradientAccumulateWithMomentum
    : public HloStatefulGradientAccumulate {
 public:
  explicit HloStatefulGradientAccumulateWithMomentum(
      absl::Span<HloInstruction* const> operands, int32 num_mini_batches);
  uint64 NumberOfInplaceOperands() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulateWithMomentum(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

class HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm
    : public HloStatefulGradientAccumulate {
 public:
  explicit HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm(
      absl::Span<HloInstruction* const> operands, int32 num_mini_batches);
  uint64 NumberOfInplaceOperands() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction>
CreateStatefulGradientAccumulationWithMomentumAndAllReduceWithNorm(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

// Gradient accumulation is split into the following ops:
// * HloGradientAccumulatorCreate - this op creates the gradient accumulation
//   buffer and zeros it at the begining/after the gradients have been applied.
// * HloGradientAccumulatorAdd - this op takes the accumulator on the LHS and
//   the gradient on the RHS. This op is converted into an elementwise add
//   before lowering.
// * HloGradientAccumulatorSink - this op combines accumulators when they are
//   used in multiple pipeline stages and unifies them into a single buffer.
class HloGradientAccumulatorCreate : public HloPoplarInstruction {
 public:
  explicit HloGradientAccumulatorCreate(const Shape& shape);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGradientAccumulatorCreate(
    const Shape& shape);

class HloGradientAccumulatorAdd : public HloPoplarInstruction {
 public:
  explicit HloGradientAccumulatorAdd(HloInstruction* const accumulator,
                                     HloInstruction* const gradient);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGradientAccumulatorAdd(
    HloInstruction* const accumulator, HloInstruction* const gradient);

class HloGradientAccumulatorSink : public HloPoplarInstruction {
 public:
  explicit HloGradientAccumulatorSink(
      absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

  // The number of mini batches which will be accumulated.
  int32 MiniBatchesToAccumulate() const { return num_mini_batches_; }

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

 protected:
  int32 num_mini_batches_;
};

std::unique_ptr<HloInstruction> CreateGradientAccumulatorSink(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H
