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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_

#include <memory>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloArgMinMaxBase : public HloPoplarInstruction {
 protected:
  HloArgMinMaxBase(HloInstruction* input, const Shape& output_shape, int64 axis,
                   const PoplarOp& opcode);

 public:
  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;

  int64 Axis() const { return axis; }

  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const int64 axis;
};

class HloArgMax : public HloArgMinMaxBase {
 public:
  HloArgMax(HloInstruction* input, const Shape& output_shape, int64 axis)
      : HloArgMinMaxBase(input, output_shape, axis, PoplarOp::ArgMax) {}

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloArgMax(HloInstruction* input,
                                                const Shape& shape, int64 axis);

class HloArgMin : public HloArgMinMaxBase {
 public:
  HloArgMin(HloInstruction* input, const Shape& output_shape, int64 axis)
      : HloArgMinMaxBase(input, output_shape, axis, PoplarOp::ArgMin) {}

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloArgMin(HloInstruction* input,
                                                const Shape& shape, int64 axis);

class HloMaxAndArgMax : public HloArgMinMaxBase {
 public:
  HloMaxAndArgMax(HloInstruction* input, const Shape& output_shape, int64 axis)
      : HloArgMinMaxBase(input, output_shape, axis, PoplarOp::MaxAndArgMax) {}

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloMaxAndArgMax(HloInstruction* input,
                                                      const Shape& shape,
                                                      int64 axis);

class HloMinAndArgMin : public HloArgMinMaxBase {
 public:
  HloMinAndArgMin(HloInstruction* input, const Shape& output_shape, int64 axis)
      : HloArgMinMaxBase(input, output_shape, axis, PoplarOp::MinAndArgMin) {}

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateHloMinAndArgMin(HloInstruction* input,
                                                      const Shape& shape,
                                                      int64 axis);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_ONE_HOT_H_
