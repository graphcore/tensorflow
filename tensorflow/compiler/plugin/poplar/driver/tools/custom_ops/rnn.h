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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RNN_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RNN_H_

#include <string>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
class HloCustomCallInstruction;
namespace poplarplugin {
namespace rnn_helper {

enum ActivationType { SOFTMAX, RELU, TANH, SIGMOID, HARD_SIGMOID };

// Helper for parsing the attribute map when converting a custom call
// instruction for an RNN.
struct RNNAttributes {
  static StatusOr<RNNAttributes> Parse(const HloCustomCallInstruction* call);
  RNNAttributes() = delete;

  int32 num_channels;
  bool is_training;
  xla::PrimitiveType partials_xla_type;
  ActivationType activation;
  ActivationType recurrent_activation;
  float available_memory_proportion;

 protected:
  RNNAttributes(int32 num_channels, bool is_training,
                xla::PrimitiveType partials_xla_type, ActivationType activation,
                ActivationType recurrent_activation,
                float available_memory_proportion);
};
}  // namespace rnn_helper

class HloRNNInstruction : public HloPoplarInstruction {
 public:
  template <typename... Args>
  explicit HloRNNInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      PoplarOp op, bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, float available_memory_proportion,
      Args&&... attributes)
      : HloPoplarInstruction(shape, operands, op, is_training, activation,
                             recurrent_activation, num_channels, partials_type,
                             attributes...),
        is_training_(is_training),
        activation_(activation),
        recurrent_activation_(recurrent_activation),
        num_channels_(num_channels),
        partials_type_(partials_type),
        available_memory_proportion_(available_memory_proportion) {}

  bool is_training() const;
  rnn_helper::ActivationType activation() const;
  rnn_helper::ActivationType recurrent_activation() const;
  int32 num_channels() const;
  xla::PrimitiveType partials_type() const;
  float available_memory_proportion() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  bool is_training_;
  rnn_helper::ActivationType activation_;
  rnn_helper::ActivationType recurrent_activation_;
  int32 num_channels_;
  xla::PrimitiveType partials_type_;
  float available_memory_proportion_;
};

class HloRNNFwdInstruction : public HloRNNInstruction {
 public:
  template <typename... Args>
  explicit HloRNNFwdInstruction(PoplarOp op, const Shape& shape,
                                absl::Span<HloInstruction* const> operands,
                                bool is_training,
                                rnn_helper::ActivationType activation,
                                rnn_helper::ActivationType recurrent_activation,
                                int32 num_channels,
                                xla::PrimitiveType partials_type,
                                float available_memory_proportion,
                                Args&&... attributes)
      : HloRNNInstruction(shape, operands, op, is_training, activation,
                          recurrent_activation, num_channels, partials_type,
                          available_memory_proportion, attributes...),
        op_(op) {}

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 private:
  const PoplarOp op_;
};

class HloRNNBwdInstruction : public HloRNNInstruction {
 public:
  template <typename... Args>
  explicit HloRNNBwdInstruction(PoplarOp op, const Shape& shape,
                                absl::Span<HloInstruction* const> operands,
                                bool is_training,
                                rnn_helper::ActivationType activation,
                                rnn_helper::ActivationType recurrent_activation,
                                int32 num_channels,
                                xla::PrimitiveType partials_type,
                                float available_memory_proportion,
                                Args&&... attributes)
      : HloRNNInstruction(shape, operands, op, is_training, activation,
                          recurrent_activation, num_channels, partials_type,
                          available_memory_proportion, attributes...),
        op_(op) {}

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;
  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;
  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 private:
  const PoplarOp op_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RNN_H_
