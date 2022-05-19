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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_LSTM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_LSTM_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"

namespace xla {
namespace poplarplugin {

class HloLSTMFwdInstruction : public HloRNNFwdInstruction {
 public:
  explicit HloLSTMFwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, const std::string& options);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloLSTMBwdInstruction : public HloRNNBwdInstruction {
 public:
  explicit HloLSTMBwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, const std::string& options);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateLSTMFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, const std::string& options);

std::unique_ptr<HloInstruction> CreateLSTMBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, const std::string& options);

class HloDynamicLSTMFwdInstruction : public HloRNNFwdInstruction {
 public:
  explicit HloDynamicLSTMFwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation,
      bool preserve_final_state, int32 num_channels,
      xla::PrimitiveType partials_type, const std::string& options);

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  bool preserve_final_state() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
  bool preserve_final_state_;
};

class HloDynamicLSTMBwdInstruction : public HloRNNBwdInstruction {
 public:
  explicit HloDynamicLSTMBwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, const std::string& options);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateDynamicLSTMFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, bool preserve_final_state,
    int32 num_channels, xla::PrimitiveType partials_type,
    const std::string& options);

std::unique_ptr<HloInstruction> CreateDynamicLSTMBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, const std::string& options);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_LSTM_H_
