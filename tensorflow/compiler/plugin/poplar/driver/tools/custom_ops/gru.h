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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_GRU_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_GRU_H_

#include <memory>
#include <string>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"

namespace xla {
namespace poplarplugin {
namespace rnn_helper {
// Helper for parsing the attribute map when converting a custom call
// instruction for an GRU.
struct GRUAttributes : public RNNAttributes {
  static StatusOr<GRUAttributes> Parse(const HloCustomCallInstruction* call);
  GRUAttributes() = delete;

  bool reset_after;

 protected:
  GRUAttributes(int32 num_channels, bool is_training,
                xla::PrimitiveType partials_xla_type,
                rnn_helper::ActivationType activation,
                rnn_helper::ActivationType recurrent_activation,
                bool reset_after);
};
}  // namespace rnn_helper

class HloGRUInstructionCommon {
 protected:
  explicit HloGRUInstructionCommon(bool reset_after);

 public:
  bool reset_after() const;

 private:
  bool reset_after_;
};

class HloGRUFwdInstruction : public HloRNNFwdInstruction,
                             public HloGRUInstructionCommon {
 public:
  explicit HloGRUFwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloGRUBwdInstruction : public HloRNNBwdInstruction,
                             public HloGRUInstructionCommon {
 public:
  explicit HloGRUBwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

std::unique_ptr<HloInstruction> CreateGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

class HloDynamicGRUFwdInstruction : public HloRNNFwdInstruction,
                                    public HloGRUInstructionCommon {
 public:
  explicit HloDynamicGRUFwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloDynamicGRUBwdInstruction : public HloRNNBwdInstruction,
                                    public HloGRUInstructionCommon {
 public:
  explicit HloDynamicGRUBwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateDynamicGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

std::unique_ptr<HloInstruction> CreateDynamicGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

class HloAUGRUFwdInstruction : public HloRNNFwdInstruction,
                               public HloGRUInstructionCommon {
 public:
  explicit HloAUGRUFwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  bool AllocatingOutput() const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

class HloAUGRUBwdInstruction : public HloRNNBwdInstruction,
                               public HloGRUInstructionCommon {
 public:
  explicit HloAUGRUBwdInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      bool is_training, rnn_helper::ActivationType activation,
      rnn_helper::ActivationType recurrent_activation, int32 num_channels,
      xla::PrimitiveType partials_type, bool reset_after);

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloCloneContext* ctx) const override;
};

std::unique_ptr<HloInstruction> CreateAUGRUFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

std::unique_ptr<HloInstruction> CreateAUGRUBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, rnn_helper::ActivationType activation,
    rnn_helper::ActivationType recurrent_activation, int32 num_channels,
    xla::PrimitiveType partials_type, bool reset_after);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_GRU_H_
