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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
class HloCustomCallInstruction;
namespace poplarplugin {
namespace rnn_helper {
// Helper for parsing the attribute map when converting a custom call
// instruction for an RNN.
struct RNNAttributes {
  static StatusOr<RNNAttributes> Parse(const HloCustomCallInstruction* call);
  RNNAttributes() = delete;

  int32 num_channels;
  bool is_training;
  xla::PrimitiveType partials_xla_type;

 private:
  RNNAttributes(int32 num_channels, bool is_training,
                xla::PrimitiveType partials_xla_type);
};
}  // namespace rnn_helper

class HloRNNInstruction : public HloPoplarInstruction {
 public:
  explicit HloRNNInstruction(const Shape& shape,
                             absl::Span<HloInstruction* const> operands,
                             const std::string& custom_call_target,
                             bool is_training, int32 num_channels,
                             xla::PrimitiveType partials_type);

  bool is_training() const;
  int32 num_channels() const;
  xla::PrimitiveType partials_type() const;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  bool is_training_;
  int32 num_channels_;
  xla::PrimitiveType partials_type_;
};

class HloRNNFwdInstruction : public HloRNNInstruction {
 public:
  explicit HloRNNFwdInstruction(const PoplibsOp::Op& op, const Shape& shape,
                                absl::Span<HloInstruction* const> operands,
                                bool is_training, int32 num_channels,
                                xla::PrimitiveType partials_type);

  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  const PoplibsOp::Op op_;
};

class HloRNNBwdInstruction : public HloRNNInstruction {
 public:
  explicit HloRNNBwdInstruction(const PoplibsOp::Op& op, const Shape& shape,
                                absl::Span<HloInstruction* const> operands,
                                bool is_training, int32 num_channels,
                                xla::PrimitiveType partials_type);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;
  bool IsPopOpsElementwise() const override;

 private:
  const PoplibsOp::Op op_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_RNN_H_
