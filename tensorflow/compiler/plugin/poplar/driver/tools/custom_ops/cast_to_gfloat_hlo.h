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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CAST_TO_GFLOAT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CAST_TO_GFLOAT_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

#include "tensorflow/compiler/plugin/poplar/kernels/popfloat/gfloat_config_utils.pb.h"

namespace xla {
namespace poplarplugin {

class HloGfloatParamsInstruction : public HloPoplarInstruction {
 public:
  explicit HloGfloatParamsInstruction(const Shape& shape, int32 mantissa_,
                                      int32 exponent_, int32 bias_,
                                      bool en_denorm_, bool en_inf_,
                                      tensorflow::DataType calc_type_);

  int32 NumberMantissaBits() const;
  int32 NumberExponentBits() const;
  int32 ExponentBias() const;
  bool IsDenormEnabled() const;
  bool InfAndNansEnabled() const;
  tensorflow::DataType CalculationType() const;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  int32 mantissa_;
  int32 exponent_;
  int32 bias_;
  bool en_denorm_;
  bool en_inf_;
  tensorflow::DataType calc_type_;

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateGfloatParams(
    const Shape& shape, int32 mantissa, int32 exponent, int32 bias,
    bool en_denorm, bool en_inf, tensorflow::DataType calc_type);

class HloCastNativeToGfloatInstruction : public HloPoplarInstruction {
 public:
  explicit HloCastNativeToGfloatInstruction(const Shape& shape,
                                            HloInstruction* const operand,
                                            HloInstruction* const params,
                                            tensorflow::DataType in_type_,
                                            tensorflow::DataType out_type_,
                                            tensorflow::DataType calc_type_,
                                            std::string cast_op_config_);

  std::string CastOpConfig() const;
  tensorflow::DataType InputType() const;
  tensorflow::DataType OutputType() const;
  tensorflow::DataType CalculationType() const;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::string cast_op_config_;
  tensorflow::DataType in_type_;
  tensorflow::DataType out_type_;
  tensorflow::DataType calc_type_;

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateCastNativeToGfloat(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, tensorflow::DataType in_type_,
    tensorflow::DataType out_type_, tensorflow::DataType calc_type_,
    std::string popfloat_cast_cfg);

class HloCastGfloatToNativeInstruction : public HloPoplarInstruction {
 public:
  explicit HloCastGfloatToNativeInstruction(
      const Shape& shape, HloInstruction* const operand,
      HloInstruction* const params, const tensorflow::DataType in_type_,
      const tensorflow::DataType out_type_,
      const tensorflow::DataType calc_type_,
      GFConfig::GfloatFormat gfloat_format_);

  const tensorflow::DataType InputType() const;
  const tensorflow::DataType OutputType() const;
  const tensorflow::DataType CalculationType() const;
  GFConfig::GfloatFormat GfloatFormat() const;

  absl::flat_hash_set<int64_t> AllocatingIndices() const override;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64_t, int64_t> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool AllowNonInplaceLowering() const override;
  bool IsPopOpsElementwise() const override;

 protected:
  std::vector<std::string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  const tensorflow::DataType in_type_;
  const tensorflow::DataType out_type_;
  const tensorflow::DataType calc_type_;
  GFConfig::GfloatFormat gfloat_format_;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;
};

std::unique_ptr<HloInstruction> CreateCastGfloatToNative(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, const tensorflow::DataType in_type_,
    const tensorflow::DataType out_type_, const tensorflow::DataType calc_type_,
    GFConfig::GfloatFormat gfloat_format_);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_CAST_TO_GFLOAT_H_
