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
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

inline const HloInstruction* LookThroughBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast ? inst->operand(0) : inst;
}

static StatusOr<poplar::program::Sequence> RandomNormal(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    double mean_val, double sd_val, const xla::Shape& output_shape,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                      AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                res, tensor_map, {debug_name_and_id, "ref"}));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq({}, debug_name_and_id);
  auto out = poprand::normal(graph, nullptr, 0, ref, dtype, mean_val, sd_val,
                             seq, {debug_name_and_id});

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
  return seq;
}

class RandomNormalScaleOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RandomNormalScaleOp");
    const HloInstruction* root = inst->fused_expression_root();
    const HloInstruction* mean1 = LookThroughBroadcast(root->operand(1));
    CHECK_EQ(mean1->opcode(), HloOpcode::kConstant);
    const HloInstruction* sd1 =
        LookThroughBroadcast(root->operand(0)->operand(1));
    CHECK_EQ(sd1->opcode(), HloOpcode::kConstant);
    const HloInstruction* mean2 = root->operand(0)->operand(0)->operand(0);
    CHECK_EQ(mean2->opcode(), HloOpcode::kConstant);
    const HloInstruction* sd2 = root->operand(0)->operand(0)->operand(1);
    CHECK_EQ(sd2->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(double mean1_val,
                        LiteralScalarToNativeType<double>(mean1->literal()));
    TF_ASSIGN_OR_RETURN(double mean2_val,
                        LiteralScalarToNativeType<double>(mean2->literal()));
    TF_ASSIGN_OR_RETURN(double sd1_val,
                        LiteralScalarToNativeType<double>(sd1->literal()));
    TF_ASSIGN_OR_RETURN(double sd2_val,
                        LiteralScalarToNativeType<double>(sd2->literal()));

    double mean_val = mean1_val + mean2_val;
    double sd_val = sd1_val * sd2_val;
    return RandomNormal(graph, res, inst, mean_val, sd_val, output_shape,
                        tensor_map, {debug_info});
  }
};

REGISTER_POPLAR_OP(Norm_scale_add, RandomNormalScaleOp);

static StatusOr<poplar::program::Sequence> RandomUniform(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    double lower_val, double upper_val, const xla::Shape& output_shape,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                      AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                res, tensor_map, {debug_name_and_id, "ref"}));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq({}, debug_name_and_id);
  auto out = poprand::uniform(graph, nullptr, 0, ref, dtype, lower_val,
                              upper_val, seq, {debug_name_and_id});

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
  return seq;
}

class RandomUniformScaleOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RandomUniformScaleOp");
    const HloInstruction* root = inst->fused_expression_root();
    const HloInstruction* shift = LookThroughBroadcast(root->operand(1));
    CHECK_EQ(shift->opcode(), HloOpcode::kConstant);
    const HloInstruction* scale =
        LookThroughBroadcast(root->operand(0)->operand(1));
    CHECK_EQ(scale->opcode(), HloOpcode::kConstant);
    const HloInstruction* lower = root->operand(0)->operand(0)->operand(0);
    CHECK_EQ(lower->opcode(), HloOpcode::kConstant);
    const HloInstruction* upper = root->operand(0)->operand(0)->operand(1);
    CHECK_EQ(upper->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(double shift_val,
                        LiteralScalarToNativeType<double>(shift->literal()));
    TF_ASSIGN_OR_RETURN(double scale_val,
                        LiteralScalarToNativeType<double>(scale->literal()));
    TF_ASSIGN_OR_RETURN(double lower_val,
                        LiteralScalarToNativeType<double>(lower->literal()));
    TF_ASSIGN_OR_RETURN(double upper_val,
                        LiteralScalarToNativeType<double>(upper->literal()));

    lower_val = lower_val * scale_val + shift_val;
    upper_val = upper_val * scale_val + shift_val;

    return RandomUniform(graph, res, inst, lower_val, upper_val, output_shape,
                         tensor_map, {debug_info});
  }
};

REGISTER_POPLAR_OP(Uniform_scale_add, RandomUniformScaleOp);

class RngOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RngOp");
    if (inst->operand_count() != 2) {
      return FailedPrecondition("RNG must have two operands.");
    }
    for (auto* op : inst->operands()) {
      if (op->opcode() != HloOpcode::kConstant) {
        return UnimplementedStrCat("RNG operation ", inst->ToString(),
                                   " has non-constant input ", op->ToString(),
                                   " which is not currently supported.");
      }
    }

    const HloInstruction* op0 = inst->operand(0);
    const HloInstruction* op1 = inst->operand(1);

    TF_ASSIGN_OR_RETURN(double op0_val,
                        LiteralScalarToNativeType<double>(op0->literal()));
    TF_ASSIGN_OR_RETURN(double op1_val,
                        LiteralScalarToNativeType<double>(op1->literal()));

    switch (inst->random_distribution()) {
      case RandomDistribution::RNG_NORMAL: {
        return RandomNormal(graph, res, inst, op0_val, op1_val, output_shape,
                            tensor_map, {debug_info});
      }
      case RandomDistribution::RNG_UNIFORM: {
        if (ShapeUtil::ElementIsIntegral(output_shape)) {
          op1_val -= 1.0;
        }
        return RandomUniform(graph, res, inst, op0_val, op1_val, output_shape,
                             tensor_map, {debug_info});
      }
      default: {
        return FailedPrecondition(
            "Unsupported random distribution type: %s.",
            RandomDistributionToString(inst->random_distribution()).c_str());
      }
    }
  }
};

REGISTER_HLO_OP(kRng, RngOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
