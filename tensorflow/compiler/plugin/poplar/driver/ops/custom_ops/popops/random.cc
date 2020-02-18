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
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>

namespace xla {
namespace poplarplugin {
// Helper functions
namespace {

static StatusOr<poplar::program::Program> random_normal(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    double mean_val, double sd_val, const xla::Shape& output_shape,
    TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::normal(graph, nullptr, 0, ref, dtype, mean_val, sd_val,
                             seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

inline const HloInstruction* LookThroughBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast ? inst->operand(0) : inst;
}

}  // namespace

class RandomNormalOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloInstruction* mean = inst->operand(0);
    const HloInstruction* sd = inst->operand(1);

    TF_ASSIGN_OR_RETURN(double mean_val,
                        LiteralScalarToNativeType<double>(mean->literal()));
    TF_ASSIGN_OR_RETURN(double sd_val,
                        LiteralScalarToNativeType<double>(sd->literal()));

    return random_normal(graph, res, inst, mean_val, sd_val, output_shape,
                         tensor_map);
  }
};

REGISTER_POPLAR_OP(RandomNormal, RandomNormalOp);

class RandomNormalScaleOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloInstruction* root =
        inst->fused_instructions_computation()->root_instruction();
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
    return random_normal(graph, res, inst, mean_val, sd_val, output_shape,
                         tensor_map);
  }
};  // namespace poplarplugin

REGISTER_POPLAR_OP(Norm_scale_add, RandomNormalScaleOp);

static StatusOr<poplar::program::Program> random_uniform(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    double lower_val, double upper_val, const xla::Shape& output_shape,
    TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, TensorLocation{inst, 0}, output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::uniform(graph, nullptr, 0, ref, dtype, lower_val,
                              upper_val, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

class RandomUniformScaleOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloInstruction* root =
        inst->fused_instructions_computation()->root_instruction();
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

    return random_uniform(graph, res, inst, lower_val, upper_val, output_shape,
                          tensor_map);
  }
};

REGISTER_POPLAR_OP(Uniform_scale_add, RandomUniformScaleOp);

class RandomUniformOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloInstruction* lower = inst->operand(0);
    const HloInstruction* upper = inst->operand(1);

    TF_ASSIGN_OR_RETURN(double lower_val,
                        LiteralScalarToNativeType<double>(lower->literal()));
    TF_ASSIGN_OR_RETURN(double upper_val,
                        LiteralScalarToNativeType<double>(upper->literal()));

    if (ShapeUtil::ElementIsIntegral(output_shape)) {
      upper_val -= 1.0;
    }
    return random_uniform(graph, res, inst, lower_val, upper_val, output_shape,
                          tensor_map);
  }
};

REGISTER_POPLAR_OP(RandomUniform, RandomUniformOp);

}  // namespace poplarplugin

}  // namespace xla
