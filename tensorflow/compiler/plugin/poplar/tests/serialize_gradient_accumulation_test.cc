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
#include "tensorflow/compiler/plugin/poplar/driver/passes/serialize_gradient_accumulation.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

struct SerializeGradientAccumulationTestSpec {
  PrimitiveType gradient_type;
  PrimitiveType accumulator_type;
};

void PrintTo(const SerializeGradientAccumulationTestSpec& spec,
             std::ostream* os) {
  *os << PrimitiveType_Name(spec.gradient_type) << ", "
      << PrimitiveType_Name(spec.accumulator_type);
}

struct SerializeGradientAccumulationTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          SerializeGradientAccumulationTestSpec> {
  const HloInstruction* UnwrapConvert(const HloInstruction* inst) const {
    if (GetParam().gradient_type == GetParam().accumulator_type) {
      return inst;
    }
    EXPECT_EQ(inst->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(inst->shape().element_type(), GetParam().accumulator_type);
    return inst->operand(0);
  }
};

INSTANTIATE_TEST_SUITE_P(
    SerializeGradientAccumulationTestCases, SerializeGradientAccumulationTest,
    ::testing::Values(SerializeGradientAccumulationTestSpec{F32, F32},
                      SerializeGradientAccumulationTestSpec{F16, F32}));

string ReplaceParams(absl::string_view s,
                     const SerializeGradientAccumulationTestSpec& spec) {
  return absl::StrReplaceAll(
      s, {
             {"$GT",
              primitive_util::LowercasePrimitiveTypeName(spec.gradient_type)},
             {"$AT", primitive_util::LowercasePrimitiveTypeName(
                         spec.accumulator_type)},
         });
}

TEST_P(SerializeGradientAccumulationTest, MultiUpdateAdd) {
  const auto hlo_template = R"(
HloModule main

_pop_op_wide_const {
  c = $GT[] constant(0)
  ROOT b = $GT[100,16] broadcast(c), dimensions={}
}

ENTRY main {
  accumulator = $AT[100, 16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = $GT[24,16] parameter(2)
  scale = $GT[] parameter(3)
  big_zero = $GT[100,16] fusion(), kind=kCustom, calls=_pop_op_wide_const
  mua = $GT[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  add = $AT[100,16] custom-call(accumulator, mua), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(
      accumulator_add->fused_expression_root()));

  const auto* entry = module->entry_computation();
  const auto* offsets = entry->parameter_instruction(1);
  const auto* updates = entry->parameter_instruction(2);
  const auto* scale = entry->parameter_instruction(3);

  const auto offsets_index = accumulator_add->operand_index(offsets);
  const auto updates_index = accumulator_add->operand_index(updates);
  const auto scale_index = accumulator_add->operand_index(scale);

  const auto accumulator_index = accumulator_add->operand_index(accumulator);
  EXPECT_EQ(accumulator_index, 0);

  if (param.gradient_type == param.accumulator_type) {
    EXPECT_TRUE(Match(
        accumulator_add->fused_expression_root(),
        m::CustomCall(m::Parameter(accumulator_index),
                      m::Parameter(offsets_index), m::Parameter(updates_index),
                      m::Parameter(scale_index))));
  } else {
    EXPECT_TRUE(Match(accumulator_add->fused_expression_root(),
                      m::CustomCall(m::Parameter(accumulator_index),
                                    m::Parameter(offsets_index),
                                    m::Convert(m::Parameter(updates_index)),
                                    m::Convert(m::Parameter(scale_index)))));
  }
}

TEST_P(SerializeGradientAccumulationTest, MultiUpdateAddWithAdds) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  accumulator = $AT[100, 16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = $GT[24,16] parameter(2)
  scale = $GT[] parameter(3)
  grads1 = $GT[100, 16] parameter(4)
  grads2 = $GT[100, 16] parameter(5)
  add_grads = $GT[100, 16] add(grads1, grads2)
  mua = $GT[100,16] custom-call(add_grads, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  add = $AT[100,16] custom-call(accumulator, mua), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(
      accumulator_add->fused_expression_root()));

  const auto* entry = module->entry_computation();
  const auto* offsets = entry->parameter_instruction(1);
  const auto* updates = entry->parameter_instruction(2);
  const auto* scale = entry->parameter_instruction(3);
  const auto* grads1 = entry->parameter_instruction(4);
  const auto* grads2 = entry->parameter_instruction(5);

  const auto offsets_index = accumulator_add->operand_index(offsets);
  const auto updates_index = accumulator_add->operand_index(updates);
  const auto scale_index = accumulator_add->operand_index(scale);
  const auto grads1_index = accumulator_add->operand_index(grads1);
  const auto grads2_index = accumulator_add->operand_index(grads2);

  const auto accumulator_index = accumulator_add->operand_index(accumulator);
  EXPECT_EQ(accumulator_index, 0);

  if (param.gradient_type == param.accumulator_type) {
    EXPECT_TRUE(Match(
        accumulator_add->fused_expression_root(),
        m::CustomCall(m::Add(m::Add(m::Parameter(accumulator_index),
                                    m::Parameter(grads1_index)),
                             m::Parameter(grads2_index)),
                      m::Parameter(offsets_index), m::Parameter(updates_index),
                      m::Parameter(scale_index))));
  } else {
    EXPECT_TRUE(Match(
        accumulator_add->fused_expression_root(),
        m::CustomCall(m::Add(m::Add(m::Parameter(accumulator_index),
                                    m::Convert(m::Parameter(grads1_index))),
                             m::Convert(m::Parameter(grads2_index))),
                      m::Parameter(offsets_index),
                      m::Convert(m::Parameter(updates_index)),
                      m::Convert(m::Parameter(scale_index)))));
  }
}

TEST_P(SerializeGradientAccumulationTest, ConcatenateWithAdd) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  p0 = $GT[10, 5] parameter(0)
  p1 = $GT[10, 2] parameter(1)
  p2 = $GT[10, 2] parameter(2)
  p3 = $GT[10, 7] parameter(3)
  p4 = $GT[10, 16] parameter(4)
  p5 = $GT[10, 16] parameter(5)
  p6 = $GT[10, 16] parameter(6)
  accumulator = $AT[10, 16] parameter(7)
  a = $GT[10, 16] concatenate(p0, p1, p2, p3), dimensions={1}
  add1 = $GT[10, 16] add(p5, p4)
  add2 = $GT[10, 16] add(add1, a)
  add3 = $GT[10, 16] add(add2, p6)
  add = $AT[10, 16] custom-call(accumulator, add3), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* p0 = entry->parameter_instruction(0);
  const auto* p1 = entry->parameter_instruction(1);
  const auto* p2 = entry->parameter_instruction(2);
  const auto* p3 = entry->parameter_instruction(3);
  const auto* p4 = entry->parameter_instruction(4);
  const auto* p5 = entry->parameter_instruction(5);
  const auto* p6 = entry->parameter_instruction(6);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p6));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p3));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p2));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p1));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p0));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p4));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p5));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_P(SerializeGradientAccumulationTest, ScaleMultiplyConcatenate) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  p0 = $GT[10, 5] parameter(0)
  p1 = $GT[10, 2] parameter(1)
  p2 = $GT[10, 2] parameter(2)
  p3 = $GT[10, 7] parameter(3)
  accumulator = $AT[10, 16] parameter(4)
  a = $GT[10, 16] concatenate(p0, p1, p2, p3), dimensions={1}
  scale = $GT[] constant(0.1)
  bscale = $GT[10, 16] broadcast(scale), dimensions={}
  m = $GT[10, 16] multiply(a, bscale)
  add = $AT[10, 16] custom-call(accumulator, m), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* p0 = entry->parameter_instruction(0);
  const auto* p1 = entry->parameter_instruction(1);
  const auto* p2 = entry->parameter_instruction(2);
  const auto* p3 = entry->parameter_instruction(3);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p3));
    const auto* scale = UnwrapConvert(next->operand(2));
    EXPECT_EQ(scale->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(accumulator_add->operand(scale->parameter_number())->opcode(),
              HloOpcode::kConstant);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p2));
    const auto* scale = UnwrapConvert(next->operand(2));
    EXPECT_EQ(scale->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(accumulator_add->operand(scale->parameter_number())->opcode(),
              HloOpcode::kConstant);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p1));
    const auto* scale = UnwrapConvert(next->operand(2));
    EXPECT_EQ(scale->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(accumulator_add->operand(scale->parameter_number())->opcode(),
              HloOpcode::kConstant);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p0));
    const auto* scale = UnwrapConvert(next->operand(2));
    EXPECT_EQ(scale->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(accumulator_add->operand(scale->parameter_number())->opcode(),
              HloOpcode::kConstant);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_P(SerializeGradientAccumulationTest, AddsWithZero) {
  const auto hlo_template = R"(
HloModule main

_pop_op_wide_const {
  c = $GT[] constant(0)
  ROOT b = $GT[100,16] broadcast(c), dimensions={}
}

ENTRY main {
  accumulator = $AT[100, 16] parameter(0)
  big_zero = $GT[100,16] fusion(), kind=kCustom, calls=_pop_op_wide_const
  add = $AT[100,16] custom-call(accumulator, big_zero), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add, accumulator);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);
}

TEST_P(SerializeGradientAccumulationTest, TransposeConcatenateWithAdd) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  p0 = $GT[5, 10] parameter(0)
  p1 = $GT[2, 10] parameter(1)
  p2 = $GT[2, 10] parameter(2)
  p3 = $GT[7, 10] parameter(3)
  p4 = $GT[10, 16] parameter(4)
  p5 = $GT[10, 16] parameter(5)
  p6 = $GT[10, 16] parameter(6)
  accumulator = $AT[10, 16] parameter(7)
  a = $GT[16, 10] concatenate(p0, p1, p2, p3), dimensions={0}
  a_t = $GT[10, 16] transpose(a), dimensions={1,0}
  add1 = $GT[10, 16] add(p5, p4)
  add2 = $GT[10, 16] add(add1, a_t)
  add3 = $GT[10, 16] add(add2, p6)
  add = $AT[10, 16] custom-call(accumulator, add3), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* p0_t = entry->parameter_instruction(0)->users()[0];
  const auto* p1_t = entry->parameter_instruction(1)->users()[0];
  const auto* p2_t = entry->parameter_instruction(2)->users()[0];
  const auto* p3_t = entry->parameter_instruction(3)->users()[0];
  const auto* p4 = entry->parameter_instruction(4);
  const auto* p5 = entry->parameter_instruction(5);
  const auto* p6 = entry->parameter_instruction(6);

  EXPECT_EQ(p0_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p1_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p2_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p3_t->opcode(), HloOpcode::kTranspose);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p6));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p3_t));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p2_t));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p1_t));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p0_t));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p4));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p5));
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_P(SerializeGradientAccumulationTest, TransposeScaleMultiplyConcatenate) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  p0 = $GT[5, 10] parameter(0)
  p1 = $GT[2, 10] parameter(1)
  p2 = $GT[2, 10] parameter(2)
  p3 = $GT[7, 10] parameter(3)
  accumulator = $AT[10, 16] parameter(4)
  a = $GT[16, 10] concatenate(p0, p1, p2, p3), dimensions={0}
  scale = $GT[] constant(0.1)
  bscale = $GT[16, 10] broadcast(scale), dimensions={}
  m = $GT[16, 10] multiply(a, bscale)
  m_t = $GT[10, 16] transpose(m), dimensions={1,0}
  add = $AT[10, 16] custom-call(accumulator, m_t), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* p0_t = entry->parameter_instruction(0)->users()[0];
  const auto* p1_t = entry->parameter_instruction(1)->users()[0];
  const auto* p2_t = entry->parameter_instruction(2)->users()[0];
  const auto* p3_t = entry->parameter_instruction(3)->users()[0];

  EXPECT_EQ(p0_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p1_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p2_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(p3_t->opcode(), HloOpcode::kTranspose);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p3_t));
    EXPECT_EQ(UnwrapConvert(next->operand(2))->opcode(), HloOpcode::kParameter);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p2_t));
    EXPECT_EQ(UnwrapConvert(next->operand(2))->opcode(), HloOpcode::kParameter);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p1_t));
    EXPECT_EQ(UnwrapConvert(next->operand(2))->opcode(), HloOpcode::kParameter);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), accumulator_add->operand_index(p0_t));
    EXPECT_EQ(UnwrapConvert(next->operand(2))->opcode(), HloOpcode::kParameter);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_P(SerializeGradientAccumulationTest, TransposeAdds) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  accumulator = $AT[100, 16] parameter(0)
  grads1 = $GT[16, 100] parameter(1)
  grads2 = $GT[16, 100] parameter(2)
  add_grads = $GT[16, 100] add(grads1, grads2)
  add_grads_t = $GT[100, 16] transpose(add_grads), dimensions={1,0}
  add = $AT[100,16] custom-call(accumulator, add_grads_t), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[100, 16]) tuple(add)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* grads1_t = entry->parameter_instruction(1)->users()[0];
  const auto* grads2_t = entry->parameter_instruction(2)->users()[0];
  EXPECT_EQ(grads1_t->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(grads2_t->opcode(), HloOpcode::kTranspose);

  const auto grads1_t_index = accumulator_add->operand_index(grads1_t);
  const auto grads2_t_index = accumulator_add->operand_index(grads2_t);

  const auto accumulator_index = accumulator_add->operand_index(accumulator);
  EXPECT_EQ(accumulator_index, 0);

  if (param.gradient_type == param.accumulator_type) {
    EXPECT_TRUE(Match(accumulator_add->fused_expression_root(),
                      m::Add(m::Add(m::Parameter(accumulator_index),
                                    m::Parameter(grads1_t_index)),
                             m::Parameter(grads2_t_index))));
  } else {
    EXPECT_TRUE(Match(accumulator_add->fused_expression_root(),
                      m::Add(m::Add(m::Parameter(accumulator_index),
                                    m::Convert(m::Parameter(grads1_t_index))),
                             m::Convert(m::Parameter(grads2_t_index)))));
  }
}

TEST_P(SerializeGradientAccumulationTest, AddWithMultipleUsers) {
  const auto hlo_template = R"(
HloModule main

ENTRY main {
  accumulator = $AT[16, 100] parameter(0)
  grad = $GT[16, 100] parameter(1)
  add = $AT[16, 100] custom-call(accumulator, grad), custom_call_target="GradientAccumulatorAdd"
  ROOT t = ($AT[16, 100], $GT[16, 100]) tuple(add, grad)
}
)";
  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  auto accumulator_add = root->operand(0);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(accumulator_add));
  auto accumulator = accumulator_add->operand(0);

  SerializeGradientAccumulation sga;
  EXPECT_TRUE(sga.Run(module).ValueOrDie());
  accumulator_add = root->operand(0);
  EXPECT_EQ(accumulator_add->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(accumulator_add->shape().element_type(), param.accumulator_type);

  const auto* entry = module->entry_computation();
  const auto* grad = entry->parameter_instruction(1);
  const auto grad_index = accumulator_add->operand_index(grad);
  const auto accumulator_index = accumulator_add->operand_index(accumulator);
  EXPECT_EQ(accumulator_index, 0);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    const auto* added = UnwrapConvert(next->operand(1));
    EXPECT_EQ(added->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(added->parameter_number(), grad_index);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->shape().element_type(), param.accumulator_type);
    EXPECT_EQ(next->parameter_number(), accumulator_index);
  }

  EXPECT_EQ(entry->root_instruction()->operand(1), grad);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
