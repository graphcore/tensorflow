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

using SerializeGradientAccumulationTest = HloTestBase;

TEST_F(SerializeGradientAccumulationTest, MultiUpdateAdd) {
  const string& hlo_string = R"(
HloModule main

_pop_op_wide_const {
  c = f32[] constant(0)
  ROOT b = f32[100,16] broadcast(c), dimensions={}
}

ENTRY main {
  accumulator = f32[100, 16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  big_zero = f32[100,16] fusion(), kind=kCustom, calls=_pop_op_wide_const
  mua = f32[100,16] custom-call(big_zero, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  add = f32[100,16] custom-call(accumulator, mua), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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

  EXPECT_TRUE(Match(accumulator_add->fused_expression_root(),
                    m::CustomCall(m::Parameter(0), m::Parameter(1),
                                  m::Parameter(2), m::Parameter(3))));

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(
      accumulator_add->fused_expression_root()));
  EXPECT_EQ(accumulator_add->operand(0), accumulator);
}

TEST_F(SerializeGradientAccumulationTest, MultiUpdateAddWithAdds) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  accumulator = f32[100, 16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  scale = f32[] parameter(3)
  grads1 = f32[100, 16] parameter(4)
  grads2 = f32[100, 16] parameter(5)
  add_grads = f32[100, 16] add(grads1, grads2)
  mua = f32[100,16] custom-call(add_grads, offsets, updates, scale), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  add = f32[100,16] custom-call(accumulator, mua), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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

  EXPECT_TRUE(Match(
      accumulator_add->fused_expression_root(),
      m::CustomCall(
          m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
          m::Parameter(3), m::Parameter(4), m::Parameter(5))));

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(
      accumulator_add->fused_expression_root()));
  EXPECT_EQ(accumulator_add->operand(0), accumulator);
}

TEST_F(SerializeGradientAccumulationTest, ConcatenateWithAdd) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[10, 5] parameter(0)
  p1 = f32[10, 2] parameter(1)
  p2 = f32[10, 2] parameter(2)
  p3 = f32[10, 7] parameter(3)
  p4 = f32[10, 16] parameter(4)
  p5 = f32[10, 16] parameter(5)
  p6 = f32[10, 16] parameter(6)
  accumulator = f32[10, 16] parameter(7)
  a = f32[10, 16] concatenate(p0, p1, p2, p3), dimensions={1}
  add1 = f32[10, 16] add(p5, p4)
  add2 = f32[10, 16] add(add1, a)
  add3 = f32[10, 16] add(add2, p6)
  add = f32[10, 16] custom-call(accumulator, add3), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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
  EXPECT_EQ(accumulator_add->operand(0), accumulator);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 7);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 6);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 5);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 4);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 3);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 1);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_F(SerializeGradientAccumulationTest, ScaleMultiplyConcatenate) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[10, 5] parameter(0)
  p1 = f32[10, 2] parameter(1)
  p2 = f32[10, 2] parameter(2)
  p3 = f32[10, 7] parameter(3)
  accumulator = f32[10, 16] parameter(4)
  a = f32[10, 16] concatenate(p0, p1, p2, p3), dimensions={1}
  scale = f32[] constant(0.1)
  bscale = f32[10, 16] broadcast(scale), dimensions={}
  m = f32[10, 16] multiply(a, bscale)
  add = f32[10, 16] custom-call(accumulator, m), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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
  EXPECT_EQ(accumulator_add->operand(0), accumulator);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 5);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 4);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 3);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 1);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_F(SerializeGradientAccumulationTest, AddsWithZero) {
  const string& hlo_string = R"(
HloModule main

_pop_op_wide_const {
  c = f32[] constant(0)
  ROOT b = f32[100,16] broadcast(c), dimensions={}
}

ENTRY main {
  accumulator = f32[100, 16] parameter(0)
  big_zero = f32[100,16] fusion(), kind=kCustom, calls=_pop_op_wide_const
  add = f32[100,16] custom-call(accumulator, big_zero), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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
}

TEST_F(SerializeGradientAccumulationTest, TransposeConcatenateWithAdd) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[5, 10] parameter(0)
  p1 = f32[2, 10] parameter(1)
  p2 = f32[2, 10] parameter(2)
  p3 = f32[7, 10] parameter(3)
  p4 = f32[10, 16] parameter(4)
  p5 = f32[10, 16] parameter(5)
  p6 = f32[10, 16] parameter(6)
  accumulator = f32[10, 16] parameter(7)
  a = f32[16, 10] concatenate(p0, p1, p2, p3), dimensions={0}
  a_t = f32[10, 16] transpose(a), dimensions={1,0}
  add1 = f32[10, 16] add(p5, p4)
  add2 = f32[10, 16] add(add1, a_t)
  add3 = f32[10, 16] add(add2, p6)
  add = f32[10, 16] custom-call(accumulator, add3), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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
  EXPECT_EQ(accumulator_add->operand(0), accumulator);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 7);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 6);
    EXPECT_EQ(accumulator_add->operand(6)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 5);
    EXPECT_EQ(accumulator_add->operand(5)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 4);
    EXPECT_EQ(accumulator_add->operand(4)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApply)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 3);
    EXPECT_EQ(accumulator_add->operand(3)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 2);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kAdd);
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 1);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_F(SerializeGradientAccumulationTest, TransposeScaleMultiplyConcatenate) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[5, 10] parameter(0)
  p1 = f32[2, 10] parameter(1)
  p2 = f32[2, 10] parameter(2)
  p3 = f32[7, 10] parameter(3)
  accumulator = f32[10, 16] parameter(4)
  a = f32[16, 10] concatenate(p0, p1, p2, p3), dimensions={0}
  scale = f32[] constant(0.1)
  bscale = f32[16, 10] broadcast(scale), dimensions={}
  m = f32[16, 10] multiply(a, bscale)
  m_t = f32[10, 16] transpose(m), dimensions={1,0}
  add = f32[10, 16] custom-call(accumulator, m_t), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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
  EXPECT_EQ(accumulator_add->operand(0), accumulator);

  HloInstruction* next = accumulator_add->fused_expression_root();
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 5);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
    EXPECT_EQ(accumulator_add->operand(5)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 4);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
    EXPECT_EQ(accumulator_add->operand(4)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 3);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
    EXPECT_EQ(accumulator_add->operand(3)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_TRUE(IsPoplarInstruction(PoplarOp::SliceApplyabY)(next));
    EXPECT_EQ(next->operand(1)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(1)->parameter_number(), 1);
    EXPECT_EQ(next->operand(2)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->operand(2)->parameter_number(), 2);
    EXPECT_EQ(accumulator_add->operand(1)->opcode(), HloOpcode::kTranspose);
  }
  next = next->mutable_operand(0);
  {
    EXPECT_EQ(next->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(next->parameter_number(), 0);
  }
}

TEST_F(SerializeGradientAccumulationTest, TransposeAdds) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  accumulator = f32[100, 16] parameter(0)
  grads1 = f32[16, 100] parameter(1)
  grads2 = f32[16, 100] parameter(2)
  add_grads = f32[16, 100] add(grads1, grads2)
  add_grads_t = f32[100, 16] transpose(add_grads), dimensions={1,0}
  add = f32[100,16] custom-call(accumulator, add_grads_t), custom_call_target="GradientAccumulatorAdd"
  ROOT t = (f32[100, 16]) tuple(add)
}
)";
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

  EXPECT_TRUE(
      Match(accumulator_add->fused_expression_root(),
            m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2))));

  EXPECT_EQ(accumulator_add->operand(0), accumulator);
  EXPECT_EQ(accumulator_add->operand(1)->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(accumulator_add->operand(2)->opcode(), HloOpcode::kTranspose);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
