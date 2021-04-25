/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_casts.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

namespace xla {
namespace poplarplugin {
namespace {

struct RecomputeCastsTest : HloTestBase {
  RecomputeCastsTest() : pipeline_("test") {}

  void SetUpModule(const std::string& hlo) {
    ASSERT_FALSE(hlo.empty());

    auto module_result = ParseAndReturnVerifiedModule(hlo);
    ASSERT_TRUE(module_result.ok());

    module_owner_ = module_result.ConsumeValueOrDie();
    module_ = module_owner_.get();
  }

  const HloInstruction* GetOperandForInstruction(
      const std::string& instruction_name, int64 index) {
    const auto instruction = FindInstruction(module_, instruction_name);
    return instruction->operand(index);
  }

  HloPassPipeline pipeline_;
  VerifiedHloModule* module_;

 private:
  std::unique_ptr<VerifiedHloModule> module_owner_;
};

const char* recomputable_cast =
    R"(
HloModule main

ENTRY main {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  param2 = f16[] parameter(2)
  cast = f32[] convert(f16[] param2)
  add1 = f32[] add(f32[] param0, f32[] cast)
  add2 = f32[] add(f32[] param1, f32[] cast)
  ROOT result = f32[] add(f32[] add1, f32[] add2)
}
)";
TEST_F(RecomputeCastsTest, CastConsumersGetUniqueCloneOfCast) {
  SetUpModule(recomputable_cast);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  const auto add1_cast = GetOperandForInstruction("add1", 1);
  const auto add2_cast = GetOperandForInstruction("add2", 1);

  ASSERT_EQ(add1_cast->opcode(), HloOpcode::kConvert);
  ASSERT_EQ(add2_cast->opcode(), HloOpcode::kConvert);
  ASSERT_EQ(add1_cast->user_count(), 1);
  ASSERT_EQ(add2_cast->user_count(), 1);
  ASSERT_NE(add1_cast, add2_cast);
}

TEST_F(RecomputeCastsTest, ClonedCastIsRemoved) {
  SetUpModule(recomputable_cast);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  // 'cast' is the instruction that gets cloned for each consumer and
  // hence is not used anymore and should be removed.
  const auto cast = FindInstruction(module_, "cast");
  ASSERT_FALSE(cast);
}

TEST_F(RecomputeCastsTest, ClonedCastsInheritControlDependencies) {
  using ::testing::Contains;

  SetUpModule(recomputable_cast);

  // Add control dependencies we expect to be inherited by clones.
  const auto param0 = FindInstruction(module_, "param0");
  const auto cast = FindInstruction(module_, "cast");
  const auto result = FindInstruction(module_, "result");
  param0->AddControlDependencyTo(cast);
  cast->AddControlDependencyTo(result);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  const auto add1_cast = GetOperandForInstruction("add1", 1);
  const auto add2_cast = GetOperandForInstruction("add2", 1);

  ASSERT_THAT(add1_cast->control_predecessors(), Contains(param0));
  ASSERT_THAT(add2_cast->control_predecessors(), Contains(param0));

  const auto param0_successors = param0->control_successors();
  ASSERT_EQ(param0_successors.size(), 2);

  ASSERT_THAT(add1_cast->control_successors(), Contains(result));
  ASSERT_THAT(add2_cast->control_successors(), Contains(result));

  const auto result_dependencies = result->control_predecessors();
  ASSERT_EQ(result_dependencies.size(), 2);
}

TEST_F(RecomputeCastsTest, ClonedCastSkipControlSuccessorThatIntroducesCycle) {
  using ::testing::IsEmpty;

  SetUpModule(recomputable_cast);

  // Add a control dependency from cast -> param0. If this gets copied
  // by the add1 cast then it'd introduce a cycle since we'd have
  // param0 -> cast.clone -> param0
  const auto param0 = FindInstruction(module_, "param0");
  const auto cast = FindInstruction(module_, "cast");
  cast->AddControlDependencyTo(param0);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  const auto add1_cast = GetOperandForInstruction("add1", 1);

  ASSERT_THAT(add1_cast->control_successors(), IsEmpty());
}

const char* non_parameter_cast =
    R"(
HloModule main

ENTRY main {
  a = f16[] parameter(0)
  b = f16[] parameter(1)
  c = f32[] parameter(2)
  add1 = f16[] add(f16[] a, f16[] b)
  cast = f32[] convert(f16[] add1)
  add2 = f32[] add(f32[] c, f32[] cast)
  ROOT root = f32[] add(f32[] add2, f32[] cast)
}
)";
TEST_F(RecomputeCastsTest, OnlyParameterCastsGetCloned) {
  SetUpModule(non_parameter_cast);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_FALSE(pipeline_.Run(module_).ValueOrDie());

  const auto add2_cast = GetOperandForInstruction("add2", 1);
  const auto root_cast = GetOperandForInstruction("root", 1);

  ASSERT_EQ(add2_cast, root_cast);
  ASSERT_EQ(add2_cast->opcode(), HloOpcode::kConvert);
  ASSERT_EQ(add2_cast->user_count(), 2);
}

const char* cast_needs_control_dependency =
    R"(
HloModule main

ENTRY main {
  a = f32[] parameter(0)
  c = f16[] parameter(2)
  cast = f32[] convert(f16[] c)
  b = f32[] parameter(1)
  d = f32[] parameter(3)
  clamp1 = f32[] clamp(f32[] cast, f32[] a, f32[] b)
  clamp2 = f32[] clamp(f32[] cast, f32[] clamp1, f32[] d)
  ROOT f = f32[] add(f32[] clamp1, f32[] clamp2)
}
)";
TEST_F(RecomputeCastsTest, CastsAreLastProcessedOperand) {
  using ::testing::AllOf;
  using ::testing::Gt;

  SetUpModule(cast_needs_control_dependency);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  const auto ordered_instructions =
      module_->entry_computation()->MakeInstructionPostOrder();
  const auto instruction_pos =
      [&ordered_instructions](const HloInstruction* inst) {
        return std::find(ordered_instructions.begin(),
                         ordered_instructions.end(), inst);
      };

  for (auto instruction_name : std::vector<std::string>{"clamp1", "clamp2"}) {
    const auto clamp = FindInstruction(module_, instruction_name);

    const auto cast = clamp->operand(0);
    ASSERT_EQ(cast->opcode(), HloOpcode::kConvert);

    const auto cast_pos = instruction_pos(cast);
    const auto operand1_pos = instruction_pos(clamp->operand(1));
    const auto operand2_pos = instruction_pos(clamp->operand(2));

    ASSERT_THAT(cast_pos, AllOf(Gt(operand1_pos), Gt(operand2_pos)));
  }
}

TEST_F(RecomputeCastsTest, TestTwoCastsSameOp) {
  const std::string hlo =
      R"(
HloModule main

ENTRY main {
  a = f16[] parameter(0)
  b = f16[] parameter(1)
  cast1 = f32[] convert(a)
  cast2 = f32[] convert(b)
  add1 = f32[] add(cast1, cast2)
  log = f32[] log(add1)
  add2 = f32[] add(cast1, cast2), control-predecessors={log}
  ROOT root = f32[] add(log, add2)
}
)";

  SetUpModule(hlo);

  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_TRUE(pipeline_.Run(module_).ValueOrDie());

  auto a = FindInstruction(module_, "a");
  auto b = FindInstruction(module_, "b");
  auto add1 = FindInstruction(module_, "add1");
  auto add2 = FindInstruction(module_, "add2");

  auto cast1_0 = add1->operand(0);
  EXPECT_THAT(cast1_0->operands(), ::testing::ElementsAre(a));
  auto cast1_1 = add2->operand(0);
  EXPECT_THAT(cast1_1->operands(), ::testing::ElementsAre(a));

  auto cast2_0 = add1->operand(1);
  EXPECT_THAT(cast2_0->operands(), ::testing::ElementsAre(b));
  auto cast2_1 = add2->operand(1);
  EXPECT_THAT(cast2_1->operands(), ::testing::ElementsAre(b));

  EXPECT_THAT(cast1_0->control_predecessors(), ::testing::ElementsAre());
  EXPECT_THAT(cast2_0->control_predecessors(), ::testing::ElementsAre(cast1_0));
  EXPECT_THAT(cast1_1->control_predecessors(), ::testing::ElementsAre(cast2_0));
  EXPECT_THAT(cast2_1->control_predecessors(), ::testing::ElementsAre(cast1_1));
}

struct BlockedRecomputeCastTest : RecomputeCastsTest,
                                  ::testing::WithParamInterface<std::string> {
  void SetUp() override { SetUpModule(GetParam()); }
};

// This test expects the hlo it gets to contain a cast used by 2 adds,
// e.g..
//  cast = f32[] convert(f16[] c)
//  add1 = f32[] add(f32[] cast, f32[] a)
//  add2 = f32[] add(f32[] cast, f32[] b)
TEST_P(BlockedRecomputeCastTest, CastsAreNotCloned) {
  pipeline_.AddPass<RecomputeCasts>();
  ASSERT_FALSE(pipeline_.Run(module_).ValueOrDie());

  const auto add1 = FindInstruction(module_, "add1");
  const auto add2 = FindInstruction(module_, "add2");

  ASSERT_TRUE(add1);
  ASSERT_TRUE(add2);

  const auto add1_cast = add1->operand(0);
  const auto add2_cast = add2->operand(0);

  ASSERT_EQ(add1_cast, add2_cast);
  ASSERT_EQ(add1_cast->opcode(), HloOpcode::kConvert);
  ASSERT_EQ(add1_cast->user_count(), 2);
}

const char* casts_in_fused_call =
    R"(
HloModule main

_pop_op_test (a: f32[], b: f32[], c: f16[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f16[] parameter(2)
  cast = f32[] convert(f16[] c)
  add1 = f32[] add(f32[] cast, f32[] a)
  add2 = f32[] add(f32[] cast, f32[] b)
  ROOT f = f32[] add(f32[] add1, f32[] add2)
}

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f16[] parameter(2)
  ROOT f = f32[] fusion(f32[] a, f32[] b, f16[] c), kind=kCustom, calls=_pop_op_test
}
)";

const char* casts_in_resource_update =
    R"(
HloModule top

stage_0_fwd {
  t = token[] after-all()
  feed = (f32[], token[]) infeed(t)
  input = f32[] get-tuple-element(feed), index=0
  ROOT stage_0_fwd_tuple = (f32[]) tuple(input)
}

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  ru_arg2 = f16[] parameter(2)
  cast = f32[] convert(f16[] ru_arg2)
  add1 = f32[] add(cast, ru_arg1)
  add2 = f32[] add(cast, ru_arg0)
  ROOT t = (f32[],f32[]) tuple(add1, add2)
}

pipeline {
  param0 = f32[] parameter(0), parameter_replication={false}
  param1 = f16[] parameter(1), parameter_replication={false}
  pipeline_stage_0 = (f32[]) call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0

  call_ru = (f32[],f32[]) call(stage_0_fwd.0, param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f16[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

const char* casts_in_resource_update_subcall =
    R"(
HloModule top

stage_0_fwd {
  t = token[] after-all()
  feed = (f32[], token[]) infeed(t)
  input = f32[] get-tuple-element(feed), index=0
  ROOT stage_0_fwd_tuple = (f32[]) tuple(input)
}

TestCall (a: f32[], b: f32[], c: f16[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f16[] parameter(2)
  cast = f32[] convert(f16[] c)
  add1 = f32[] add(f32[] cast, f32[] a)
  add2 = f32[] add(f32[] cast, f32[] b)
  ROOT f = f32[] add(f32[] add1, f32[] add2)
}

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  ru_arg2 = f16[] parameter(2)
  result = f32[] call(ru_arg0, ru_arg1, ru_arg2), to_apply=TestCall
  ROOT t = (f32[],f32[]) tuple(result, result)
}

pipeline {
  param0 = f32[] parameter(0), parameter_replication={false}
  param1 = f16[] parameter(1), parameter_replication={false}
  pipeline_stage_0 = (f32[]) call(), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  stage_0_fwd.0 = f32[] get-tuple-element(pipeline_stage_0), index=0

  call_ru = (f32[],f32[]) call(stage_0_fwd.0, param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f16[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";

std::string TestName(
    const ::testing::TestParamInfo<BlockedRecomputeCastTest::ParamType>& info) {
  if (info.param == casts_in_fused_call) {
    return "casts_in_fused_call";
  }
  if (info.param == casts_in_resource_update) {
    return "casts_in_resource_update";
  }

  if (info.param == casts_in_resource_update_subcall) {
    return "casts_in_resource_update_subcall";
  }

  return "Unknown";
}

INSTANTIATE_TEST_SUITE_P(Recompute, BlockedRecomputeCastTest,
                         ::testing::Values(casts_in_fused_call,
                                           casts_in_resource_update,
                                           casts_in_resource_update_subcall),
                         std::bind(TestName, std::placeholders::_1));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
