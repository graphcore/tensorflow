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

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_merger.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using ResourceUpdateMergerTest = HloTestBase;

struct GetNonRUCall {
  const absl::InlinedVector<HloInstruction*, 2> root_operands;

  explicit GetNonRUCall(absl::InlinedVector<HloInstruction*, 2> root_operands)
      : root_operands(root_operands) {
    EXPECT_FALSE(root_operands.empty());
  }

  HloInstruction* operator()(int64 output_tuple_idx) {
    auto merged_ru_gte = root_operands.at(output_tuple_idx);
    EXPECT_EQ(merged_ru_gte->opcode(), HloOpcode::kGetTupleElement);

    auto merged_ru = merged_ru_gte->operands().at(0);
    EXPECT_EQ(merged_ru->opcode(), HloOpcode::kCall);

    // Check it's a resource update.
    EXPECT_TRUE(absl::StrContains(merged_ru->name(), "merged_resource_update"));
    EXPECT_TRUE(IsResourceUpdate(merged_ru));

    auto merged_ru_comp = merged_ru->to_apply();
    auto merged_ru_comp_root = merged_ru_comp->root_instruction();

    auto call_gte =
        merged_ru_comp_root->operands().at(merged_ru_gte->tuple_index());
    EXPECT_EQ(call_gte->opcode(), HloOpcode::kGetTupleElement);

    auto call = call_gte->operands().at(0);
    EXPECT_EQ(call->opcode(), HloOpcode::kCall);
    EXPECT_TRUE(absl::StrContains(call->name(), "_non_ru_call"));
    EXPECT_FALSE(IsResourceUpdate(call));
    return call;
  }
};

TEST_F(ResourceUpdateMergerTest, SingleUpdateNoMerge) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add0 = f32[] add(param0, param1)
  ROOT t = (f32[],f32[]) tuple(add0, param1)
}

ENTRY entry_comp {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)

  ru0 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"

  gte0 = f32[] get-tuple-element(ru0), index=0
  gte1 = f32[] get-tuple-element(ru0), index=1
  ROOT root = (f32[], f32[]) tuple(gte0, gte1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateMerger().Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(ResourceUpdateMergerTest, TwoIndependentUpdatesMerge) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add0 = f32[] add(param0, param1)
  ROOT t = (f32[],f32[]) tuple(add0, param1)
}

ENTRY entry_comp {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)

  ru0 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru1 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"

  gte0 = f32[] get-tuple-element(ru0), index=0
  gte1 = f32[] get-tuple-element(ru0), index=1
  gte2 = f32[] get-tuple-element(ru1), index=0
  gte3 = f32[] get-tuple-element(ru1), index=1
  ROOT root = (f32[], f32[], f32[], f32[]) tuple(gte0, gte1, gte2, gte3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_changed,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateMerger().Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify the shape of the root instruction.
  const auto root = module->entry_computation()->root_instruction();
  const auto root_operands = root->operands();
  EXPECT_EQ(root_operands.size(), 4);

  GetNonRUCall get_non_ru_call(root_operands);

  auto non_ru0 = get_non_ru_call(0);
  auto non_ru1 = get_non_ru_call(1);
  EXPECT_EQ(non_ru0, non_ru1);

  auto non_ru2 = get_non_ru_call(2);
  auto non_ru3 = get_non_ru_call(3);
  EXPECT_EQ(non_ru2, non_ru3);

  EXPECT_NE(non_ru0, non_ru2);
  EXPECT_EQ(non_ru0->to_apply(), non_ru2->to_apply());
}

TEST_F(ResourceUpdateMergerTest, TwoDependentUpdatesMerge) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add0 = f32[] add(param0, param1)
  ROOT t = (f32[],f32[]) tuple(add0, param1)
}

ENTRY entry_comp {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)

  ru0 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(ru0), index=0
  gte1 = f32[] get-tuple-element(ru0), index=1

  ru1 = (f32[],f32[]) call(gte0, gte1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte2 = f32[] get-tuple-element(ru1), index=0
  gte3 = f32[] get-tuple-element(ru1), index=1
  
  ROOT root = (f32[], f32[]) tuple(gte2, gte3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_changed,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateMerger().Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify the shape of the root instruction.
  const auto root = module->entry_computation()->root_instruction();
  const auto root_operands = root->operands();
  EXPECT_EQ(root_operands.size(), 2);

  GetNonRUCall get_non_ru_call(root_operands);

  auto non_ru0 = get_non_ru_call(0);
  auto non_ru1 = get_non_ru_call(1);

  EXPECT_EQ(non_ru0, non_ru1);
  EXPECT_EQ(non_ru0->to_apply(), non_ru1->to_apply());

  auto check_non_ru_call_operand = [](HloInstruction* non_ru_call,
                                      int64 op_idx) {
    EXPECT_EQ(non_ru_call->name(), "ru1_non_ru_call");

    auto gte = non_ru_call->operands().at(op_idx);
    EXPECT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);

    auto call = gte->operands().at(0);
    EXPECT_EQ(call->opcode(), HloOpcode::kCall);
    EXPECT_EQ(call->name(), "ru0_non_ru_call");
  };

  check_non_ru_call_operand(non_ru0, 0);
  check_non_ru_call_operand(non_ru0, 1);
  check_non_ru_call_operand(non_ru1, 0);
  check_non_ru_call_operand(non_ru1, 1);
}

TEST_F(ResourceUpdateMergerTest, TwoIndependentDifferingUpdatesMerge) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add0 = f32[] add(param0, param1)
  ROOT t = (f32[],f32[]) tuple(add0, param1)
}

resource_update_second {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  sub0 = f32[] subtract(param0, param1)
  ROOT t = (f32[]) tuple(sub0)
}

ENTRY entry_comp {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)

  ru0 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru1 = (f32[]) call(param0, param1), to_apply=resource_update_second, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"

  gte0 = f32[] get-tuple-element(ru0), index=0
  gte1 = f32[] get-tuple-element(ru0), index=1
  gte2 = f32[] get-tuple-element(ru1), index=0
  ROOT root = (f32[], f32[], f32[]) tuple(gte0, gte1, gte2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_changed,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateMerger().Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify the shape of the root instruction.
  const auto root = module->entry_computation()->root_instruction();
  const auto root_operands = root->operands();
  EXPECT_EQ(root_operands.size(), 3);

  GetNonRUCall get_non_ru_call(root_operands);

  auto non_ru0 = get_non_ru_call(0);
  auto non_ru1 = get_non_ru_call(1);
  EXPECT_EQ(non_ru0, non_ru1);
  EXPECT_EQ(non_ru0->to_apply(), non_ru1->to_apply());

  auto ru0_math_op = non_ru0->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru0_math_op->opcode(), HloOpcode::kAdd);

  auto ru1_math_op = non_ru1->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru1_math_op->opcode(), HloOpcode::kAdd);

  auto non_ru2 = get_non_ru_call(2);
  EXPECT_NE(non_ru0, non_ru2);
  EXPECT_NE(non_ru0->to_apply(), non_ru2->to_apply());

  auto ru2_math_op = non_ru2->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru2_math_op->opcode(), HloOpcode::kSubtract);
}

TEST_F(ResourceUpdateMergerTest, TwoDependentDifferingUpdatesMerge) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  add0 = f32[] add(param0, param1)
  ROOT t = (f32[],f32[]) tuple(add0, param1)
}

resource_update_second {
  gac_c = s32[] constant(2)
  gac = () custom-call(s32[] gac_c), custom_call_target="GradientAccumulationCount"

  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  sub0 = f32[] subtract(param0, param1)
  ROOT t = (f32[]) tuple(sub0)
}

ENTRY entry_comp {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)

  ru0 = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(ru0), index=0
  gte1 = f32[] get-tuple-element(ru0), index=1

  ru1 = (f32[]) call(gte0, gte1), to_apply=resource_update_second, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"

  gte2 = f32[] get-tuple-element(ru1), index=0
  gte3 = f32[] get-tuple-element(ru0), index=0
  gte4 = f32[] get-tuple-element(ru0), index=1
  ROOT root = (f32[], f32[], f32[]) tuple(gte3, gte4, gte2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_ops_changed,
                          CustomOpReplacer().Run(module.get()));
  EXPECT_TRUE(custom_ops_changed);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateMerger().Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify the shape of the root instruction.
  const auto root = module->entry_computation()->root_instruction();
  const auto root_operands = root->operands();
  EXPECT_EQ(root_operands.size(), 3);

  GetNonRUCall get_non_ru_call(root_operands);

  auto non_ru0 = get_non_ru_call(0);
  auto non_ru1 = get_non_ru_call(1);
  auto non_ru2 = get_non_ru_call(2);

  EXPECT_EQ(non_ru0, non_ru1);
  EXPECT_EQ(non_ru0->to_apply(), non_ru1->to_apply());

  EXPECT_NE(non_ru1, non_ru2);
  EXPECT_NE(non_ru1->to_apply(), non_ru2->to_apply());

  auto check_non_ru_call_operand_ru0 = [](HloInstruction* non_ru_call,
                                          int64 op_idx) {
    EXPECT_EQ(non_ru_call->name(), "ru0_non_ru_call");

    auto param = non_ru_call->operands().at(op_idx);
    EXPECT_EQ(param->opcode(), HloOpcode::kParameter);
  };

  check_non_ru_call_operand_ru0(non_ru0, 0);
  check_non_ru_call_operand_ru0(non_ru0, 1);
  check_non_ru_call_operand_ru0(non_ru1, 0);
  check_non_ru_call_operand_ru0(non_ru1, 1);

  auto check_non_ru_call_operand_ru1 = [](HloInstruction* non_ru_call,
                                          int64 op_idx) {
    EXPECT_EQ(non_ru_call->name(), "ru1_non_ru_call");

    auto gte = non_ru_call->operands().at(op_idx);
    EXPECT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);

    auto call = gte->operands().at(0);
    EXPECT_EQ(call->opcode(), HloOpcode::kCall);
    EXPECT_EQ(call->name(), "ru0_non_ru_call");
  };

  check_non_ru_call_operand_ru1(non_ru2, 0);
  check_non_ru_call_operand_ru1(non_ru2, 1);

  auto ru0_math_op = non_ru0->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru0_math_op->opcode(), HloOpcode::kAdd);

  auto ru1_math_op = non_ru1->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru1_math_op->opcode(), HloOpcode::kAdd);

  auto ru2_math_op = non_ru2->to_apply()->root_instruction()->operands().at(0);
  EXPECT_EQ(ru2_math_op->opcode(), HloOpcode::kSubtract);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
