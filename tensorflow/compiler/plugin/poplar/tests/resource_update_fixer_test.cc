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

#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_fixer.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using ResourceUpdateFixerTest = HloTestBase;

TEST_F(ResourceUpdateFixerTest, NothingToFix) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  add0 = f32[] add(ru_arg0, ru_arg1)
  ROOT t = (f32[],f32[]) tuple(add0, ru_arg1)
}

loop {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT root = (f32[], f32[]) tuple(gte1, gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateFixer().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ResourceUpdateFixerTest, FixTuple) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  add0 = f32[] add(ru_arg0, ru_arg1)
  ROOT t = (f32[],f32[]) tuple(add0, ru_arg1)
}

loop {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  ROOT call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateFixer().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* comp = FindComputation(module.get(), "loop");
  HloInstruction *call0, *call1;
  EXPECT_TRUE(Match(comp->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op(&call0), 0),
                             m::GetTupleElement(m::Op(&call1), 1))));
  EXPECT_EQ(call0, call1);
  EXPECT_TRUE(IsResourceUpdate(call0));
}

TEST_F(ResourceUpdateFixerTest, DuplicateOutputs) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  ROOT t = (f32[],f32[]) tuple(ru_arg1, ru_arg0)
}

loop {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  param2 = f32[] parameter(2)
  param3 = f32[] parameter(3)
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  gte2 = f32[] get-tuple-element(call_ru), index=1
  ROOT root = (f32[], f32[], f32[], f32[]) tuple(gte1, gte0, gte0, gte2)
}

ENTRY e {
  e0 = f32[] parameter(0)
  e1 = f32[] parameter(1)
  e2 = f32[] parameter(2)
  e3 = f32[] parameter(3)
  ROOT e.call = (f32[], f32[], f32[], f32[]) call(e0, e1, e2, e3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateFixer().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* comp = FindComputation(module.get(), "loop");
  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  EXPECT_TRUE(
      Match(resource_update->root_instruction(),
            m::Tuple(m::Parameter(1), m::Parameter(0), m::Copy(m::Parameter(1)),
                     m::Copy(m::Parameter(0)))));

  HloInstruction *call0, *call1, *call2, *call3;
  EXPECT_TRUE(Match(comp->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op(&call0), 1),
                             m::GetTupleElement(m::Op(&call1), 0),
                             m::GetTupleElement(m::Op(&call2), 2),
                             m::GetTupleElement(m::Op(&call3), 3))));
  EXPECT_EQ(call0, call1);
  EXPECT_EQ(call0, call2);
  EXPECT_EQ(call0, call3);
  EXPECT_TRUE(IsResourceUpdate(call0));
}

TEST_F(ResourceUpdateFixerTest, DuplicateOutputsAndUsers) {
  const std::string hlo_string = R"(
HloModule top

resource_update {
  ru_arg0 = f32[] parameter(0)
  ru_arg1 = f32[] parameter(1)
  ROOT root = (f32[],f32[]) tuple(ru_arg1, ru_arg0)
  t = token[] after-all()
  outfeed = token[] outfeed(root, t)
}

loop {
  param0 = f32[] parameter(0)
  param1 = f32[] parameter(1)
  param2 = f32[] parameter(2)
  param3 = f32[] parameter(3)
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  gte2 = f32[] get-tuple-element(call_ru), index=1
  ROOT root = (f32[], f32[], f32[], f32[]) tuple(gte1, gte0, gte0, gte2)
}

ENTRY e {
  e0 = f32[] parameter(0)
  e1 = f32[] parameter(1)
  e2 = f32[] parameter(2)
  e3 = f32[] parameter(3)
  ROOT e.call = (f32[], f32[], f32[], f32[]) call(e0, e1, e2, e3), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ResourceUpdateFixer().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* resource_update =
      FindComputation(module.get(), "resource_update");
  HloInstruction* ru_root = resource_update->root_instruction();
  EXPECT_TRUE(Match(
      ru_root, m::Tuple(m::Parameter(1), m::Parameter(0),
                        m::Copy(m::Parameter(1)), m::Copy(m::Parameter(0)))));
  HloInstruction* outfeed = FindInstruction(module.get(), "outfeed");
  EXPECT_TRUE(Match(
      outfeed, m::Outfeed(m::Tuple(m::GetTupleElement(m::Op().Is(ru_root), 0),
                                   m::GetTupleElement(m::Op().Is(ru_root), 1)),
                          m::AfterAll())));

  HloComputation* comp = FindComputation(module.get(), "loop");
  HloInstruction *call0, *call1, *call2, *call3;
  EXPECT_TRUE(Match(comp->root_instruction(),
                    m::Tuple(m::GetTupleElement(m::Op(&call0), 1),
                             m::GetTupleElement(m::Op(&call1), 0),
                             m::GetTupleElement(m::Op(&call2), 2),
                             m::GetTupleElement(m::Op(&call3), 3))));
  EXPECT_EQ(call0, call1);
  EXPECT_EQ(call0, call2);
  EXPECT_EQ(call0, call3);
  EXPECT_TRUE(IsResourceUpdate(call0));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
