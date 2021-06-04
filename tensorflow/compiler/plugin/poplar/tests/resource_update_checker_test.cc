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
#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/resource_update_checker.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ResourceUpdateCheckerTest = HloTestBase;

TEST_F(ResourceUpdateCheckerTest, AllOk) {
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
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
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
                          ResourceUpdateChecker().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ResourceUpdateCheckerTest, NonGteUser) {
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
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ROOT t_ru = ((f32[],f32[])) tuple(call_ru)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = ((f32[],f32[])) call(e.weights0, e.weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto status = ResourceUpdateChecker().Run(module.get());
  EXPECT_FALSE(status.ok());
}

TEST_F(ResourceUpdateCheckerTest, DuplicateGte) {
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
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  gte2 = f32[] get-tuple-element(call_ru), index=1
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
  auto status = ResourceUpdateChecker().Run(module.get());
  EXPECT_FALSE(status.ok());
}

TEST_F(ResourceUpdateCheckerTest, MultipleUsers) {
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
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  add1 = f32[] add(gte0, gte1)
  ROOT root = (f32[], f32[]) tuple(gte1, add1)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[]) call(e.weights0, e.weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto status = ResourceUpdateChecker().Run(module.get());
  EXPECT_FALSE(status.ok());
}

TEST_F(ResourceUpdateCheckerTest, MultipleUses) {
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
  call_ru = (f32[],f32[]) call(param0, param1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[] get-tuple-element(call_ru), index=0
  gte1 = f32[] get-tuple-element(call_ru), index=1
  ROOT root = (f32[], f32[], f32[]) tuple(gte1, gte1, gte0)
}

ENTRY e {
  e.weights0 = f32[] parameter(0)
  e.weights1 = f32[] parameter(1)
  ROOT e.call = (f32[], f32[], f32[]) call(e.weights0, e.weights1), to_apply=loop, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto status = ResourceUpdateChecker().Run(module.get());
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
