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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_canonicalizer.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using RemoteBufferCanonicalizerTest = HloTestBase;

TEST_F(RemoteBufferCanonicalizerTest, TestLoadStore) {
  const std::string hlo = R"(
HloModule top
ENTRY e {
  param0 = f32[] parameter(0)
  param1 = f32[1] parameter(1)
  
  param0_load = f32[] custom-call(param0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_reshape = f32[1] reshape(param0_load)
  add = f32[1] add(load_reshape, param1)
  store_reshape = f32[] reshape(add)
  param0_store = f32[] custom-call(param0, store_reshape), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  ROOT t = (f32[]) tuple(param0_store)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RemoteBufferCanonicalizer(annotations).Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();

  HloInstruction* store;
  EXPECT_TRUE(Match(root, m::Tuple(m::Reshape(m::Op(&store)))));
  EXPECT_TRUE(IsPoplarInstruction(RemoteParameterStore, store));

  HloInstruction *in0, *in1;
  EXPECT_TRUE(Match(
      store, m::CustomCall(m::Op(&in0), m::Add(m::Op(&in1), m::Parameter(1)))));
  EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, in1));
  EXPECT_TRUE(Match(in1, m::CustomCall((m::Op().Is(in0)))));
  EXPECT_TRUE(Match(in0, m::Reshape(m::Parameter(0))));
}

TEST_F(RemoteBufferCanonicalizerTest, TestLoadOnly) {
  const std::string hlo = R"(
HloModule top
ENTRY e {
  param0 = f32[] parameter(0)
  param1 = f32[1] parameter(1)
  
  param0_load = f32[] custom-call(param0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  load_reshape = f32[1] reshape(param0_load)
  ROOT t = (f32[]) tuple(load_reshape)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());

  CompilerAnnotations annotations(module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RemoteBufferCanonicalizer(annotations).Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* load;
  EXPECT_TRUE(Match(root, m::Tuple(m::Op(&load))));
  EXPECT_TRUE(IsPoplarInstruction(RemoteParameterLoad, load));
  EXPECT_TRUE(Match(load, m::CustomCall(m::Reshape(m::Parameter(0)))));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
