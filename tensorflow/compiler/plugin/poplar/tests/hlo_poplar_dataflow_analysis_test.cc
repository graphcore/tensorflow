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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloPoplarDataflowAnalysisTest = HloTestBase;

TEST_F(HloPoplarDataflowAnalysisTest, TestCallGraphNotFlat) {
  std::string hlo = R"(
 HloModule top

comp {
  i0 = (f32[2], f32[2]) parameter(0)
  i1 = f32[2] get-tuple-element(i0), index=0
  i2 = f32[2] get-tuple-element(i0), index=1
  i3 = f32[2] sine(i1)
  i4 = f32[2] sine(i2)
  ROOT i5 = f32[2] add(i3, i4)
}

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = (f32[2], f32[2]) parameter(1)
  gte1.0 = f32[2] get-tuple-element(arg1), index=0
  gte1.1 = f32[2] get-tuple-element(arg1), index=1
  arg2 = f32[2] parameter(2)
  t0 = (f32[2], f32[2]) tuple(arg0, gte1.0)
  t1 = (f32[2], f32[2]) tuple(arg2, gte1.1)
  c0 = f32[2] call(t0), to_apply=comp
  c1 = f32[2] call(t1), to_apply=comp
  ROOT t2 = (f32[2], f32[2]) tuple(c0, c1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  EXPECT_FALSE(HloPoplarDataflowAnalysis::Run(m.get(), annotations).ok());
}

TEST_F(HloPoplarDataflowAnalysisTest, TestSimpleInplaceReadWrite) {
  std::string hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  sine0 = f32[2] sine(arg0)
  sine1 = f32[2] sine(arg1)
  ROOT add = f32[2] add(sine0, sine1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 2);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* arg1 = FindInstruction(m.get(), "arg1");
  HloInstruction* sine0 = FindInstruction(m.get(), "sine0");
  HloInstruction* sine1 = FindInstruction(m.get(), "sine1");
  HloInstruction* add = FindInstruction(m.get(), "add");

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  auto arg1_buffer = analysis->GetUniqueBufferAt(arg1);
  EXPECT_NE(arg0_buffer, arg1_buffer);

  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(sine0));
  EXPECT_EQ(arg1_buffer, analysis->GetUniqueBufferAt(sine1));

  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(add));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestSimpleInplaceReadOnly) {
  std::string hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  ROOT reshape = f32[2,1] reshape(arg0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 1);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* reshape = FindInstruction(m.get(), "reshape");

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(reshape));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestSimpleLogicalBinaryElementwise) {
  std::string hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = s32[2] parameter(0)
  arg1 = s32[2] parameter(1)
  ROOT and0 = s32[2] and(arg0, arg1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 2);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* arg1 = FindInstruction(m.get(), "arg1");
  HloInstruction* and0 = FindInstruction(m.get(), "and0");

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  auto arg1_buffer = analysis->GetUniqueBufferAt(arg1);
  EXPECT_NE(arg0_buffer, arg1_buffer);
  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(and0));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestConstant) {
  std::string hlo = R"(
 HloModule top

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  c0 = f32[2] constant({0.1, 0.2})
  ROOT add = f32[2] add(arg0, c0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 2);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* c0 = FindInstruction(m.get(), "c0");
  HloInstruction* add = FindInstruction(m.get(), "add");

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  auto c0_buffer = analysis->GetUniqueBufferAt(c0);
  EXPECT_NE(arg0_buffer, c0_buffer);
  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(add));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestPopopsFusionNoAlias) {
  std::string hlo = R"(
 HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

 ENTRY cluster_1 {
  arg0 = f16[2] parameter(0)
  convert0 = f32[2] convert(arg0)
  c0 = f32[] constant(10.0)
  ROOT reduce = f32[] reduce(convert0, c0), dimensions={0}, to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());
  EXPECT_TRUE(FuseOpsLate(annotations).Run(m.get()).ValueOrDie());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 3);

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPopOpsFusion(root));
  EXPECT_TRUE(analysis->BufferIsDefinedAt(root));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestPopopsFusionAlias) {
  std::string hlo = R"(
 HloModule top

identity {
  param0 = f32[] parameter(0)
  ROOT param1 = f32[] parameter(1)
}

ENTRY reduce-window-identity {
  arg0 = f32[1,32,64] parameter(0)
  c0 = f32[] constant(0)
  ROOT reduce-window = f32[1,32,64] reduce-window(arg0, c0), window={size=1x1x1 pad=0_0x0_0x0_0}, to_apply=identity
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());
  EXPECT_TRUE(FuseOpsLate(annotations).Run(m.get()).ValueOrDie());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 2);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* c0 = FindInstruction(m.get(), "c0");
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPopOpsFusion(root));

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  auto c0_buffer = analysis->GetUniqueBufferAt(c0);
  EXPECT_NE(arg0_buffer, c0_buffer);
  HloPoplarBufferSet union_set({&arg0_buffer, &c0_buffer});
  EXPECT_THAT(analysis->GetBufferSet(root), union_set);
}

TEST_F(HloPoplarDataflowAnalysisTest, TestNonPopopsFusion) {
  std::string hlo = R"(
HloModule top
fused_computation {
  p0 = f32[10,20,30] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  slice = f32[1,1,30] dynamic-slice(p0, p1, p2, p3), dynamic_slice_sizes={1,1,30}
  ROOT dus = f32[10,20,30] dynamic-update-slice(p0, slice, p1, p3, p2)
}

ENTRY test {
  arg0 = f32[10,20,30] parameter(0)
  arg1 = s32[] parameter(1)
  arg2 = s32[] parameter(2)
  arg3 = s32[] parameter(3)
  ROOT fusion = f32[10,20,30] fusion(arg0, arg1, arg2, arg3), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 5);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* fusion = FindInstruction(m.get(), "fusion");
  EXPECT_EQ(analysis->GetUniqueBufferAt(arg0),
            analysis->GetUniqueBufferAt(fusion));
}

TEST_F(HloPoplarDataflowAnalysisTest, TestCustomCall) {
  std::string hlo = R"(
HloModule top
ENTRY main {
  arg0 = f32[] parameter(0)
  arg1 = f32[] parameter(1)

  arg0_load = f32[] custom-call(arg0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  add0 = f32[] add(arg1, arg0_load)

  ROOT arg0_store = f32[] custom-call(arg0, add0), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto annotations = CompilerAnnotations(m.get());
  // Mark arg0 as in remote memory.
  annotations.remote_parameter_infos.emplace(0);

  ASSERT_TRUE(CustomOpReplacer().Run(m.get()).ValueOrDie());
  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloPoplarDataflowAnalysis::Run(m.get(), annotations));

  EXPECT_THAT(analysis->buffer_count(), 3);

  HloInstruction* arg0 = FindInstruction(m.get(), "arg0");
  HloInstruction* arg1 = FindInstruction(m.get(), "arg1");
  HloInstruction *arg0_load, *arg0_store;
  TF_ASSERT_OK(GetRemoteLoadStoreUsers(arg0, &arg0_load, &arg0_store));

  EXPECT_TRUE(analysis->BufferIsDefinedAt(arg0));
  EXPECT_TRUE(analysis->BufferIsDefinedAt(arg1));
  EXPECT_TRUE(analysis->BufferIsDefinedAt(arg0_load));

  auto arg0_buffer = analysis->GetUniqueBufferAt(arg0);
  EXPECT_EQ(arg0_buffer.locality(), BufferLocality::kRemoteMemory);

  auto arg0_load_buffer = analysis->GetUniqueBufferAt(arg0_load);
  EXPECT_EQ(arg0_load_buffer.locality(), BufferLocality::kDeviceMemory);

  EXPECT_EQ(arg0_buffer, analysis->GetUniqueBufferAt(arg0_store));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
