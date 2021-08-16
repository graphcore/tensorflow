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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_canonicalize.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "absl/memory/memory.h"

#include <poplar/Device.hpp>
#include <poplar/replication_factor.hpp>
#include <poprand/RandomGen.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class SlicePlanTest : public HloTestBase {};

std::unique_ptr<CompilerResources> GetMockResources(HloModule* module,
                                                    bool merge_infeeds) {
  auto resources = CompilerResources::CreateTestDefault(module);
  resources->merge_infeed_io_copies = merge_infeeds;
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), poplar::replication_factor(1));
  poplin::addCodelets(*resources->main_graph);
  popnn::addCodelets(*resources->main_graph);
  popops::addCodelets(*resources->main_graph);
  poprand::addCodelets(*resources->main_graph);
  return std::move(resources);
}

HloPassPipeline GetMockPipeline(CompilerResources& resources) {
  HloPassPipeline pipeline("mock_pipeline");
  pipeline.AddPass<CustomOpReplacer>();
  pipeline.AddPass<MultiUpdateCanonicalize>();
  pipeline.AddPass<ModuleFlatten>(resources.annotations);
  pipeline.AddPass<InplaceFinder>();
  pipeline.AddPass<ShardingPass>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
  pipeline.AddPass<AllocationFinder>(resources.annotations);
  pipeline.AddPass<HloPassFix<ForwardAllocation>>(resources.annotations);
  pipeline.AddPass<HloMemoryScheduler>(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
  return pipeline;
}

TEST_F(SlicePlanTest, LookupsSharedInput) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice"
  ROOT t = (f32[24,16], s32[12,16]) tuple(slice1, slice2)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice1 = root->operand(0);
  auto slice2 = root->operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice1));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, slice2));
  EXPECT_EQ(plan1, plan2);
}

TEST_F(SlicePlanTest, LookupsNotSharedInput) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input1 = f32[100,16] parameter(0)
  input2 = f32[100,16] parameter(1)
  offsets1 = s32[24,1] parameter(2)
  offsets2 = s32[12,1] parameter(3)
  slice1 = f32[24,16] custom-call(input1, offsets1), custom_call_target="MultiSlice"
  slice2 = f32[12,16] custom-call(input2, offsets2), custom_call_target="MultiSlice"
  ROOT t = (f32[24,16], f32[12,16]) tuple(slice1, slice2)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice1 = root->operand(0);
  auto slice2 = root->operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice1));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, slice2));
  EXPECT_NE(plan1, plan2);
}

TEST_F(SlicePlanTest, ShareSliceAndUpdatePlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  slice = f32[24,16] custom-call(input, offsets), custom_call_target="MultiSlice"
  one = f32[] constant(1)
  big_one = f32[24,16] broadcast(one), dimensions={}
  slice_modified = f32[24,16] add(slice, big_one)
  lr = f32[] constant(-0.1)
  update = f32[100,16] custom-call(input, offsets, slice_modified, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  ROOT t = (f32[24,16], f32[100,16]) tuple(slice, update)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice = root->operand(0);
  auto update = root->operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, update));
  EXPECT_EQ(plan1, plan2);
}

TEST_F(SlicePlanTest, ShareMultipleSliceAndUpdatePlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  big_one2 = f32[12,16] broadcast(one), dimensions={}
  slice2_modified = f32[12,16] add(slice2, big_one2)
  lr = f32[] constant(-0.1)
  concat_offsets = s32[36,1] concatenate(offsets1, offsets2), dimensions={0}
  concat_updates = f32[36,16] concatenate(slice1_modified, slice2_modified), dimensions={0}
  update = f32[100,16] custom-call(input, concat_offsets, concat_updates, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  ROOT t = (f32[24,16], f32[12,16], f32[100,16]) tuple(slice1, slice2, update)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice1 = root->operand(0);
  auto slice2 = root->operand(1);
  auto update = root->operand(2);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice1));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, slice2));
  TF_ASSERT_OK_AND_ASSIGN(auto plan3, GetSlicePlan(*resources, update));
  EXPECT_EQ(plan1, plan2);
  EXPECT_EQ(plan1, plan3);
}

TEST_F(SlicePlanTest, DontShareSliceAndUpdatePlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  lr = f32[] constant(-0.1)
  update = f32[100,16] custom-call(input, offsets1, slice1_modified, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  ROOT t = (f32[24,16], f32[12,16], f32[100,16]) tuple(slice1, slice2, update)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice1 = root->operand(0);
  auto slice2 = root->operand(1);
  auto update = root->operand(2);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice1));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, slice2));
  TF_ASSERT_OK_AND_ASSIGN(auto plan3, GetSlicePlan(*resources, update));
  EXPECT_EQ(plan1, plan2);
  EXPECT_NE(plan1, plan3);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
