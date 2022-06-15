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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
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

using SlicePlanTest = HloPoplarTestBase;

HloPassPipeline GetMockPipeline(CompilerResources& resources) {
  HloPassPipeline pipeline("mock_pipeline");
  pipeline.AddPass<CustomOpReplacer>();
  pipeline.AddPass<ModuleFlatten>(resources.annotations);
  pipeline.AddPass<InplaceFinder>(resources.annotations);
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
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
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
  slice1 = f32[24,16] custom-call(input1, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input2, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
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

TEST_F(SlicePlanTest, ShareSliceAndUpdateAddPlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  slice = f32[24,16] custom-call(input, offsets), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one = f32[24,16] broadcast(one), dimensions={}
  slice_modified = f32[24,16] add(slice, big_one)
  lr = f32[] constant(-0.1)
  update = f32[100,16] custom-call(input, offsets, slice_modified, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
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

TEST_F(SlicePlanTest, ShareMultipleSliceAndUpdateAddPlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  big_one2 = f32[12,16] broadcast(one), dimensions={}
  slice2_modified = f32[12,16] add(slice2, big_one2)
  lr = f32[] constant(-0.1)
  concat_offsets = s32[36,1] concatenate(offsets1, offsets2), dimensions={0}
  concat_updates = f32[36,16] concatenate(slice1_modified, slice2_modified), dimensions={0}
  update = f32[100,16] custom-call(input, concat_offsets, concat_updates, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
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

TEST_F(SlicePlanTest, ShareMultipleSlicesAndUpdate) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  lr = f32[] constant(-0.1)
  update = f32[100,16] custom-call(input, offsets1, slice1_modified, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
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

TEST_F(SlicePlanTest, ShareSliceAndUpdatePlan) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  slice = f32[24,16] custom-call(input, offsets), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one = f32[24,16] broadcast(one), dimensions={}
  slice_modified = f32[24,16] add(slice, big_one)
  update = f32[100,16] custom-call(input, offsets, slice_modified), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
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

TEST_F(SlicePlanTest, ShareMultipleSliceAndUpdatePlan2) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  big_one2 = f32[12,16] broadcast(one), dimensions={}
  slice2_modified = f32[12,16] add(slice2, big_one2)
  concat_offsets = s32[36,1] concatenate(offsets1, offsets2), dimensions={0}
  concat_updates = f32[36,16] concatenate(slice1_modified, slice2_modified), dimensions={0}
  update = f32[100,16] custom-call(input, concat_offsets, concat_updates), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
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
  auto slice = root->operand(0);
  auto update = root->operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, update));
  EXPECT_EQ(plan1, plan2);
}

TEST_F(SlicePlanTest, ShareMultipleSliceAndUpdatePlan3) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets1 = s32[24,1] parameter(1)
  offsets2 = s32[12,1] parameter(2)
  slice1 = f32[24,16] custom-call(input, offsets1), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  slice2 = f32[12,16] custom-call(input, offsets2), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice1_modified = f32[24,16] add(slice1, big_one1)
  update = f32[100,16] custom-call(input, offsets1, slice1_modified), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
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

TEST_F(SlicePlanTest, DontSharePlans) {
  const string& hlo_string = R"(
HloModule main

ENTRY main {
  input = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  slice1 = f32[24,16] custom-call(input, offsets), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}"
  one = f32[] constant(1)
  big_one1 = f32[24,16] broadcast(one), dimensions={}
  slice_modified = f32[24,16] add(slice1, big_one1)
  update1 = f32[100,16] custom-call(input, offsets, slice_modified), custom_call_target="MultiUpdate", backend_config="{\"indices_are_sorted\":false}\n"
  lr = f32[] constant(-0.1)
  update2 = f32[100,16] custom-call(input, offsets, slice_modified, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n"
  ROOT t = (f32[24,16], f32[100,16], f32[100,16]) tuple(slice1, update1, update2)
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
  // All plans are different because multiple op types are used.
  TF_ASSERT_OK_AND_ASSIGN(auto plan1, GetSlicePlan(*resources, slice1));
  TF_ASSERT_OK_AND_ASSIGN(auto plan2, GetSlicePlan(*resources, slice2));
  TF_ASSERT_OK_AND_ASSIGN(auto plan3, GetSlicePlan(*resources, update));
  EXPECT_NE(plan1, plan2);
  EXPECT_NE(plan1, plan3);
  EXPECT_NE(plan2, plan3);
}

TEST_F(SlicePlanTest, JointPlanGradientAccumulation) {
  const std::string hlo_string = R"(
HloModule top

stage_0_fwd {
  input = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  slice1 = f32[24,16] custom-call(input, offsets), custom_call_target="MultiSlice", backend_config="{\"indices_are_sorted\":false}", sharding={maximal device=0}
  ROOT stage_0_fwd_tuple = (f32[24,16]) tuple(slice1)
}

stage_0_bwd {
  gradient_accumulation_buffer = f32[100,16] parameter(0)
  offsets = s32[24,1] parameter(1)
  updates = f32[24,16] parameter(2)
  lr = f32[] constant(-0.1)
  gradient_accumulation_buffer_updated = f32[100,16] custom-call(gradient_accumulation_buffer, offsets, updates, lr), custom_call_target="MultiUpdateAdd", backend_config="{\"indices_are_sorted\":false}\n", sharding={maximal device=0}
  ROOT stage_0_bwd_tuple = (f32[100,16]) tuple(gradient_accumulation_buffer_updated)
}

resource_update {
  resource_update_p0 = f32[100,16] parameter(0)
  ga_buffer = f32[100,16] parameter(1)
  updated_param = f32[100,16] add(resource_update_p0, ga_buffer)
  ROOT t = (f32[100,16]) tuple(updated_param)
}

pipeline {
  pipeline_p0 = f32[100,16] parameter(0), sharding={maximal device=0}
  pipeline_p1 = s32[24, 1] parameter(1), sharding={maximal device=0}
  pipeline_stage_0 = (f32[24,16]) call(pipeline_p0, pipeline_p1), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0.0 = f32[24,16] get-tuple-element(pipeline_stage_0), index=0

  pipeline_accumulator = f32[100,16] custom-call(pipeline_p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  pipeline_stage_0_bwd = (f32[100,16]) call(pipeline_accumulator, pipeline_p1, pipeline_stage_0.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_stage_0_bwd.0 = f32[100,16] get-tuple-element(pipeline_stage_0_bwd), index=0

  pipeline_accumulator_sink = f32[100,16] custom-call(pipeline_stage_0_bwd.0), custom_call_target="GradientAccumulatorSink", backend_config="{\"num_mini_batches\":1}\n"

  call_ru = (f32[100,16]) call(pipeline_p0, pipeline_accumulator_sink), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  pipeline_p0_updated = f32[100,16] get-tuple-element(call_ru), index=0
  ROOT pipeline_tuple = (f32[100,16], s32[24, 1]) tuple(pipeline_p0_updated, pipeline_p1)
}

ENTRY e {
  e.weights0 = f32[100,16] parameter(0), parameter_replication={false}
  e.weights1 = s32[24, 1] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[100,16], s32[24, 1]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":0}}}"
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  TF_EXPECT_OK(
      EmbeddingPlansPreplanning(*resources).Run(module.get()).status());

  const HloInstruction* multi_slice =
      FindInstruction(module.get(), "multi-slice");
  const HloInstruction* multi_update_add =
      FindInstruction(module.get(), "multi-update-add");

  // Plan is shared, even though gradient accumulation is used.
  TF_ASSERT_OK_AND_ASSIGN(auto multi_slice_plan,
                          GetSlicePlan(*resources, multi_slice));
  TF_ASSERT_OK_AND_ASSIGN(auto multi_update_add_plan,
                          GetSlicePlan(*resources, multi_update_add));
  EXPECT_EQ(multi_slice_plan, multi_update_add_plan);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
