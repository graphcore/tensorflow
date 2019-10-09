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
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

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
  auto resources = absl::make_unique<CompilerResources>(
      poplar::OptionFlags(), poplar::OptionFlags(), poplar::OptionFlags(),
      false, false, merge_infeeds, 1, 0, 0, 1, 64, module,
      IpuOptions::FloatingPointBehaviour(), false, "", false);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), 0, poplar::replication_factor(1));
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

popops::SlicePlan* check_operands_slice_plans_match(
    const HloInstruction* inst, const CompilerResources& resources) {
  EXPECT_GT(inst->operand_count(), 0);
  popops::SlicePlan* inst_plan{nullptr};

  for (auto i = 0; i < inst->operand_count(); i++) {
    auto plan = resources.slice_plan_mappings.find(inst->operand(i));
    EXPECT_NE(plan, resources.slice_plan_mappings.end());
    if (i > 0) {
      EXPECT_EQ(plan->second, inst_plan);
    } else {
      inst_plan = plan->second;
    }
  }
  return inst_plan;
}

TEST_F(SlicePlanTest, SeparateLookups) {
  const string& hlo_string = R"(
HloModule main

ENTRY main (arg0.1: s32[24,1], arg1.2: s32[24,1], arg2.3: f32[], arg3.4: f32[100,16], arg4.5: f32[100,16], arg5.6: f32[100,16], arg6.7: f32[100,16]) -> (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) {
  input = f32[100,16] parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  offsets = s32[24,1] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice = f32[24,16] custom-call(f32[100,16] input, s32[24,1] offsets), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup"}, backend_config="null\n"
  input.2 = f32[100,16] parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  offsets.2 = s32[24,1] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice.2 = f32[24,16] custom-call(f32[100,16] input.2, s32[24,1] offsets.2), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup_1"}, backend_config="null\n"
  gradients = f32[100,16] parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  lr = f32[] parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  constant.15 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  constant.16 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  update = f32[100,16] custom-call(f32[100,16] %constant.15, s32[24,1] offsets, f32[24,16] %constant.16), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul = f32[100,16] multiply(f32[100,16] broadcast_lr, f32[100,16] update), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  sub = f32[100,16] subtract(f32[100,16] gradients, f32[100,16] mul), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  gradients.2 = f32[100,16] parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr.2 = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  constant.18 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  constant.19 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  update.2 = f32[100,16]{1,0} custom-call(f32[100,16] constant.18, s32[24,1] offsets.2, f32[24,16] %constant.19), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul.2 = f32[100,16] multiply(f32[100,16] broadcast_lr.2, f32[100,16] update.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  sub.2 = f32[100,16] subtract(f32[100,16] gradients.2, f32[100,16] mul.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  ROOT tuple.53 = (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) tuple(f32[24,16] slice, f32[24,16] slice.2, f32[100,16] update, f32[100,16] update.2), metadata={op_name="XLA_Retvals"}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EmbeddingPlansPreplanning embeddings_preplanning;
  TF_EXPECT_OK(embeddings_preplanning.Plan(module.get(), *resources));
  EntryVisitor visitor(*resources);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice = root->operand(0);
  auto slice_2 = root->operand(1);
  auto update = root->operand(2);
  auto update_2 = root->operand(3);

  auto plan_slice = check_operands_slice_plans_match(slice, *resources);
  auto plan_update = check_operands_slice_plans_match(update, *resources);
  auto plan_slice_2 = check_operands_slice_plans_match(slice_2, *resources);
  auto plan_update_2 = check_operands_slice_plans_match(update_2, *resources);

  EXPECT_EQ(plan_slice, plan_update);
  EXPECT_EQ(plan_slice_2, plan_update_2);
  EXPECT_NE(plan_slice, plan_slice_2);
}

TEST_F(SlicePlanTest, SharedOffsets) {
  const string& hlo_string = R"(
HloModule main

ENTRY main (arg0.1: s32[24,1], arg2.3: f32[], arg3.4: f32[100,16], arg4.5: f32[100,16], arg5.6: f32[100,16], arg6.7: f32[100,16]) -> (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) {
  input = f32[100,16] parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  offsets = s32[24,1] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice = f32[24,16] custom-call(f32[100,16] input, s32[24,1] offsets), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup"}, backend_config="null\n"
  input.2 = f32[100,16] parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice.2 = f32[24,16] custom-call(f32[100,16] input.2, s32[24,1] offsets), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup_1"}, backend_config="null\n"
  gradients = f32[100,16] parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  lr = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  constant.15 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  constant.16 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  update = f32[100,16] custom-call(f32[100,16] %constant.15, s32[24,1] offsets, f32[24,16] %constant.16), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul = f32[100,16] multiply(f32[100,16] broadcast_lr, f32[100,16] update), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  sub = f32[100,16] subtract(f32[100,16] gradients, f32[100,16] mul), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  gradients.2 = f32[100,16] parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr.2 = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  constant.18 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  constant.19 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  update.2 = f32[100,16] custom-call(f32[100,16] constant.18, s32[24,1] offsets, f32[24,16] %constant.19), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul.2 = f32[100,16] multiply(f32[100,16] broadcast_lr.2, f32[100,16] update.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  sub.2 = f32[100,16] subtract(f32[100,16] gradients.2, f32[100,16] mul.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  ROOT tuple.53 = (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) tuple(f32[24,16] slice, f32[24,16] slice.2, f32[100,16] update, f32[100,16] update.2), metadata={op_name="XLA_Retvals"}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EmbeddingPlansPreplanning embeddings_preplanning;
  TF_EXPECT_OK(embeddings_preplanning.Plan(module.get(), *resources));
  EntryVisitor visitor(*resources);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice = root->operand(0);
  auto slice_2 = root->operand(1);
  auto update = root->operand(2);
  auto update_2 = root->operand(3);

  auto plan_slice = check_operands_slice_plans_match(slice, *resources);
  auto plan_update = check_operands_slice_plans_match(update, *resources);
  auto plan_slice_2 = check_operands_slice_plans_match(slice_2, *resources);
  auto plan_update_2 = check_operands_slice_plans_match(update_2, *resources);

  EXPECT_EQ(plan_slice, plan_update);
  EXPECT_EQ(plan_slice_2, plan_update_2);
  EXPECT_EQ(plan_slice, plan_slice_2);
}

TEST_F(SlicePlanTest, SharedOffsetsReshape) {
  const string& hlo_string = R"(
HloModule main

ENTRY main (arg0.1: s32[24], arg2.3: f32[], arg3.4: f32[100,16], arg4.5: f32[100,16], arg5.6: f32[100,16], arg6.7: f32[100,16]) -> (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) {
  input = f32[100,16] parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  offsets = s32[24] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice = f32[24,16] custom-call(f32[100,16] input, s32[24] offsets), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup"}, backend_config="null\n"
  input.2 = f32[100,16] parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  slice.2 = f32[24,16] custom-call(f32[100,16] input.2, s32[24] offsets), custom_call_target="Popops::MultiSlice", metadata={op_type="IpuMultiSlice" op_name="vs/embedding_lookup_1"}, backend_config="null\n"
  gradients = f32[100,16] parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  lr = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  constant.15 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  constant.16 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}
  update = f32[100,16] custom-call(f32[100,16] %constant.15, s32[24] offsets, f32[24,16] %constant.16), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients/vs/embedding_lookup_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul = f32[100,16] multiply(f32[100,16] broadcast_lr, f32[100,16] update), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  sub = f32[100,16] subtract(f32[100,16] gradients, f32[100,16] mul), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent/update_vs/x/ResourceApplyGradientDescent"}
  gradients.2 = f32[100,16] parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  broadcast_lr.2 = f32[100,16] broadcast(f32[] lr), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  constant.18 = f32[100,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  constant.19 = f32[24,16] constant({...}), metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}
  update.2 = f32[100,16] custom-call(f32[100,16] constant.18, s32[24] offsets, f32[24,16] %constant.19), custom_call_target="Popops::MultiUpdate", metadata={op_type="IpuMultiUpdate" op_name="vs/gradients_1/vs/embedding_lookup_1_grad/IpuMultiUpdate"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1}\n"
  mul.2 = f32[100,16] multiply(f32[100,16] broadcast_lr.2, f32[100,16] update.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  sub.2 = f32[100,16] subtract(f32[100,16] gradients.2, f32[100,16] mul.2), metadata={op_type="ResourceApplyGradientDescent" op_name="vs/GradientDescent_1/update_vs/y/ResourceApplyGradientDescent"}
  ROOT tuple.53 = (f32[24,16], f32[24,16], f32[100,16], f32[100,16]) tuple(f32[24,16] slice, f32[24,16] slice.2, f32[100,16] update, f32[100,16] update.2), metadata={op_name="XLA_Retvals"}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EmbeddingPlansPreplanning embeddings_preplanning;
  TF_EXPECT_OK(embeddings_preplanning.Plan(module.get(), *resources));
  EntryVisitor visitor(*resources);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto root = entry_computation->root_instruction();
  auto slice = root->operand(0);
  auto slice_2 = root->operand(1);
  auto update = root->operand(2);
  auto update_2 = root->operand(3);

  auto plan_slice = check_operands_slice_plans_match(slice, *resources);
  auto plan_update = check_operands_slice_plans_match(update, *resources);
  auto plan_slice_2 = check_operands_slice_plans_match(slice_2, *resources);
  auto plan_update_2 = check_operands_slice_plans_match(update_2, *resources);

  EXPECT_EQ(plan_slice, plan_update);
  EXPECT_EQ(plan_slice_2, plan_update_2);
  EXPECT_EQ(plan_slice, plan_slice_2);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
