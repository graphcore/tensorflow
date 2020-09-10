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

#include <poplar/Device.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
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

namespace xla {
namespace poplarplugin {
namespace {

class DeferredVisitorTest : public HloTestBase {};

std::unique_ptr<CompilerResources> GetMockResources(HloModule* module,
                                                    bool merge_infeeds) {
  auto resources = CompilerResources::CreateTestDefault(module);
  resources->merge_infeed_io_copies = merge_infeeds;
  resources->streams_indices.InitializeIndexTensors(*resources, {});
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
  pipeline.AddPass<ModuleFlatten>(resources.annotations);
  pipeline.AddPass<CopyInserter>();
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

TEST_F(DeferredVisitorTest, TestDeferredAllocation) {
  const string& hlo_string = R"(

HloModule module
_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

ENTRY cluster (arg0.1: (f32[1,4,4,2], f32[2], f32[1,1,2,2])) -> f32[1,4,4,2] {
  arg = (f32[1,4,4,2], f32[2], f32[1,1,2,2]) parameter(0)
  gte0 = f32[1,4,4,2] get-tuple-element(arg), index=0
  gte2 = f32[1,1,2,2] get-tuple-element(arg), index=2
  convolution.5 = f32[1,4,4,2] convolution( gte0, gte2), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1 = f32[2] get-tuple-element(arg), index=1
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto arg = gte1->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[0];
  poplar::Tensor arg_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), arg)[1];
  EXPECT_EQ(root_tensor, gte1_tensor);
  EXPECT_EQ(gte1_tensor, arg_tensor);
}

TEST_F(DeferredVisitorTest, TestDeferredAllocationNestedTuple) {
  const string& hlo_string = R"(

HloModule module
_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

ENTRY cluster (arg0.1: ((f32[1,4,4,2], f32[2], f32[1,1,2,2]))) -> f32[1,4,4,2] {
  arg = ((f32[1,4,4,2], f32[2], f32[1,1,2,2])) parameter(0)
  gte = (f32[1,4,4,2], f32[2], f32[1,1,2,2]) get-tuple-element(arg), index=0
  gte0 = f32[1,4,4,2] get-tuple-element(gte), index=0
  gte2 = f32[1,1,2,2] get-tuple-element(gte), index=2
  convolution.5 = f32[1,4,4,2] convolution(gte0, gte2), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1 = f32[2] get-tuple-element(gte), index=1
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto gte = gte1->operand(0);
  auto arg = gte->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[0];
  poplar::Tensor gte_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte)[1];
  poplar::Tensor arg_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), arg)[1];
  EXPECT_EQ(root_tensor, gte1_tensor);
  EXPECT_EQ(gte1_tensor, gte_tensor);
  EXPECT_EQ(gte_tensor, arg_tensor);
}

TEST_F(DeferredVisitorTest, TestDeferredAllocationDoubleNestedTuple) {
  const string& hlo_string = R"(

HloModule module
_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

ENTRY cluster (arg0.1: ((f32[1,4,4,2], (f32[2], f32[1,1,2,2])))) -> f32[1,4,4,2] {
  arg = ((f32[1,4,4,2], (f32[2], f32[1,1,2,2]))) parameter(0)
  gte = (f32[1,4,4,2], (f32[2], f32[1,1,2,2])) get-tuple-element(arg), index=0
  gte0 = f32[1,4,4,2] get-tuple-element(gte), index=0
  gte1 = (f32[2], f32[1,1,2,2]) get-tuple-element(gte), index=1
  gte1.1 = f32[1,1,2,2] get-tuple-element(gte1), index=1
  convolution.5 = f32[1,4,4,2] convolution(gte0, gte1.1), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1.0 = f32[2] get-tuple-element(gte1), index=0
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1.0), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1.0 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1_0 = root->operand(1);
  auto gte1 = gte1_0->operand(0);
  auto gte = gte1->operand(0);
  auto arg = gte->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_0_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1_0)[0];
  poplar::Tensor gte1_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[0];
  poplar::Tensor gte_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte)[1];
  poplar::Tensor arg_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), arg)[1];
  EXPECT_EQ(root_tensor, gte1_0_tensor);
  EXPECT_EQ(gte1_0_tensor, gte1_tensor);
  EXPECT_EQ(gte1_tensor, gte_tensor);
  EXPECT_EQ(gte_tensor, arg_tensor);
}

TEST_F(DeferredVisitorTest,
       TestDeferredAllocationMultipleDeferredAllocationsNestedTuple) {
  const string& hlo_string = R"(

HloModule module
_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

_pop_op_conv_biasadd.1 (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

ENTRY cluster (arg0.1: ((((f32[1,4,4,2], f32[1,4,4,2]), (f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]))))) -> (f32[1,4,4,2], f32[1,4,4,2]) {
  arg = (((f32[1,4,4,2], f32[1,4,4,2]), (f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]))) parameter(0)
  gte = ((f32[1,4,4,2], f32[1,4,4,2]), (f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2])) get-tuple-element(arg), index=0
  gte0 = (f32[1,4,4,2], f32[1,4,4,2]) get-tuple-element(gte), index=0
  gte0.0 = f32[1,4,4,2] get-tuple-element(gte0), index=0
  gte1 = (f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) get-tuple-element(gte), index=1
  gte1.1 = f32[1,1,2,2] get-tuple-element(gte1), index=1
  convolution.0 = f32[1,4,4,2] convolution(gte0.0, gte1.1), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1.0 = f32[2] get-tuple-element((f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) gte1), index=0
  fusion.0 = f32[1,4,4,2] fusion(convolution.0, gte1.0), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"

  gte0.1 = f32[1,4,4,2] get-tuple-element(gte0), index=1
  gte1.3 = f32[1,1,2,2] get-tuple-element((f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) gte1), index=3
  convolution.1 = f32[1,4,4,2] convolution(gte0.1, gte1.3), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1.2 = f32[2] get-tuple-element((f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) gte1), index=2
  fusion.1 = f32[1,4,4,2] fusion(convolution.1, gte1.2), kind=kCustom, calls=_pop_op_conv_biasadd.1, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(fusion.0, fusion.1)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto root_tuple = entry_computation->root_instruction();

  // Verify that gte1.0 has a tensor and all the deferred allocations have that
  // tensor too.
  auto fusion_0 = root_tuple->operand(0);
  auto gte1_0 = fusion_0->operand(1);
  auto gte1 = gte1_0->operand(0);
  auto gte = gte1->operand(0);
  auto arg = gte->operand(0);
  poplar::Tensor fusion_0_input_one_tensor =
      FindInstructionInput(tensor_map, *resources.get(), fusion_0, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_0_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1_0)[0];
  poplar::Tensor gte1_tensor_zero =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[0];
  poplar::Tensor gte_tensor_two =
      FindInstructionOutputs(tensor_map, *resources.get(), gte)[2];
  poplar::Tensor arg_tensor_two =
      FindInstructionOutputs(tensor_map, *resources.get(), arg)[2];
  EXPECT_EQ(fusion_0_input_one_tensor, gte1_0_tensor);
  EXPECT_EQ(gte1_0_tensor, gte1_tensor_zero);
  EXPECT_EQ(gte1_tensor_zero, gte_tensor_two);
  EXPECT_EQ(gte_tensor_two, arg_tensor_two);

  // Verify that gte1.2 has a tensor and all the deferred allocations have that
  // tensor too.
  auto fusion_1 = root_tuple->operand(1);
  auto gte1_2 = fusion_1->operand(1);
  EXPECT_EQ(gte1, gte1_2->operand(0));

  poplar::Tensor fusion_1_input_one_tensor =
      FindInstructionInput(tensor_map, *resources.get(), fusion_1, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_2_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1_2)[0];
  poplar::Tensor gte1_tensor_two =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[2];
  poplar::Tensor gte_tensor_four =
      FindInstructionOutputs(tensor_map, *resources.get(), gte)[4];
  poplar::Tensor arg_tensor_four =
      FindInstructionOutputs(tensor_map, *resources.get(), arg)[4];
  EXPECT_EQ(fusion_1_input_one_tensor, gte1_2_tensor);
  EXPECT_EQ(gte1_2_tensor, gte1_tensor_two);
  EXPECT_EQ(gte1_tensor_two, gte_tensor_four);
  EXPECT_EQ(gte_tensor_four, arg_tensor_four);
}

TEST_F(DeferredVisitorTest, TestDeferredAllocationNestedTupleInfeed) {
  const string& hlo_string = R"(

HloModule module
_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  arg_0 = f32[1,4,4,2] parameter(0)
  arg_1 = f32[2] parameter(1)
  broadcast.6.clone = f32[1,4,4,2] broadcast(arg_1), dimensions={3}
  ROOT add.7.clone = f32[1,4,4,2] add(arg_0, broadcast.6.clone)
}

ENTRY cluster (arg: f32[1,1,2,2]) -> f32[1,4,4,2] {
  arg = f32[1,1,2,2] parameter(0)
  after-all = token[] after-all()
  infeed = ((f32[1,4,4,2], f32[2]), token[]) infeed(token[] after-all), infeed_config="\010\001\022\005feed5\"\002\001\001"
  gte = (f32[1,4,4,2], f32[2]) get-tuple-element(((f32[1,4,4,2], f32[2]), token[]) infeed), index=0
  gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[2]) gte), index=0
  convolution.5 = f32[1,4,4,2] convolution(gte0, arg), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2]) gte), index=1
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceOperands\":[\"0\"]}}"
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto gte = gte1->operand(0);
  auto infeed = gte->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.GetMutableSequence(), false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte1)[0];
  poplar::Tensor gte_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), gte)[1];
  poplar::Tensor infeed_tensor =
      FindInstructionOutputs(tensor_map, *resources.get(), infeed)[1];
  EXPECT_EQ(root_tensor, gte1_tensor);
  EXPECT_EQ(gte1_tensor, gte_tensor);
  EXPECT_EQ(gte_tensor, infeed_tensor);
}

TEST_F(DeferredVisitorTest, TestDeferredAllocationLoopsInputWithLayout) {
  const string& hlo_string = R"(
HloModule module

while_Sum-reduction.13 (x.14: f32[], y.15: f32[]) -> f32[] {
  x.14 = f32[] parameter(0)
  y.15 = f32[] parameter(1)
  ROOT add.16 = f32[] add(x.14, y.15)
}

_functionalize_body_1__.17 (arg_tuple.18: (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2])) -> (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) {
  arg_tuple.18 = (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) parameter(0)
  get-tuple-element.21 = f32[] get-tuple-element(arg_tuple.18), index=2
  get-tuple-element.19 = s32[] get-tuple-element(arg_tuple.18), index=0
  constant.26 = s32[] constant(1)
  add.27 = s32[] add(get-tuple-element.19, constant.26)
  get-tuple-element.20 = s32[] get-tuple-element(arg_tuple.18), index=1
  constant.30 = s32[] constant(2)
  add.31 = s32[] add(get-tuple-element.20, constant.30)
  get-tuple-element.24 = f32[1,4,4,2] get-tuple-element(arg_tuple.18), index=5
  get-tuple-element.25 = f32[1,1,2,2] get-tuple-element(arg_tuple.18), index=6
  convolution.48 = f32[1,4,4,2] convolution(get-tuple-element.24, get-tuple-element.25), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte = (f32[2], f32[2]) get-tuple-element(arg_tuple.18), index=4
  get-tuple-element.23 = f32[2] get-tuple-element(gte), index=0
  constant.33 = f32[] constant(0.1)
  broadcast.34 = f32[2] broadcast(constant.33), dimensions={}
  constant.32 = f32[2] constant({16, 16})
  multiply.35 = f32[2] multiply(broadcast.34, constant.32)
  subtract.36 = f32[2] subtract(get-tuple-element.23, multiply.35)
  broadcast.49 = f32[1,4,4,2] broadcast(subtract.36), dimensions={3}
  add.50 = f32[1,4,4,2] add(convolution.48, broadcast.49)
  convert.51 = f32[1,4,4,2] convert(add.50)
  constant.52 = f32[] constant(0)
  convert.53 = f32[] convert(constant.52)
  reduce.54 = f32[] reduce(convert.51, convert.53), dimensions={0,1,2,3}, to_apply=while_Sum-reduction.13
  convert.55 = f32[] convert(reduce.54)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.18), index=3
  tuple.56 = (f32[2], f32[2]) tuple(f32[2] subtract.36, f32[2] subtract.36)
  constant.40 = f32[] constant(0.1)
  broadcast.41 = f32[1,4,4,2] broadcast(constant.40), dimensions={}
  constant.28 = f32[] constant(1)
  broadcast.29 = f32[1,4,4,2] broadcast(constant.28), dimensions={}
  reverse.38 = f32[1,1,2,2] reverse(get-tuple-element.25), dimensions={0,1}
  convolution.39 = f32[1,4,4,2] convolution(broadcast.29, reverse.38), window={size=1x1}, dim_labels=b01f_01oi->b01f
  multiply.42 = f32[1,4,4,2] multiply(broadcast.41, convolution.39)
  subtract.43 = f32[1,4,4,2] subtract(get-tuple-element.24, multiply.42)
  tuple.58 = (f32[1,4,4,2]) tuple(subtract.43)
  get-tuple-element.59 = f32[1,4,4,2] get-tuple-element(tuple.58), index=0
  constant.44 = f32[] constant(0.1)
  broadcast.45 = f32[1,1,2,2] broadcast(constant.44), dimensions={}
  convolution.37 = f32[1,1,2,2] convolution(get-tuple-element.24, broadcast.29), window={size=4x4}, dim_labels=f01b_i01o->01bf
  multiply.46 = f32[1,1,2,2] multiply(broadcast.45, convolution.37)
  subtract.47 = f32[1,1,2,2] subtract(get-tuple-element.25, multiply.46)
  tuple.60 = (f32[1,1,2,2]) tuple(subtract.47)
  get-tuple-element.61 = f32[1,1,2,2] get-tuple-element(tuple.60), index=0
  ROOT tuple.62 = (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) tuple(add.27, add.31, convert.55, get-tuple-element.22, tuple.56, get-tuple-element.59, get-tuple-element.61)
}

cond_wrapper.77 (inputs.78: (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2])) -> pred[] {
  arg_tuple.64 = (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) parameter(0)
  get-tuple-element.67 = f32[] get-tuple-element(arg_tuple.64), index=2
  get-tuple-element.70 = f32[1,4,4,2] get-tuple-element(arg_tuple.64), index=5
  get-tuple-element.71 = f32[1,1,2,2] get-tuple-element(arg_tuple.64), index=6
  get-tuple-element.65 = s32[] get-tuple-element(arg_tuple.64), index=0
  get-tuple-element.68 = s32[] get-tuple-element(arg_tuple.64), index=3
  less-than.74 = pred[] compare(get-tuple-element.65, get-tuple-element.68), direction=LT
  get-tuple-element.66 = s32[] get-tuple-element(arg_tuple.64), index=1
  constant.72 = s32[] constant(2)
  less-than.73 = pred[] compare(get-tuple-element.66, constant.72), direction=LT
  ROOT and.75 = pred[] and(pred[] less-than.74, pred[] less-than.73)
}

ENTRY cluster_4790582643659166751_f15n_0__.98 (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[1,1,2,2]) -> (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) {
  constant1 = s32[] constant(0)
  constant2 = f32[] constant(10)
  constant3 = s32[] constant(10)
  arg1.2 = f32[2] parameter(1)
  arg0.1 = f32[1,4,4,2] parameter(0)
  arg2.3 = f32[1,1,2,2] parameter(2)
  t = (f32[2], f32[2]) tuple(arg1.2, arg1.2)
  tuple.12 = (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) tuple(constant1, constant1, constant2, constant3, t, arg0.1, arg2.3)
  ROOT while.81 = (s32[], s32[], f32[], s32[], (f32[2], f32[2]), f32[1,4,4,2], f32[1,1,2,2]) while(tuple.12), condition=cond_wrapper.77, body=_functionalize_body_1__.17
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Check that the two last inputs to the loop are the same tensor - these
  // parameters have allocation targets in the entry computation and inside the
  // loop - but because they already have a layout at the callsite we just use
  // that rather than creating a new layout and copying.
  auto entry_tensor_map = resources->tensor_maps.GetTensorMapForComputation(
      entry_computation->name());
  auto entry_root_instruction = entry_computation->root_instruction();
  auto entry_loop_input_tensors = FindInstructionOutputs(
      entry_tensor_map, *resources.get(), entry_root_instruction->operand(0));
  auto entry_loop_tensors = FindInstructionOutputs(
      entry_tensor_map, *resources.get(), entry_root_instruction);

  auto loop_body = entry_root_instruction->while_body();
  auto loop_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation(loop_body->name());
  EXPECT_EQ(loop_body->num_parameters(), 1);
  auto loop_tuple = loop_body->parameter_instruction(0);
  auto loop_tuple_tensors =
      FindInstructionOutputs(loop_tensor_map, *resources.get(), loop_tuple);

  // Check the tensors.
  auto num_inputs = CountShapes(loop_tuple->shape());
  EXPECT_EQ(entry_loop_tensors.size(), num_inputs);

  EXPECT_EQ(entry_loop_input_tensors.size(), loop_tuple_tensors.size());
  EXPECT_EQ(entry_loop_input_tensors[0], loop_tuple_tensors[0]);
  // Duplicate operand gets reallocated.
  EXPECT_NE(entry_loop_input_tensors[1], loop_tuple_tensors[1]);
  EXPECT_EQ(entry_loop_input_tensors[2], loop_tuple_tensors[2]);
  EXPECT_EQ(entry_loop_input_tensors[3], loop_tuple_tensors[3]);
  EXPECT_EQ(entry_loop_input_tensors[4], loop_tuple_tensors[4]);
  EXPECT_EQ(entry_loop_input_tensors[5], loop_tuple_tensors[5]);
  EXPECT_EQ(entry_loop_input_tensors[6], loop_tuple_tensors[6]);

  EXPECT_EQ(entry_loop_tensors.size(), loop_tuple_tensors.size());
  for (int64 i = 0; i != entry_loop_tensors.size(); i++) {
    if (i == 2) {
      // 2nd input is reallocated inside of the loop.
      EXPECT_NE(entry_loop_tensors[i], loop_tuple_tensors[i]);
    } else {
      EXPECT_EQ(entry_loop_tensors[i], loop_tuple_tensors[i]);
    }
  }
}

TEST_F(DeferredVisitorTest, TestDeferredAllocationInsideLoops) {
  const string& hlo_string = R"(
HloModule module

while_Sum-reduction.13 (x.14: f32[], y.15: f32[]) -> f32[] {
  x.14 = f32[] parameter(0)
  y.15 = f32[] parameter(1)
  ROOT add.16 = f32[] add(x.14, y.15)
}

_functionalize_body_1__.17 (arg_tuple.18: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) {
  arg_tuple.18 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) parameter(0)
  get-tuple-element.21 = f32[] get-tuple-element(arg_tuple.18), index=2
  get-tuple-element.19 = s32[] get-tuple-element(arg_tuple.18), index=0
  constant.26 = s32[] constant(1)
  add.27 = s32[] add(get-tuple-element.19, constant.26)
  get-tuple-element.20 = s32[] get-tuple-element(arg_tuple.18), index=1
  constant.30 = s32[] constant(1)
  add.31 = s32[] add(get-tuple-element.20, constant.30)
  get-tuple-element.24 = f32[1,4,4,2] get-tuple-element(arg_tuple.18), index=5
  get-tuple-element.25 = f32[1,1,2,2] get-tuple-element(arg_tuple.18), index=6
  convolution.48 = f32[1,4,4,2] convolution(get-tuple-element.24, get-tuple-element.25), window={size=1x1}, dim_labels=b01f_01io->b01f
  get-tuple-element.23 = f32[2] get-tuple-element(arg_tuple.18), index=4
  constant.33 = f32[] constant(0.1)
  broadcast.34 = f32[2] broadcast(constant.33), dimensions={}
  constant.32 = f32[2] constant({16, 16})
  multiply.35 = f32[2] multiply(broadcast.34, constant.32)
  subtract.36 = f32[2] subtract(get-tuple-element.23, multiply.35)
  broadcast.49 = f32[1,4,4,2] broadcast(subtract.36), dimensions={3}
  add.50 = f32[1,4,4,2] add(convolution.48, broadcast.49)
  convert.51 = f32[1,4,4,2] convert(add.50)
  constant.52 = f32[] constant(0)
  convert.53 = f32[] convert(constant.52)
  reduce.54 = f32[] reduce(convert.51, convert.53), dimensions={0,1,2,3}, to_apply=while_Sum-reduction.13
  convert.55 = f32[] convert(reduce.54)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.18), index=3
  tuple.56 = (f32[2]) tuple(f32[2] subtract.36)
  get-tuple-element.57 = f32[2] get-tuple-element((f32[2]) tuple.56), index=0
  constant.40 = f32[] constant(0.1)
  broadcast.41 = f32[1,4,4,2] broadcast(constant.40), dimensions={}
  constant.28 = f32[] constant(1)
  broadcast.29 = f32[1,4,4,2] broadcast(constant.28), dimensions={}
  reverse.38 = f32[1,1,2,2] reverse(get-tuple-element.25), dimensions={0,1}
  convolution.39 = f32[1,4,4,2] convolution(broadcast.29, reverse.38), window={size=1x1}, dim_labels=b01f_01oi->b01f
  multiply.42 = f32[1,4,4,2] multiply(broadcast.41, convolution.39)
  subtract.43 = f32[1,4,4,2] subtract(get-tuple-element.24, multiply.42)
  tuple.58 = (f32[1,4,4,2]) tuple(subtract.43)
  get-tuple-element.59 = f32[1,4,4,2] get-tuple-element(tuple.58), index=0
  constant.44 = f32[] constant(0.1)
  broadcast.45 = f32[1,1,2,2] broadcast(constant.44), dimensions={}
  convolution.37 = f32[1,1,2,2] convolution(get-tuple-element.24, broadcast.29), window={size=4x4}, dim_labels=f01b_i01o->01bf
  multiply.46 = f32[1,1,2,2] multiply(broadcast.45, convolution.37)
  subtract.47 = f32[1,1,2,2] subtract(get-tuple-element.25, multiply.46)
  tuple.60 = (f32[1,1,2,2]) tuple(subtract.47)
  get-tuple-element.61 = f32[1,1,2,2] get-tuple-element(tuple.60), index=0
  ROOT tuple.62 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) tuple(add.27, add.31, convert.55, get-tuple-element.22, get-tuple-element.57, get-tuple-element.59, get-tuple-element.61)
}

_functionalize_cond_1__.63 (arg_tuple.64: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> (pred[]) {
  arg_tuple.64 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) parameter(0)
  get-tuple-element.67 = f32[] get-tuple-element(arg_tuple.64), index=2
  get-tuple-element.69 = f32[2] get-tuple-element(arg_tuple.64), index=4
  get-tuple-element.70 = f32[1,4,4,2] get-tuple-element(arg_tuple.64), index=5
  get-tuple-element.71 = f32[1,1,2,2] get-tuple-element(arg_tuple.64), index=6
  get-tuple-element.65 = s32[] get-tuple-element(arg_tuple.64), index=0
  get-tuple-element.68 = s32[] get-tuple-element(arg_tuple.64), index=3
  less-than.74 = pred[] compare(get-tuple-element.65, get-tuple-element.68), direction=LT
  get-tuple-element.66 = s32[] get-tuple-element(arg_tuple.64), index=1
  constant.72 = s32[] constant(2)
  less-than.73 = pred[] compare(get-tuple-element.66, constant.72), direction=LT
  and.75 = pred[] and(pred[] less-than.74, pred[] less-than.73)
  ROOT tuple.76 = (pred[]) tuple(pred[] and.75)
}

cond_wrapper.77 (inputs.78: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> pred[] {
  inputs.78 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) parameter(0)
  call.79 = (pred[]) call(inputs.78), to_apply=_functionalize_cond_1__.63
  ROOT get-tuple-element.80 = pred[] get-tuple-element((pred[]) call.79), index=0
}

ENTRY cluster_4790582643659166751_f15n_0__.98 (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[1,1,2,2]) -> (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) {
  constant.4 = s32[] constant(0)
  constant.5 = s32[] constant(0)
  constant.6 = f32[] constant(0)
  constant.7 = s32[] constant(10)
  constant.8 = s32[] constant(0)
  constant.9 = s32[] constant(0)
  constant.10 = f32[] constant(0)
  constant.11 = s32[] constant(10)
  arg1.2 = f32[2] parameter(1)
  arg0.1 = f32[1,4,4,2] parameter(0)
  arg2.3 = f32[1,1,2,2] parameter(2)
  tuple.12 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) tuple(constant.8, constant.9, constant.10, constant.11, arg1.2, arg0.1, arg2.3)
  ROOT while.81 = (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) while(tuple.12), condition=cond_wrapper.77, body=_functionalize_body_1__.17
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto layout_d4 = LayoutUtil::MakeLayout({3, 2, 1, 0});
  Array4D<float> input_arr({
      // clang-format off
    {  // i0=0
        {  // i1=0
            { 1,  2},  // i2=0
            { 3,  4},  // i2=1
            { 5,  6},  // i2=2
            { 7,  8},  // i2=3
        },
        {  // i1=1
            { 9, 10},  // i2=0
            {11, 12},  // i2=1
            {13, 14},  // i2=2
            {15, 16},  // i2=3
        },
        {  // i1=2
            {17, 18},  // i2=0
            {19, 20},  // i2=1
            {21, 22},  // i2=2
            {23, 24},  // i2=3
        },
        {  // i1=3
            {25, 26},  // i2=0
            {27, 28},  // i2=1
            {29, 30},  // i2=2
            {31, 32},  // i2=3
        },
    },
      // clang-format on
  });
  auto input_literal =
      LiteralUtil::CreateR4FromArray4DWithLayout<float>(input_arr, layout_d4);
  auto biases = LiteralUtil::CreateR1<float>({-100, 100});

  Array4D<float> weights_arr({
      // clang-format off
    {  // i0=0
        {  // i1=0
            {1, 2},  // i2=0
            {3, 4},  // i2=1
        },
    },
      // clang-format on
  });
  auto weights_literal =
      LiteralUtil::CreateR4FromArray4DWithLayout<float>(weights_arr, layout_d4);
  auto result = ExecuteAndTransfer(std::move(module),
                                   {&input_literal, &biases, &weights_literal});
  auto result_tuple = result.DecomposeTuple();
  // Expect correct value for the biases.
  EXPECT_EQ(result_tuple[4], LiteralUtil::CreateR1<float>({-103.2, 96.8}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
