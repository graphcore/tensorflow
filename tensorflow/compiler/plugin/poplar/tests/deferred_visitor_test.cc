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
  resources->streams_indices.InitializeIndexTensors(*resources, {}, {});
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

void GetProgramNamesFlattened(const Json::Value& json,
                              std::vector<std::string>* out) {
  for (const auto& program : json["programs"]) {
    for (const auto& name : program.getMemberNames()) {
      if (name == "Sequence") {
        // Recurse.
        GetProgramNamesFlattened(program[name], out);
      } else {
        out->push_back(name);
      }
    }
  }
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
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
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
  auto seq = visitor.GetRawSequence();
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1, seq, false)
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
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
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
  auto seq = visitor.GetRawSequence();
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1, seq, false)
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
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1.0), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
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
  auto seq = visitor.GetRawSequence();
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1, seq, false)
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
  fusion.0 = f32[1,4,4,2] fusion(convolution.0, gte1.0), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"

  gte0.1 = f32[1,4,4,2] get-tuple-element(gte0), index=1
  gte1.3 = f32[1,1,2,2] get-tuple-element((f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) gte1), index=3
  convolution.1 = f32[1,4,4,2] convolution(gte0.1, gte1.3), window={size=1x1}, dim_labels=b01f_01io->b01f
  gte1.2 = f32[2] get-tuple-element((f32[2], f32[1,1,2,2], f32[2], f32[1,1,2,2]) gte1), index=2
  fusion.1 = f32[1,4,4,2] fusion(convolution.1, gte1.2), kind=kCustom, calls=_pop_op_conv_biasadd.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
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
  auto seq = visitor.GetRawSequence();
  poplar::Tensor fusion_0_input_one_tensor =
      FindInstructionInput(tensor_map, *resources.get(), fusion_0, 1, seq,
                           false)
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
      FindInstructionInput(tensor_map, *resources.get(), fusion_1, 1, seq,
                           false)
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
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
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
  auto seq = visitor.GetRawSequence();
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1, seq, false)
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

TEST_F(DeferredVisitorTest, TestGroupingOfStreamCopiesFromInfeed) {
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
  ROOT fusion = f32[1,4,4,2] fusion(convolution.5, gte1), kind=kCustom, calls=_pop_op_conv_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), /*merge_infeeds=*/true);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  const auto* entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get and flatten the resulting sequence.
  const auto seq = visitor.GetRawSequence();
  std::ostringstream oss;
  poplar::program::dumpProgram(GetMasterGraph(*resources.get()), seq, oss);
  Json::Value json;
  EXPECT_TRUE(Json::Reader().parse(oss.str().c_str(), json));
  std::vector<std::string> program_names;
  GetProgramNamesFlattened(json, &program_names);

  // Check that the stream copies are grouped at the beginning, ready to be
  // merged by Poplar.
  ASSERT_GE(program_names.size(), 3);
  EXPECT_EQ(program_names[0], "StreamCopy");
  EXPECT_EQ(program_names[1], "StreamCopy");
  EXPECT_EQ(program_names[2], "StreamCopy");
}

TEST_F(DeferredVisitorTest, TestFunctionWithDeferredInputs) {
  const string& hlo_string = R"(
HloModule module

func {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)
  p2 = f32[8,8] parameter(2)
  dot = f32[8,8] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add = f32[8,8] add(dot, p2)
  ROOT t = (f32[8,8]) tuple(add)
}

func2 {
  p0.2 = f32[8,8] parameter(0)
  p1.2 = f32[8,8] parameter(1)
  p2.2 = f32[8,8] parameter(2)
  dot.2 = f32[8,8] dot(p0.2, p1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.2 = f32[8,8] add(dot.2, p2.2)
  ROOT t.2 = (f32[8,8]) tuple(add.2)
}

func3 {
  p0.3 = f32[8,8] parameter(0)
  p1.3 = f32[8,8] parameter(1)
  p2.3 = f32[8,8] parameter(2)
  dot.3 = f32[8,8] dot(p0.3, p1.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  add.3 = f32[8,8] add(dot.3, p2.3)
  ROOT t.3 = (f32[8,8]) tuple(add.3)
}

ENTRY main {
  arg0 = f32[8,8] parameter(0)
  arg1 = f32[8,8] parameter(1)
  arg2 = f32[8,8] parameter(2)
  arg3 = f32[8,8] parameter(3)
  arg4 = f32[8,8] parameter(4)
  arg5 = f32[8,8] parameter(5)
  arg6 = f32[8,8] parameter(6)

  c1 = (f32[8,8]) call(arg0, arg1, arg2), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c1_gte = f32[8,8] get-tuple-element(c1), index=0
  c2 = (f32[8,8]) call(arg3, c1_gte, arg4), to_apply=func2, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c2_gte = f32[8,8] get-tuple-element(c2), index=0
  ROOT c3 = (f32[8,8]) call(arg5, arg6, c2_gte), to_apply=func3, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
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

  auto entry_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("main");
  auto func_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("func");

  // In this test we check that arg2 and arg4 get a deferred layout from the
  // function call and that the third call has just a copy.
  HloInstruction* arg2 = FindInstruction(module.get(), "arg2");
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg2_ts,
      FindInstructionOutputTensors(entry_tensor_map, *resources.get(), arg2));
  ASSERT_EQ(arg2_ts.size(), 1);

  HloInstruction* arg4 = FindInstruction(module.get(), "arg4");
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg4_ts,
      FindInstructionOutputTensors(entry_tensor_map, *resources.get(), arg4));
  ASSERT_EQ(arg4_ts.size(), 1);

  HloInstruction* c2_gte = FindInstruction(module.get(), "c2_gte");
  TF_ASSERT_OK_AND_ASSIGN(
      auto c2_gte_ts,
      FindInstructionOutputTensors(entry_tensor_map, *resources.get(), c2_gte));
  ASSERT_EQ(c2_gte_ts.size(), 1);

  HloInstruction* p2 = FindInstruction(module.get(), "p2");
  TF_ASSERT_OK_AND_ASSIGN(
      auto p2_ts,
      FindInstructionOutputTensors(func_tensor_map, *resources.get(), p2));
  ASSERT_EQ(p2_ts.size(), 1);

  EXPECT_EQ(arg2_ts[0].getContiguousRegions(), p2_ts[0].getContiguousRegions());
  EXPECT_EQ(arg4_ts[0].getContiguousRegions(), p2_ts[0].getContiguousRegions());
  EXPECT_EQ(c2_gte_ts[0].getContiguousRegions(),
            p2_ts[0].getContiguousRegions());
}

TEST_F(DeferredVisitorTest, TestFunctionKeepInputLayouts) {
  const string& hlo_string = R"(
HloModule module

func {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)
  dot = f32[8,8] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t = (f32[8,8]) tuple(dot)
}

func2 {
  p0.2 = f32[8,8] parameter(0)
  p1.2 = f32[8,8] parameter(1)
  dot.2 = f32[8,8] dot(p0.2, p1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2.2 = f32[8,8] parameter(2)
  add.2 = f32[8,8] add(dot.2, p2.2)
  ROOT t.2 = (f32[8,8]) tuple(dot.2)
}

ENTRY main {
  arg0 = f32[8,8] parameter(0)
  arg1 = f32[8,8] parameter(1)

  main_dot = f32[8,8] dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

  c1 = (f32[8,8]) call(main_dot, arg0), to_apply=func, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"

  zero = f32[] constant(0)
  bzero = f32[8,8] broadcast(zero), dimensions={}
  c2 = (f32[8,8]) call(arg1, main_dot, bzero), to_apply=func2, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  c1_gte = f32[8,8] get-tuple-element(c1), index=0
  c2_gte = f32[8,8] get-tuple-element(c2), index=0
  ROOT root_t = (f32[8,8],f32[8,8]) tuple(c1_gte, c2_gte)
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();

  HloInstruction* c2 = FindInstruction(module.get(), "c2");
  // Mark c2 as not reallocating.
  {
    auto backend_config =
        c2->backend_config<PoplarBackendConfig>().ValueOrDie();
    auto* call_config = backend_config.mutable_call_config();
    auto* function_cfg = call_config->mutable_function_config();
    function_cfg->set_keep_input_layouts(true);
    TF_ASSERT_OK(c2->set_backend_config(backend_config));
  }

  EntryVisitor visitor(*resources.get(), entry_computation);

  TF_EXPECT_OK(entry_computation->Accept(&visitor));
  auto entry_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("main");
  auto func_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("func");
  auto func2_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("func2");

  HloInstruction* main_dot = FindInstruction(module.get(), "main_dot");
  TF_ASSERT_OK_AND_ASSIGN(auto main_dot_ts,
                          FindInstructionOutputTensors(
                              entry_tensor_map, *resources.get(), main_dot));
  ASSERT_EQ(main_dot_ts.size(), 1);

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  TF_ASSERT_OK_AND_ASSIGN(
      auto p0_ts,
      FindInstructionOutputTensors(func_tensor_map, *resources.get(), p0));
  ASSERT_EQ(p0_ts.size(), 1);

  HloInstruction* p1_2 = FindInstruction(module.get(), "p1.2");
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1_2_ts,
      FindInstructionOutputTensors(func2_tensor_map, *resources.get(), p1_2));
  ASSERT_EQ(p1_2_ts.size(), 1);

  // c1 is allowed to reallocate, so the input tensors will not match.
  EXPECT_NE(main_dot_ts[0].getContiguousRegions(),
            p0_ts[0].getContiguousRegions());
  // c2 is not allowed to reallocate.
  EXPECT_EQ(main_dot_ts[0].getContiguousRegions(),
            p1_2_ts[0].getContiguousRegions());

  // bzero input to c2 has aliasing therefore it's reallocated.
  HloInstruction* bzero = FindInstruction(module.get(), "bzero");
  TF_ASSERT_OK_AND_ASSIGN(
      auto bzero_ts,
      FindInstructionOutputTensors(entry_tensor_map, *resources.get(), bzero));
  ASSERT_EQ(bzero_ts.size(), 1);

  HloInstruction* p2_2 = FindInstruction(module.get(), "p2.2");
  TF_ASSERT_OK_AND_ASSIGN(
      auto p2_2_ts,
      FindInstructionOutputTensors(func2_tensor_map, *resources.get(), p2_2));
  ASSERT_EQ(p2_2_ts.size(), 1);

  EXPECT_NE(bzero_ts[0].getContiguousRegions(),
            p2_2_ts[0].getContiguousRegions());
}

TEST_F(DeferredVisitorTest, TestConditional) {
  const std::string hlo_string = R"(
HloModule module
true_fn (arg_tuple.11: (f32[2,2], f32[2,2], f32[2,2])) -> (f32[2,2]) {
  arg_tuple.11 = (f32[2,2], f32[2,2], f32[2,2]) parameter(0)
  get-tuple-element.12 = f32[2,2] get-tuple-element(arg_tuple.11), index=0
  get-tuple-element.13 = f32[2,2] get-tuple-element(arg_tuple.11), index=1
  dot.15 = f32[2,2] dot(get-tuple-element.12, get-tuple-element.13), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.16 = (f32[2,2]) tuple(dot.15)
}

false_fn (arg_tuple.18: (f32[2,2], f32[2,2], f32[2,2])) -> (f32[2,2]) {
  arg_tuple.18 = (f32[2,2], f32[2,2], f32[2,2]) parameter(0)
  get-tuple-element.19 = f32[2,2] get-tuple-element(arg_tuple.18), index=0
  get-tuple-element.21 = f32[2,2] get-tuple-element(arg_tuple.18), index=2
  dot.22 = f32[2,2] dot(get-tuple-element.19, get-tuple-element.21), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.23 = (f32[2,2]) tuple(dot.22)
}

ENTRY main (arg0.1: f32[2,2], arg1.2: f32[2,2], arg2.3: f32[2,2], arg3.4: pred[]) -> f32[2,2] {
  arg3.4 = pred[] parameter(3), parameter_replication={false}
  arg0.1 = f32[2,2] parameter(0), parameter_replication={false}
  arg1.2 = f32[2,2] parameter(1), parameter_replication={false}
  arg2.3 = f32[2,2] parameter(2), parameter_replication={false}
  tuple.9 = (f32[2,2], f32[2,2], f32[2,2]) tuple(arg0.1, arg1.2, arg2.3)
  conditional.24 = (f32[2,2]) conditional(arg3.4, tuple.9, tuple.9), true_computation=true_fn, false_computation=false_fn
  ROOT get-tuple-element.25 = f32[2,2] get-tuple-element(conditional.24), index=0
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get(), false);
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  EXPECT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  auto entry_computation = module->entry_computation();

  EntryVisitor visitor(*resources.get(), entry_computation);
  VLOG(0) << module->ToString();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));
  auto entry_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("main");
  auto true_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("true_fn");
  auto false_tensor_map =
      resources->tensor_maps.GetTensorMapForComputation("false_fn");

  HloInstruction* dot_15 = FindInstruction(module.get(), "dot.15");
  TF_ASSERT_OK_AND_ASSIGN(
      auto dot_15_ts,
      FindInstructionOutputTensors(true_tensor_map, *resources.get(), dot_15));
  ASSERT_EQ(dot_15_ts.size(), 1);

  HloInstruction* dot_22 = FindInstruction(module.get(), "dot.22");
  TF_ASSERT_OK_AND_ASSIGN(
      auto dot_22_ts,
      FindInstructionOutputTensors(false_tensor_map, *resources.get(), dot_22));
  ASSERT_EQ(dot_22_ts.size(), 1);

  HloInstruction* cond = FindInstruction(module.get(), "conditional.24");
  TF_ASSERT_OK_AND_ASSIGN(
      auto cond_ts,
      FindInstructionOutputTensors(entry_tensor_map, *resources.get(), cond));
  ASSERT_EQ(cond_ts.size(), 1);

  HloInstruction* arg0_1 = FindInstruction(module.get(), "arg0.1");
  HloInstruction* arg1_2 = FindInstruction(module.get(), "arg1.2");
  HloInstruction* arg2_3 = FindInstruction(module.get(), "arg2.3");

  poplar::Tensor gte0_1_tensor =
      FindInstructionOutputs(entry_tensor_map, *resources.get(), arg0_1)[0];
  poplar::Tensor gte1_2_tensor =
      FindInstructionOutputs(entry_tensor_map, *resources.get(), arg1_2)[0];
  poplar::Tensor gte2_3_tensor =
      FindInstructionOutputs(entry_tensor_map, *resources.get(), arg2_3)[0];

  HloInstruction* get_tuple_element_12 =
      FindInstruction(module.get(), "get-tuple-element.12");
  HloInstruction* get_tuple_element_13 =
      FindInstruction(module.get(), "get-tuple-element.13");

  poplar::Tensor get_tuple_element_12_tensor = FindInstructionOutputs(
      true_tensor_map, *resources.get(), get_tuple_element_12)[0];
  poplar::Tensor get_tuple_element_13_tensor = FindInstructionOutputs(
      true_tensor_map, *resources.get(), get_tuple_element_13)[0];

  HloInstruction* get_tuple_element_19 =
      FindInstruction(module.get(), "get-tuple-element.19");
  HloInstruction* get_tuple_element_21 =
      FindInstruction(module.get(), "get-tuple-element.21");

  poplar::Tensor get_tuple_element_19_tensor = FindInstructionOutputs(
      false_tensor_map, *resources.get(), get_tuple_element_19)[0];
  poplar::Tensor get_tuple_element_21_tensor = FindInstructionOutputs(
      false_tensor_map, *resources.get(), get_tuple_element_21)[0];

  // Check the common input is the same in both branches.
  // True case:
  EXPECT_EQ(gte0_1_tensor.getContiguousRegions(),
            get_tuple_element_12_tensor.getContiguousRegions());
  // False case:
  EXPECT_EQ(gte0_1_tensor.getContiguousRegions(),
            get_tuple_element_19_tensor.getContiguousRegions());
  // For completeness, check that the true and false case are the same too.
  EXPECT_EQ(get_tuple_element_12_tensor.getContiguousRegions(),
            get_tuple_element_19_tensor.getContiguousRegions());

  // Check the per-branch rhs is the same as the correspoding entry argument.
  // True case:
  EXPECT_EQ(gte1_2_tensor.getContiguousRegions(),
            get_tuple_element_13_tensor.getContiguousRegions());
  // False case:
  EXPECT_EQ(gte2_3_tensor.getContiguousRegions(),
            get_tuple_element_21_tensor.getContiguousRegions());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
