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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dropout_with_reference_finder.h"

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <poprand/codelets.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using DropoutWithReferenceFinderTest = HloTestBase;

std::unique_ptr<CompilerResources> GetMockResources(poplar::Device& device,
                                                    HloModule* module) {
  auto resources = CompilerResources::CreateTestDefault(module);
  resources->streams_indices.InitializeIndexTensors(*resources, {});
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph =
      absl::make_unique<poplar::Graph>(device, poplar::replication_factor(1));
  resources->shard_to_ipu_id = {0};

  poprand::addCodelets(*resources->main_graph);

  return std::move(resources);
}

poplar::Device CreateIpuModel(int ipu_count = 1, int num_tiles = 1216) {
  poplar::IPUModel model;
  model.numIPUs = ipu_count;
  model.tilesPerIPU = num_tiles;
  return model.createDevice();
}

TEST_F(DropoutWithReferenceFinderTest, TestMatched) {
  const string& hlo_string = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[10] parameter(0)
  c0_param_1 = s32[2] parameter(1)
  ROOT c0_dropout = (f32[10], s32[2]) custom-call(c0_param_0, c0_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : true,\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
}

comp_1 {
  c1_param_0 = f32[10] parameter(0)
  c1_param_1 = s32[2] parameter(1)
  ROOT c1_dropout = (f32[10], s32[2]) custom-call(c1_param_0, c1_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : false,\n\t\"rate\" : 0.5,\n\t\"scale\" : 3.0\n}"
}

ENTRY main {
  param_0 = f32[10] parameter(0)
  param_1 = s32[2] parameter(1)
  param_2 = f32[10] parameter(2)

  c0 = (f32[10], s32[2]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  seed = s32[2] get-tuple-element(c0), index=1

  c1 = (f32[10], s32[2]) call(param_2, seed), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  ROOT output = f32[10] get-tuple-element(c1), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto device = CreateIpuModel(1, 4);
  auto resources = GetMockResources(device, module.get());

  CompilerAnnotations& annotations = resources->annotations;

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(ModuleFlatten(annotations).Run(module.get()).ValueOrDie());
  ASSERT_TRUE(
      DropoutWithReferenceFinder(annotations).Run(module.get()).ValueOrDie());

  EXPECT_TRUE(HloTrivialScheduler().Run(module.get()).ValueOrDie());

  auto entry = module->entry_computation();
  auto order = module->schedule().sequence(entry).instructions();
  EntryVisitor visitor(*resources.get(), entry);
  TF_ASSERT_OK(entry->AcceptOrdered(&visitor, order));

  EXPECT_EQ(resources->reference_tensors.size(), 1);
  // Check that the dropout output has the same mapping as the reference tensor.
  auto reference_tensor =
      std::begin(resources->reference_tensors)->second.first;

  auto tensor_map = resources->tensor_maps.GetTensorMapForComputation("comp_1");
  HloComputation* comp_1 = FindComputation(module.get(), "comp_1");
  HloInstruction* bwd_dropout = comp_1->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(
      auto bwd_dropout_ts,
      FindInstructionOutputTensors(tensor_map, *resources.get(), bwd_dropout));
  ASSERT_EQ(bwd_dropout_ts.size(), 2);

  EXPECT_EQ(reference_tensor.getContiguousRegions(),
            bwd_dropout_ts[0].getContiguousRegions());
}

TEST_F(DropoutWithReferenceFinderTest, TestSeedIsOutput) {
  const string& hlo_string = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[10] parameter(0)
  c0_param_1 = s32[2] parameter(1)
  ROOT c0_dropout = (f32[10], s32[2]) custom-call(c0_param_0, c0_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : true,\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
}

comp_1 {
  c1_param_0 = f32[10] parameter(0)
  c1_param_1 = s32[2] parameter(1)
  ROOT c1_dropout = (f32[10], s32[2]) custom-call(c1_param_0, c1_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : false,\n\t\"rate\" : 0.5,\n\t\"scale\" : 3.0\n}"
}

ENTRY main {
  param_0 = f32[10] parameter(0)
  param_1 = s32[2] parameter(1)
  param_2 = f32[10] parameter(2)

  c0 = (f32[10], s32[2]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  seed = s32[2] get-tuple-element(c0), index=1

  c1 = (f32[10], s32[2]) call(param_2, seed), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  output = f32[10] get-tuple-element(c1), index=0
  ROOT tuple = (f32[10], s32[2]) tuple(output, seed)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CompilerAnnotations annotations(module.get());

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(ModuleFlatten(annotations).Run(module.get()).ValueOrDie());
  EXPECT_FALSE(
      DropoutWithReferenceFinder(annotations).Run(module.get()).ValueOrDie());
}

TEST_F(DropoutWithReferenceFinderTest, NoDropoutReferenceTensorCreators) {
  const string& hlo_string = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[10] parameter(0)
  c0_param_1 = s32[2] parameter(1)
  ROOT c0_dropout = (f32[10], s32[2]) custom-call(c0_param_0, c0_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : false,\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
}

comp_1 {
  c1_param_0 = f32[10] parameter(0)
  c1_param_1 = s32[2] parameter(1)
  ROOT c1_dropout = (f32[10], s32[2]) custom-call(c1_param_0, c1_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : false,\n\t\"rate\" : 0.5,\n\t\"scale\" : 3.0\n}"
}

ENTRY main {
  param_0 = f32[10] parameter(0)
  param_1 = s32[2] parameter(1)
  param_2 = f32[10] parameter(2)

  c0 = (f32[10], s32[2]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  seed = s32[2] get-tuple-element(c0), index=1

  c1 = (f32[10], s32[2]) call(param_2, seed), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  ROOT output = f32[10] get-tuple-element(c1), index=0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CompilerAnnotations annotations(module.get());

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(ModuleFlatten(annotations).Run(module.get()).ValueOrDie());
  EXPECT_FALSE(
      DropoutWithReferenceFinder(annotations).Run(module.get()).ValueOrDie());
}

TEST_F(DropoutWithReferenceFinderTest, TestSeedIsRoot) {
  const string& hlo_string = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[10] parameter(0)
  c0_param_1 = s32[2] parameter(1)
  ROOT c0_dropout = (f32[10], s32[2]) custom-call(c0_param_0, c0_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : true,\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
}

comp_1 {
  c1_param_0 = f32[10] parameter(0)
  c1_param_1 = s32[2] parameter(1)
  ROOT c1_dropout = (f32[10], s32[2]) custom-call(c1_param_0, c1_param_1), custom_call_target="Dropout", backend_config="{\n\t\"can_create_reference_tensor\" : false,\n\t\"rate\" : 0.5,\n\t\"scale\" : 3.0\n}"
}

ENTRY main {
  param_0 = f32[10] parameter(0)
  param_1 = s32[2] parameter(1)
  param_2 = f32[10] parameter(2)

  c0 = (f32[10], s32[2]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
  seed = s32[2] get-tuple-element(c0), index=1

  ROOT c1 = (f32[10], s32[2]) call(param_2, seed), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\"}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CompilerAnnotations annotations(module.get());

  ASSERT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  ASSERT_TRUE(ModuleFlatten(annotations).Run(module.get()).ValueOrDie());
  EXPECT_FALSE(
      DropoutWithReferenceFinder(annotations).Run(module.get()).ValueOrDie());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
