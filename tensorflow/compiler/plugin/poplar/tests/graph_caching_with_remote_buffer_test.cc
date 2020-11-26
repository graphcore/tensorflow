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

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <popops/codelets.hpp>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_merger.h"
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

using GraphCachingWithRemoteBuffersTest = HloTestBase;

std::unique_ptr<CompilerResources> GetMockResources(poplar::Device& device,
                                                    HloModule* module) {
  auto resources = CompilerResources::CreateTestDefault(module);
  resources->streams_indices.InitializeIndexTensors(*resources, {});
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph =
      absl::make_unique<poplar::Graph>(device, poplar::replication_factor(1));

  popops::addCodelets(*resources->main_graph);

  return std::move(resources);
}

poplar::Device CreateIpuModel(int ipu_count = 1, int num_tiles = 1216) {
  poplar::IPUModel model;
  model.numIPUs = ipu_count;
  model.tilesPerIPU = num_tiles;
  return model.createDevice();
}

TEST_F(GraphCachingWithRemoteBuffersTest, WithBufferMerging) {
  const string& hlo_string = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[] parameter(0)
  c0_param_1 = f32[] parameter(1)

  c0_load_0 = f32[] custom-call(c0_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  c0_load_1 = f32[] custom-call(c0_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c0_add = f32[] add(c0_load_0, c0_load_1)
  c0_subtract = f32[] subtract(c0_load_0, c0_load_1)

  c0_new_0 = f32[] custom-call(c0_param_0, c0_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  c0_new_1 = f32[] custom-call(c0_param_1, c0_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  ROOT c0_t = (f32[], f32[]) tuple(c0_new_0, c0_new_1)
}

comp_1 {
  c1_param_0 = f32[] parameter(0)
  c1_param_1 = f32[] parameter(1)

  c1_load_0 = f32[] custom-call(c1_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"
  c1_load_1 = f32[] custom-call(c1_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":1}\n"

  c1_add = f32[] add(c1_load_0, c1_load_1)
  c1_subtract = f32[] subtract(c1_load_0, c1_load_1)

  c1_new_0 = f32[] custom-call(c1_param_0, c1_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"
  c1_new_1 = f32[] custom-call(c1_param_1, c1_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":1}\n"

  ROOT c1_t = (f32[], f32[]) tuple(c1_new_0, c1_new_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  c0 = (f32[], f32[]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\"}}}"
  new_param_0 = f32[] get-tuple-element(c0), index=0
  new_param_1 = f32[] get-tuple-element(c0), index=1
  param_2 = f32[] parameter(2)
  param_3 = f32[] parameter(3)
  c1 = (f32[], f32[]) call(param_2, param_3), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\"}}}"
  new_param_2 = f32[] get-tuple-element(c1), index=0
  new_param_3 = f32[] get-tuple-element(c1), index=1
  ROOT t = (f32[], f32[], f32[], f32[]) tuple(new_param_0, new_param_1, new_param_2, new_param_3)
}
)";
  auto device = CreateIpuModel(1, 4);

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(4);
  config.set_input_mapping({0, 1, 2, 3});
  config.set_resource_update_to_input_index({0, 1, 2, 3});

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  auto resources = GetMockResources(device, module.get());

  for (int64 i = 0; i != 4; ++i) {
    resources->annotations.remote_parameter_infos.insert(
        RemoteParameterInfo{i, false, absl::StrCat(i, ".0"),
                            /*buffer_offset=*/0, /*num_merged=*/1});
  }

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(InplaceFinder().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(RemoteBufferMerger(resources->annotations, THREESTATE_ON)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_TRUE(HloTrivialScheduler().Run(module.get()).ValueOrDie());

  // Check that there is only one buffer name now.
  absl::flat_hash_set<std::string> buffer_names;
  for (auto& info : resources->annotations.remote_parameter_infos) {
    buffer_names.insert(info.buffer_name);
  }
  ASSERT_EQ(buffer_names.size(), 1);

  const std::string buffer_name = *std::begin(buffer_names);
  auto entry = module->entry_computation();
  auto order = module->schedule().sequence(entry).instructions();
  EntryVisitor visitor(*resources.get(), entry);

  TF_ASSERT_OK(entry->AcceptOrdered(&visitor, order));

  // Check that only two computations have been compiled - entry and one of the
  // comps.
  CHECK_EQ(resources->tensor_maps.size(), 2);

  poplar::program::Sequence main_program;
  main_program.add(resources->preamble_sequence);
  main_program.add(visitor.GetSequenceAndInitializeCounters());

  poplar::Engine engine(*resources->main_graph, main_program);

  // Run on the device.
  device.attach();
  engine.load(device);

  // Set up the values.
  std::vector<float> inputs = {10.f, 2.f, -1.f, 20.f};
  for (int64 i = 0; i != 4; ++i) {
    engine.copyToRemoteBuffer(&inputs[i], buffer_name, i);
  }

  // Run the program.
  engine.run(0);

  // Get the values back.
  std::vector<float> outputs(4);
  for (int64 i = 0; i != 4; ++i) {
    engine.copyFromRemoteBuffer(buffer_name, &outputs[i], i);
  }

  EXPECT_THAT(outputs, ::testing::ElementsAre(12.f, 8.f, 19.f, -21.f));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
