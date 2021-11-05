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

#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_merger.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

struct GraphCachingWithRemoteBuffersTestSpec {
  bool hw;
  unsigned n;
  unsigned replication_factor;
  unsigned shard_size;
};

std::ostream& operator<<(std::ostream& os,
                         const GraphCachingWithRemoteBuffersTestSpec& spec) {
  return os << "{ "
            << "n: " << spec.n
            << ", replication factor: " << spec.replication_factor
            << ", shard_size: " << spec.shard_size << " }";
}

class GraphCachingWithRemoteBuffersTest
    : public HloPoplarTestBase,
      public ::testing::WithParamInterface<
          GraphCachingWithRemoteBuffersTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    GraphCachingWithRemoteBuffersTestCases, GraphCachingWithRemoteBuffersTest,
    ::testing::ValuesIn(std::vector<GraphCachingWithRemoteBuffersTestSpec>{
        {false, 0, 1, 0},
        {false, 1, 1, 1},
        {true, 0, 1, 0},
        {true, 1, 1, 1},
        {true, 2, 2, 1},
        {true, 256, 4, 64},
    }));

TEST_P(GraphCachingWithRemoteBuffersTest, WithBufferMerging) {
  auto param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto tf_ipu_count, GetMaxIpuCount());
  if (param.hw && param.replication_factor > tf_ipu_count) {
    GTEST_SKIP() << "Skipping test, replication factor "
                 << param.replication_factor << ", max ipu: " << tf_ipu_count;
    return;
  }

  const std::string& hlo_string_template = R"(
HloModule module

comp_0 {
  c0_param_0 = f32[$N] parameter(0)
  c0_param_1 = f32[$N] parameter(1)

  c0_load_0 = f32[$RN] custom-call(c0_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  c0_load_1 = f32[$RN] custom-call(c0_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"

  c0_add = f32[$RN] add(c0_load_0, c0_load_1)
  c0_subtract = f32[$RN] subtract(c0_load_0, c0_load_1)

  c0_new_0 = f32[$N] custom-call(c0_param_0, c0_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"
  c0_new_1 = f32[$N] custom-call(c0_param_1, c0_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"

  ROOT c0_t = (f32[$N], f32[$N]) tuple(c0_new_0, c0_new_1)
}

comp_1 {
  c1_param_0 = f32[$N] parameter(0)
  c1_param_1 = f32[$N] parameter(1)

  c1_load_0 = f32[$RN] custom-call(c1_param_0), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"
  c1_load_1 = f32[$RN] custom-call(c1_param_1), custom_call_target="RemoteParameterLoad", backend_config="{\"replication_factor\":$R}\n"

  c1_add = f32[$RN] add(c1_load_0, c1_load_1)
  c1_subtract = f32[$RN] subtract(c1_load_0, c1_load_1)

  c1_new_0 = f32[$N] custom-call(c1_param_0, c1_add), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"
  c1_new_1 = f32[$N] custom-call(c1_param_1, c1_subtract), custom_call_target="RemoteParameterStore", backend_config="{\"replication_factor\":$R}\n"

  ROOT c1_t = (f32[$N], f32[$N]) tuple(c1_new_0, c1_new_1)
}

ENTRY main {
  param_0 = f32[$N] parameter(0)
  param_1 = f32[$N] parameter(1)
  c0 = (f32[$N], f32[$N]) call(param_0, param_1), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\"}}}"
  new_param_0 = f32[$N] get-tuple-element(c0), index=0
  new_param_1 = f32[$N] get-tuple-element(c0), index=1
  param_2 = f32[$N] parameter(2)
  param_3 = f32[$N] parameter(3)
  c1 = (f32[$N], f32[$N]) call(param_2, param_3), to_apply=comp_1, backend_config="{\"callConfig\":{\"type\":\"Function\", \"functionConfig\":{\"numModifiedRemoteBufferInputs\":\"2\"}}}"
  new_param_2 = f32[$N] get-tuple-element(c1), index=0
  new_param_3 = f32[$N] get-tuple-element(c1), index=1
  ROOT t = (f32[$N], f32[$N], f32[$N], f32[$N]) tuple(new_param_0, new_param_1, new_param_2, new_param_3)
}
)";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template,
      {{"$RN",
        param.shard_size ? std::to_string(param.shard_size) : std::string()},
       {"$R", std::to_string(param.replication_factor)},
       {"$N", param.n ? std::to_string(param.n) : std::string()}});

  poplar::Device device;
  if (param.hw) {
    TF_ASSERT_OK_AND_ASSIGN(device,
                            CreateIpuDevice(param.replication_factor, 4));
  } else {
    device = CreateIpuModel();
  }

  auto config = GetModuleConfigForTest();
  config.set_argument_input_indices({});
  config.set_resource_input_indices({0, 1, 2, 3});
  config.set_resource_input_initialized({true, true, true, true});
  config.set_resource_update_to_input_index({0, 1, 2, 3});

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  auto resources =
      GetMockResources(device, module.get(), param.replication_factor);

  for (int64 i = 0; i != 4; ++i) {
    resources->annotations.remote_parameter_infos.insert(RemoteParameterInfo{
        i, param.replication_factor > 1, absl::StrCat(i, ".0"),
        /*buffer_offset=*/0, /*num_merged=*/1});
  }

  EXPECT_TRUE(CustomOpReplacer().Run(module.get()).ValueOrDie());
  EXPECT_TRUE(
      InplaceFinder(resources->annotations).Run(module.get()).ValueOrDie());
  EXPECT_TRUE(RemoteBufferMerger(resources->annotations, THREESTATE_ON)
                  .Run(module.get())
                  .ValueOrDie());

  // Check that there is only one buffer name now.
  absl::flat_hash_set<std::string> buffer_names;
  for (auto& info : resources->annotations.remote_parameter_infos) {
    buffer_names.insert(info.buffer_name);
  }
  ASSERT_EQ(buffer_names.size(), 1);

  const std::string buffer_name = *std::begin(buffer_names);

  TF_ASSERT_OK_AND_ASSIGN(auto engine, Compile(*resources, module.get()));
  // Check that only two computations have been compiled - entry and one of the
  // comps.
  CHECK_EQ(resources->tensor_maps.size(), 2);

  // Run on the device.
  engine.load(device);

  // Set up the values.
  std::size_t buffer_size = param.shard_size > 0 ? param.shard_size : 1;
  std::vector<float> inputs = {10.f, 2.f, -1.f, 20.f};
  for (int64 i = 0; i != 4; ++i) {
    for (unsigned replica = 0; replica < param.replication_factor; ++replica) {
      // Adding replica index to buffer content just to make sure it's different
      // across replicas.
      std::vector<float> buffer(buffer_size, inputs[i] + replica);
      engine.copyToRemoteBuffer(buffer.data(), buffer_name, i, replica);
    }
  }

  // Run the program.
  engine.run(0);

  // Get the values back.
  std::vector<float> outputs = {12.f, 8.f, 19.f, -21.f};
  for (int64 i = 0; i != 4; ++i) {
    auto value = outputs[i];
    std::vector<float> buffer(buffer_size);
    for (unsigned replica = 0; replica < param.replication_factor; ++replica) {
      // Calculate output value.
      // Input: { A, B, C, D}
      // Output: { A + B, A - B, C + D, C - D}
      // Adding replica index will result in double-adding for even input
      // indices.
      std::vector<float> output(buffer_size,
                                outputs[i] + (i % 2 ? 0 : 2 * replica));
      engine.copyFromRemoteBuffer(buffer_name, buffer.data(), i, replica);
      EXPECT_THAT(buffer, ::testing::Eq(output));
    }
  }
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
