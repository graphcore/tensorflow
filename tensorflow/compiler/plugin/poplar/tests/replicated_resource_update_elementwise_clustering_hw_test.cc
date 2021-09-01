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

#include "tensorflow/compiler/plugin/poplar/driver/passes/replicated_resource_update_elementwise_clustering.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/elementwise_cluster.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
using tu = HloPoplarTestUtil;
namespace {

std::string GetInfeedCopyHandle(const std::string& name, int64 shape_index) {
  return tensorflow::strings::Printf("infeed_%s.%lld", name.c_str(),
                                     shape_index);
}

struct ReplicatedResourceUpdateElementwiseClusteringHwTestSpec {
  std::string hlo;
  std::string short_name;
  int32 replication_factor;
};

std::ostream& operator<<(
    std::ostream& os,
    const ReplicatedResourceUpdateElementwiseClusteringHwTestSpec& spec) {
  return os << "{ name: " << spec.short_name
            << ", replication-factor: " << spec.replication_factor << "}";
}

class TestStreamCallback final : public poplar::StreamCallback {
 public:
  TestStreamCallback(std::string name, int64 size, float start)
      : name_(name), size_(size), current_(start) {}

 public:
  Result prefetch(void* p) override { return Result::NotAvailable; }

  void fetch(void* p) override {
    VLOG(2) << "Fetching [" << name_ << ", " << size_ << "]...";
    float* dst = static_cast<float*>(p);
    auto n = size_;
    auto current = current_;
    while (n--) {
      *dst++ = current++;
    }
  }

  void complete() override { current_ += size_; }

 private:
  std::string name_;
  int64 size_;
  float current_;
};

using RemoteBuffers = std::vector<std::vector<float>>;

class ReplicatedResourceUpdateElementwiseClusteringHwTest
    : public HloPoplarTestBase,
      public ::testing::WithParamInterface<
          ReplicatedResourceUpdateElementwiseClusteringHwTestSpec> {
 public:
  void RunTest(
      const ReplicatedResourceUpdateElementwiseClusteringHwTestSpec& param,
      bool cluster, RemoteBuffers& buffers) {
    TF_ASSERT_OK_AND_ASSIGN(auto device,
                            CreateIpuDevice(param.replication_factor, 4));

    auto config = GetModuleConfigForTest();
    config.set_argument_input_indices({});
    config.set_resource_input_indices({0, 1, 2, 3});
    config.set_resource_input_initialized({true, true, true, true});
    config.set_resource_update_to_input_index({0, 1, 2, 3});
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(param.hlo, config));

    auto resources =
        GetMockResources(device, module.get(), param.replication_factor);
    CompilerAnnotations& annotations = resources->annotations;

    TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                            CustomOpReplacer().Run(module.get()));
    TF_ASSERT_OK_AND_ASSIGN(
        bool offloaded, VariablesOffloadAndPartition(annotations, true, 0,
                                                     param.replication_factor)
                            .Run(module.get()));
    EXPECT_TRUE(offloaded);

    if (cluster) {
      TF_ASSERT_OK_AND_ASSIGN(bool changed,
                              ReplicatedResourceUpdateElementwiseClustering(
                                  param.replication_factor)
                                  .Run(module.get()));
      EXPECT_TRUE(changed);
    }

    TF_ASSERT_OK_AND_ASSIGN(bool inlined,
                            FusionInliner([](const HloInstruction* inst) {
                              return IsReplicatedParameterLoadFusion(inst) ||
                                     IsReplicatedParameterStoreFusion(inst);
                            })
                                .Run(module.get()));
    EXPECT_TRUE(inlined);
    EXPECT_TRUE(HloDCE().Run(module.get()).ok());

    EXPECT_TRUE(InplaceFinder().Run(module.get()).ok());
    EXPECT_TRUE(AllocationFinder(resources->annotations,
                                 resources->always_rearrange_copies_on_host)
                    .Run(module.get())
                    .ok());
    EXPECT_TRUE(HloPassFix<ForwardAllocation>(resources->annotations)
                    .Run(module.get())
                    .ok());
    XLA_VLOG_LINES(1, module->ToString());

    auto root = module->entry_computation()->root_instruction();
    HloComputation* repeat_computation =
        root->operand(0)->operand(0)->to_apply();
    HloInstruction* repeat_root = repeat_computation->root_instruction();
    HloComputation* resource_update =
        repeat_root->mutable_operand(0)->mutable_operand(0)->to_apply();

    HloInstruction* resource_update_root = resource_update->root_instruction();
    EXPECT_EQ(resource_update->num_parameters(), 4);
    EXPECT_EQ(ShapeUtil::TupleElementCount(resource_update_root->shape()), 4);

    TF_ASSERT_OK_AND_ASSIGN(auto engine, Compile(*resources, module.get()));
    engine.load(device);

    EXPECT_EQ(resources->annotations.remote_parameter_infos.size(), 2);

    float start = 1;
    auto& io_map = resources->annotations.input_output_aliasing_map;
    auto& inputs = io_map.GetEntryInputInfos();
    auto& outputs = io_map.GetEntryOutputInfos();
    auto& remote_infos = resources->annotations.remote_parameter_infos;

    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto& input = inputs[index];
      VLOG(1) << "Input name: " << input.Name() << ", shape: " << input.Shape()
              << ", streaming: " << input.IsStreaming();
      auto size = ShapeUtil::ElementsIn(input.Shape());
      auto per_replica_size =
          PartitionedElementCountPerReplica(size, param.replication_factor);
      auto aligned_size = per_replica_size * param.replication_factor;

      for (auto& handle : input.Handles()) {
        VLOG(1) << " handle: " << handle;
        auto it = remote_infos.find(RemoteParameterInfo(index));
        if (it != remote_infos.end()) {
          auto& info = *it;

          VLOG(1) << "Uploading data to " << info.buffer_name
                  << ", offset: " << info.buffer_offset
                  << ", replicated: " << info.is_replica_partitioned;

          std::vector<float> buffer(aligned_size);
          std::iota(buffer.begin(), buffer.begin() + size, start);
          start += size;
          for (unsigned replica = 0; replica < param.replication_factor;
               ++replica) {
            engine.copyToRemoteBuffer(
                buffer.data() + replica * per_replica_size, info.buffer_name,
                info.buffer_offset, replica);
            if (!info.is_replica_partitioned) {
              break;
            }
          }
        } else {
          std::unique_ptr<poplar::StreamCallback> infeed_callback =
              absl::make_unique<TestStreamCallback>(handle, size, 10 + start++);
          engine.connectStreamToCallback(handle, 0, std::move(infeed_callback));
        }
      }
    }

    for (const auto& infeed_info : annotations.infeed_infos) {
      VLOG(1) << "Connecting infeed " << infeed_info.config.feed_id()
              << " of shape " << infeed_info.shape << ".";
      const Shape& shape = infeed_info.shape;
      EXPECT_TRUE(shape.IsTuple());
      EXPECT_EQ(shape.tuple_shapes_size(), 2);
      const Shape& input_shape = infeed_info.shape.tuple_shapes(0);
      auto size = ShapeUtil::ElementsIn(input_shape);
      for (auto replica_id = 0; replica_id < param.replication_factor;
           ++replica_id) {
        auto handle = GetInfeedCopyHandle(infeed_info.config.feed_id(), 0);
        std::unique_ptr<poplar::StreamCallback> infeed_callback =
            absl::make_unique<TestStreamCallback>(handle, size, 1);
        engine.connectStreamToCallback(handle, replica_id,
                                       std::move(infeed_callback));
      }
    }

    // Run the program.
    engine.run(0);

    for (std::size_t index = 0; index < outputs.size(); ++index) {
      auto& output = outputs[index];
      VLOG(1) << "Output name: " << output.Name()
              << ", shape: " << output.Shape()
              << ", streaming: " << output.IsStreaming();
      auto size = ShapeUtil::ElementsIn(output.Shape());
      auto per_replica_size =
          PartitionedElementCountPerReplica(size, param.replication_factor);
      auto aligned_size = per_replica_size * param.replication_factor;
      for (auto& handle : output.Handles()) {
        VLOG(1) << " handle: " << handle;
        auto it = remote_infos.find(RemoteParameterInfo(index));
        if (it != remote_infos.end()) {
          auto& info = *it;

          VLOG(1) << "Downloading data from " << info.buffer_name
                  << ", offset: " << info.buffer_offset
                  << ", replicated: " << info.is_replica_partitioned;

          std::vector<float> buffer(aligned_size);
          for (auto replica_id = 0; replica_id < param.replication_factor;
               ++replica_id) {
            engine.copyFromRemoteBuffer(
                info.buffer_name, buffer.data() + replica_id * per_replica_size,
                info.buffer_offset, replica_id);
          }
          buffer.resize(size);
          buffers.push_back(std::move(buffer));
        }
      }
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    ReplicatedResourceUpdateElementwiseClusteringHwTestCases,
    ReplicatedResourceUpdateElementwiseClusteringHwTest,
    ::testing::ValuesIn(
        std::vector<ReplicatedResourceUpdateElementwiseClusteringHwTestSpec>{
            {tu::GetSimpleHloString(99, 7), "simple", 2},
            {tu::GetSGDHloString(99, 7), "sgd", 2},
            {tu::GetAdamLikeHloString(99, 7), "adam", 2},
            {tu::GetMomentumLikeHloString(99, 7), "momentum", 2},
            {tu::GetTwoClustersShareInputHloString(99, 7), "shared-inputs", 2},
        }));

TEST_P(ReplicatedResourceUpdateElementwiseClusteringHwTest, DoTest) {
  auto param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto tf_ipu_count, GetMaxIpuCount());
  if (param.replication_factor > tf_ipu_count) {
    GTEST_SKIP() << "Skipping test, replication factor "
                 << param.replication_factor << ", max ipu: " << tf_ipu_count;
    return;
  }
  RemoteBuffers clustered_buffers;
  RunTest(param, true, clustered_buffers);
  RemoteBuffers buffers;
  RunTest(param, false, buffers);
  EXPECT_THAT(clustered_buffers.size(), 2);
  EXPECT_THAT(clustered_buffers.size(), buffers.size());
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    EXPECT_THAT(buffers[i], ::testing::Eq(clustered_buffers[i]));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
