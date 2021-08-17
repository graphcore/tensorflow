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

#include <poplar/IPUModel.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#define TF_ASSERT_OK(rexpr) \
  auto statusor = (rexpr);  \
  ASSERT_TRUE(statusor.ok()) << statusor;

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using OpaqueHandlingTest = HloTestBase;

std::unique_ptr<CompilerResources> GetMockResources(HloModule* module,
                                                    bool merge_infeeds) {
  auto res = CompilerResources::CreateTestDefault(module);
  res->merge_infeed_io_copies = merge_infeeds;
  res->module_call_graph = CallGraph::Build(module);
  res->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), poplar::replication_factor(1));
  poplin::addCodelets(*res->main_graph);
  popnn::addCodelets(*res->main_graph);
  popops::addCodelets(*res->main_graph);
  poprand::addCodelets(*res->main_graph);
  return std::move(res);
}

poplar::Device createIpuModel(int IPUCount = 1, int IPUTileCount = 1216) {
  poplar::IPUModel model;

  model.numIPUs = IPUCount;
  model.tilesPerIPU = IPUTileCount;

  return model.createDevice();
}

std::unique_ptr<CompilerResources> GetMockResources(
    HloModule* module, bool merge_infeeds, int number_of_vgraphs,
    int64 max_inter_ipu_copies_buffer_size = 0) {
  const auto info = CompilerInformation().set_max_inter_ipu_copies_buffer_size(
      max_inter_ipu_copies_buffer_size);
  auto resources = CompilerResources::CreateTestDefault(module, info);
  resources->merge_infeed_io_copies = merge_infeeds;
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      createIpuModel(number_of_vgraphs, 4), poplar::replication_factor(1));

  // Add mock vgraphs
  for (int i = 0; i < number_of_vgraphs; ++i) {
    resources->shard_compute_graphs.emplace_back(
        resources->main_graph->createVirtualGraph(i * 4, (i + 1) * 4));
  }
  resources->shard_to_ipu_id.resize(number_of_vgraphs);
  absl::c_iota(resources->shard_to_ipu_id, 0);

  poplin::addCodelets(*resources->main_graph);
  popnn::addCodelets(*resources->main_graph);
  popops::addCodelets(*resources->main_graph);
  poprand::addCodelets(*resources->main_graph);
  return std::move(resources);
}

TEST_F(OpaqueHandlingTest, SimpleCall) {
  const string& hlo = R"(
HloModule module

comp_0 {
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  ROOT r = (opaque[], opaque[], opaque[]) tuple(param_2, param_0, param_1)
}

ENTRY entry {
  ROOT r = f32[] parameter(3)
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  call_0 = (opaque[], opaque[], opaque[]) call(param_0, param_1, param_2), to_apply=comp_0
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  auto res = GetMockResources(module.get(), false);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue&) { return 0; }));
  module->set_schedule(schedule);

  HloComputation* entry = module->GetComputationWithName("entry");
  HloInstruction* call_0 = entry->GetInstructionWithName("call_0");
  HloInstruction* param_0 = entry->parameter_instruction(0);
  HloInstruction* param_1 = entry->parameter_instruction(1);
  HloInstruction* param_2 = entry->parameter_instruction(2);

  TensorMap tensor_map;
  tensor_map.AddOutputOpaque(param_0, 0, {0});
  tensor_map.AddOutputOpaque(param_1, 0, {1});
  tensor_map.AddOutputOpaque(param_2, 0, {2});
  res->deferred_allocation_scopes.push(DeferredAllocations{tensor_map});
  ExecutionCounters execution_counters(*res, {});
  res->execution_counter_scopes.push(&execution_counters);

  TF_ASSERT_OK_AND_ASSIGN(
      auto program,
      CreateCallOp(*res, call_0, call_0->shape(), tensor_map, {}));

  auto outputs = tensor_map.FindInstructionOutputs(call_0);
  EXPECT_TRUE(outputs[0].IsOpaque());
  EXPECT_TRUE(outputs[1].IsOpaque());
  EXPECT_TRUE(outputs[2].IsOpaque());
  EXPECT_EQ(absl::any_cast<int>(outputs[0].AsOpaque()), 2);
  EXPECT_EQ(absl::any_cast<int>(outputs[1].AsOpaque()), 0);
  EXPECT_EQ(absl::any_cast<int>(outputs[2].AsOpaque()), 1);
}

TEST_F(OpaqueHandlingTest, SimpleRepeat) {
  const string& hlo = R"(
HloModule module

comp_0 {
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  ROOT r = (opaque[], opaque[], opaque[]) tuple(param_0, param_1, param_2)
}

ENTRY entry {
  ROOT r = f32[] parameter(3)
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  call_0 = (opaque[], opaque[], opaque[]) call(param_0, param_1, param_2), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  auto res = GetMockResources(module.get(), false);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue&) { return 0; }));
  module->set_schedule(schedule);

  HloComputation* entry = module->GetComputationWithName("entry");
  HloInstruction* call_0 = entry->GetInstructionWithName("call_0");
  HloInstruction* param_0 = entry->parameter_instruction(0);
  HloInstruction* param_1 = entry->parameter_instruction(1);
  HloInstruction* param_2 = entry->parameter_instruction(2);

  TensorMap tensor_map;
  tensor_map.AddOutputOpaque(param_0, 0, {0});
  tensor_map.AddOutputOpaque(param_1, 0, {1});
  tensor_map.AddOutputOpaque(param_2, 0, {2});
  res->deferred_allocation_scopes.push(DeferredAllocations{tensor_map});
  ExecutionCounters execution_counters(*res, {});
  res->execution_counter_scopes.push(&execution_counters);

  TF_ASSERT_OK_AND_ASSIGN(
      auto program,
      CreateCallOp(*res, call_0, call_0->shape(), tensor_map, {}));

  auto outputs = tensor_map.FindInstructionOutputs(call_0);
  EXPECT_TRUE(outputs[0].IsOpaque());
  EXPECT_TRUE(outputs[1].IsOpaque());
  EXPECT_TRUE(outputs[2].IsOpaque());
  EXPECT_EQ(absl::any_cast<int>(outputs[0].AsOpaque()), 0);
  EXPECT_EQ(absl::any_cast<int>(outputs[1].AsOpaque()), 1);
  EXPECT_EQ(absl::any_cast<int>(outputs[2].AsOpaque()), 2);
}

TEST_F(OpaqueHandlingTest, InvalidRepeat) {
  const string& hlo = R"(
HloModule module

comp_0 {
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  ROOT r = (opaque[], opaque[], opaque[]) tuple(param_2, param_0, param_1)
}

ENTRY entry {
  ROOT r = f32[] parameter(3)
  param_0 = opaque[] parameter(0)
  param_1 = opaque[] parameter(1)
  param_2 = opaque[] parameter(2)
  call_0 = (opaque[], opaque[], opaque[]) call(param_0, param_1, param_2), to_apply=comp_0, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"100\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  auto res = GetMockResources(module.get(), false);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue&) { return 0; }));
  module->set_schedule(schedule);

  HloComputation* entry = module->GetComputationWithName("entry");
  HloInstruction* call_0 = entry->GetInstructionWithName("call_0");
  HloInstruction* param_0 = entry->parameter_instruction(0);
  HloInstruction* param_1 = entry->parameter_instruction(1);
  HloInstruction* param_2 = entry->parameter_instruction(2);

  TensorMap tensor_map;
  tensor_map.AddOutputOpaque(param_0, 0, {0});
  tensor_map.AddOutputOpaque(param_1, 0, {1});
  tensor_map.AddOutputOpaque(param_2, 0, {2});
  res->deferred_allocation_scopes.push(DeferredAllocations{tensor_map});
  ExecutionCounters execution_counters(*res, {});
  res->execution_counter_scopes.push(&execution_counters);

  EXPECT_FALSE(
      CreateCallOp(*res, call_0, call_0->shape(), tensor_map, {}).ok());
}

TEST_F(OpaqueHandlingTest, SimplePipeline) {
  const string& hlo = R"(
HloModule module

_stage_0 {
  param_0 = opaque[] parameter(0), sharding={maximal device=0}
  param_1 = opaque[] parameter(1), sharding={maximal device=0}
  param_2 = opaque[] parameter(2), sharding={maximal device=0}
  ROOT t = (opaque[], opaque[], opaque[]) tuple(param_0, param_1, param_2), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = opaque[] parameter(0), sharding={maximal device=0}
  param_1 = opaque[] parameter(1), sharding={maximal device=0}
  param_2 = opaque[] parameter(2), sharding={maximal device=0}
  ROOT t = (opaque[], opaque[], opaque[]) tuple(param_0, param_1, param_2), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = opaque[] parameter(0), sharding={maximal device=0}
  param_1 = opaque[] parameter(1), sharding={maximal device=0}
  param_2 = opaque[] parameter(2), sharding={maximal device=0}
  ROOT t = (opaque[], opaque[], opaque[]) tuple(param_0, param_1, param_2), sharding={{maximal device=1}}
}

_stage_0_bw {
  param_0 = opaque[] parameter(0), sharding={maximal device=0}
  param_1 = opaque[] parameter(1), sharding={maximal device=0}
  param_2 = opaque[] parameter(2), sharding={maximal device=0}
  ROOT t = (opaque[], opaque[], opaque[]) tuple(param_0, param_1, param_2), sharding={{maximal device=0}}
}

ENTRY pipeline {
  param_0 = opaque[] parameter(0), sharding={maximal device=0}
  param_1 = opaque[] parameter(1), sharding={maximal device=0}
  param_2 = opaque[] parameter(2), sharding={maximal device=0}

  a = (opaque[], opaque[], opaque[]) call(param_0, param_1, param_2), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a0 = opaque[] get-tuple-element(a), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_a1 = opaque[] get-tuple-element(a), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_a2 = opaque[] get-tuple-element(a), index=2, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  b = (opaque[], opaque[], opaque[]) call(gte_a0, gte_a1, gte_a2), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b0 = opaque[] get-tuple-element(b), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_b1 = opaque[] get-tuple-element(b), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_b2 = opaque[] get-tuple-element(b), index=2, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  c = (opaque[], opaque[], opaque[]) call(gte_b0, gte_b1, gte_b2), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c0 = opaque[] get-tuple-element(c), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_c1 = opaque[] get-tuple-element(c), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  gte_c2 = opaque[] get-tuple-element(c), index=2, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  ROOT d = (opaque[], opaque[], opaque[]) call(gte_c0, gte_c1, gte_c2), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  auto res = GetMockResources(module.get(), false, 2);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(module.get(), [](const BufferValue&) { return 0; }));
  module->set_schedule(schedule);

  HloComputation* entry = module->GetComputationWithName("pipeline");
  HloInstruction* param_0 = entry->parameter_instruction(0);
  HloInstruction* param_1 = entry->parameter_instruction(1);
  HloInstruction* param_2 = entry->parameter_instruction(2);

  const auto stage_count =
      absl::c_count_if(entry->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {param_0, 0},
      {param_1, 0},
      {param_2, 0},
      {entry->GetInstructionWithName("a"), 0},
      {entry->GetInstructionWithName("gte_a0"), 0},
      {entry->GetInstructionWithName("gte_a1"), 0},
      {entry->GetInstructionWithName("gte_a2"), 0},
      {entry->GetInstructionWithName("b"), 1},
      {entry->GetInstructionWithName("gte_b0"), 1},
      {entry->GetInstructionWithName("gte_b1"), 1},
      {entry->GetInstructionWithName("gte_b2"), 1},
      {entry->GetInstructionWithName("c"), 2},
      {entry->GetInstructionWithName("gte_c0"), 2},
      {entry->GetInstructionWithName("gte_c1"), 2},
      {entry->GetInstructionWithName("gte_c2"), 2},
      {entry->GetInstructionWithName("d"), 3},
  };

  DeferredArgRBVectors inputs = {{TensorOrRemoteBuffer{absl::any{0}}},
                                 {TensorOrRemoteBuffer{absl::any{1}}},
                                 {TensorOrRemoteBuffer{absl::any{2}}}};

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *res, inputs,
      HloInstructionDescription(entry->root_instruction()), "visitor");
  TF_ASSERT_OK(entry->Accept(&visitor));

  auto outputs = visitor.outputs();
  EXPECT_EQ(outputs.size(), 3);
  EXPECT_TRUE(outputs[0].IsOpaque());
  EXPECT_TRUE(outputs[1].IsOpaque());
  EXPECT_TRUE(outputs[2].IsOpaque());
  EXPECT_EQ(absl::any_cast<int>(outputs[0].AsOpaque()), 0);
  EXPECT_EQ(absl::any_cast<int>(outputs[1].AsOpaque()), 1);
  EXPECT_EQ(absl::any_cast<int>(outputs[2].AsOpaque()), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
