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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <sstream>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

namespace xla {
namespace poplarplugin {
namespace {

class PipelineVisitorTest : public HloTestBase {};

std::unique_ptr<CompilerResources> GetMockResources(
    poplar::Device& device, HloModule* module, bool merge_infeeds,
    int number_of_vgraphs, int64 max_inter_ipu_copies_buffer_size = 0) {
  auto resources = absl::make_unique<CompilerResources>(
      poplar::OptionFlags(), poplar::OptionFlags(), poplar::OptionFlags(),
      false, false, false, false, merge_infeeds, 1, 0, 0,
      max_inter_ipu_copies_buffer_size, 0, 1, 64, module,
      IpuOptions::FloatingPointBehaviour(), false, "", false, false, false,
      poplar::OptionFlags(), 0, false, false);
  resources->streams_indices.InitializeIndexTensors(*resources, {});
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph =
      absl::make_unique<poplar::Graph>(device, poplar::replication_factor(1));

  // Add mock vgraphs
  for (int i = 0; i < number_of_vgraphs; ++i) {
    resources->shard_graphs.emplace_back(
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

poplar::Device createIpuModel(int IPUCount = 1, int IPUTileCount = 1216) {
  poplar::IPUModel model;

  model.numIPUs = IPUCount;
  model.tilesPerIPU = IPUTileCount;

  return model.createDevice();
}

// This tests that the print tensor statements get printed in the expected
// order, given a pipeline poplar control program.
TEST_F(PipelineVisitorTest, TestPipelineVisitorOrder) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(2), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  result = f32[] constant(4), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(result), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = (f32[]) call(gte_c), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-IPU-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.4"), 0},
      // Inter-IPU-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.5"), 2},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(4).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: 1
/custom-call.1: 2
/custom-call.2: 4
/custom-call: 1
/custom-call.3: 5
/custom-call.1: 2
/custom-call: 1
/custom-call.2: 4
/custom-call.1: 2
/custom-call.3: 5
/custom-call.2: 4
/custom-call: 1
/custom-call.3: 5
/custom-call.1: 2
/custom-call.2: 4
/custom-call.3: 5
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program.
TEST_F(PipelineVisitorTest, TestPipelineVisitorValue) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c), sharding={{maximal device=0}}, to_apply=_stage_0_bw, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-IPU-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.1"), 0},
      // Inter-IPU-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.2"), 2},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: 4
/custom-call: 4
/custom-call: 4
/custom-call: 4
/custom-call: 4
/custom-call: 4
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoValue) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, param_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.2"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.3"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call.1"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(4).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: 4
/custom-call: 4
/custom-call: 4
/custom-call: 4
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoValueTuples) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  a = f32[2] parameter(0), sharding={maximal device=0}
  const_0 = f32[2] constant({100,2000}), sharding={maximal device=0}
  add_0 = f32[2] add(a, const_0), sharding={maximal device=0}
  add_1 = f32[2] add(const_0, const_0), sharding={maximal device=0}
  concat = f32[4] concatenate(add_1, add_0), sharding={maximal device=0}, dimensions={0}
  out = (f32[2], f32[4], f32[2], f32[2]) tuple(add_0, concat, add_1, const_0), sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  ROOT t = ((f32[2], f32[4], f32[2], f32[2])) tuple(out), sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
}

_stage_1 {
  tuple = (f32[2], f32[4], f32[2], f32[2]) parameter(0), sharding={{maximal device=1},{maximal device=1},{maximal device=1},{maximal device=1}}
  a = f32[2] get-tuple-element(tuple), index=0, sharding={maximal device=1}
  const_1 = f32[2] constant({1,2}), sharding={maximal device=1}
  add_1 = f32[2] add(a, const_1), sharding={maximal device=1}
  ROOT t = (f32[2]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[2] parameter(0), sharding={maximal device=1}
  const_1_bw = f32[2] constant({5,10}), sharding={maximal device=1}
  add_1_bw = f32[2] add(param_0, const_1_bw), sharding={maximal device=1}
  ROOT t = (f32[2]) tuple(add_1_bw), sharding={{maximal device=1}}
}

_stage_0_bw {
  param = f32[2] parameter(0), sharding={maximal device=0}
  fifo_tuple = (f32[2], f32[4], f32[2], f32[2]) parameter(1), sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  add_1 = f32[2] get-tuple-element(fifo_tuple), index=2, sharding={maximal device=0}

  add_0 = f32[2] add(param, add_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[2] parameter(0), sharding={maximal device=0}

  a0 = ((f32[2], f32[4], f32[2], f32[2])) call(arg), to_apply=_stage_0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = (f32[2], f32[4], f32[2], f32[2]) get-tuple-element(a0), index=0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"isInplace\":true}"
  a1 = (f32[2], f32[4], f32[2], f32[2]) custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"depth\":1}", sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.2"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.3"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call.1"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {2});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(8).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
/custom-call: {306,6012}
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that poplar OnTileExecute programs overlap sufficiently for a
// pipeline computation.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoOverlap) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, param_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_0), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}
  ROOT d = (f32[]) call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.1"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call.2"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program);
  engine.enableExecutionProfiling();

  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  // Get the execution steps
  auto exec_profile = engine.getExecutionProfile();
  auto steps = exec_profile["simulation"]["steps"].asVector();

  // Only consider the on tile execute steps
  auto is_on_tile_exec_pred = [](const poplar::ProfileValue& step) -> bool {
    return step["type"].asString() == "OnTileExecute";
  };
  auto itr =
      std::stable_partition(steps.begin(), steps.end(), is_on_tile_exec_pred);

  // And only consider the computation steps
  auto is_add_pred = [](const poplar::ProfileValue& step) -> bool {
    return step.asMap().count("name") == 1 &&
           absl::StrContains(step["name"].asString(), "add") &&
           !absl::StrContains(step["name"].asString(), "Copy");
  };
  itr = std::stable_partition(steps.begin(), itr, is_add_pred);
  steps.erase(itr, steps.end());

  // Compute the total number of cycles that were overlapped
  auto overlapped_cycles = [](int accum,
                              const poplar::ProfileValue& step) -> int {
    return accum + step["cyclesOverlapped"].asInt();
  };
  int total_overlapped_cycles =
      std::accumulate(steps.begin(), steps.end(), 0, overlapped_cycles);

  // Check we overlapped enough cycles. This value was determined empirically
  ASSERT_GT(total_overlapped_cycles, 1500);
}

// This tests that poplar OnTileExecute programs overlap sufficiently for a
// pipeline computation.
TEST_F(PipelineVisitorTest, TestPipelineVisitorRevisitIPU) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_2 {
  param_0 = f32[] parameter(0), sharding={maximal device=2}
  const_1 = f32[] constant(1), sharding={maximal device=2}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=2}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=2}}
}

_stage_2_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  param_1 = f32[] parameter(1), sharding={maximal device=1}
  add_0 = f32[] add(param_0, param_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_0), sharding={{maximal device=1}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"depth\":2}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  b1 = f32[] custom-call(gte_b), custom_call_target="Fifo", backend_config="{\"depth\":2}", sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_2, sharding={{maximal device=2}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=2}, backend_config="{\"isInplace\":true}"
  d0 = (f32[]) call(gte_c), to_apply=_stage_2_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_d = f32[] get-tuple-element(d0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  e0 = (f32[]) call(gte_d, b1), to_apply=_stage_1_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_e = f32[] get-tuple-element(e0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  ROOT d = (f32[]) call(gte_e, a1), to_apply=_stage_0_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(4, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 4);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d0"), 3},
      {entry_computation->GetInstructionWithName("gte_d"), 3},
      {entry_computation->GetInstructionWithName("e0"), 4},
      {entry_computation->GetInstructionWithName("gte_e"), 4},
      {entry_computation->GetInstructionWithName("d"), 5},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call"), 0},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call.1"), 1},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.2"), 0},
      // Inter-ipu-copy between stage 1 and 2
      {entry_computation->GetInstructionWithName("custom-call.3"), 1},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.4"), 2},
      // Inter-ipu-copy between stage 3 and 4
      {entry_computation->GetInstructionWithName("custom-call.5"), 3},
      // Inter-ipu-copy between stage 1 and 4
      {entry_computation->GetInstructionWithName("custom-call.6"), 1},
      // Inter-ipu-copy between stage 4 and 5
      {entry_computation->GetInstructionWithName("custom-call.7"), 4},
      // Inter-ipu-copy between stage 0 and 5
      {entry_computation->GetInstructionWithName("custom-call.8"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 2, 1, 0, 1}, stage_assignments, {}, 3, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program);
  engine.enableExecutionProfiling();

  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  // Get the execution steps
  auto exec_profile = engine.getExecutionProfile();
  auto steps = exec_profile["simulation"]["steps"].asVector();

  // Only consider the on tile execute steps
  auto is_on_tile_exec_pred = [](const poplar::ProfileValue& step) -> bool {
    return step["type"].asString() == "OnTileExecute";
  };
  auto itr =
      std::stable_partition(steps.begin(), steps.end(), is_on_tile_exec_pred);

  // And only consider the computation steps
  auto is_add_pred = [](const poplar::ProfileValue& step) -> bool {
    return step.asMap().count("name") == 1 &&
           absl::StrContains(step["name"].asString(), "add") &&
           !absl::StrContains(step["name"].asString(), "Copy");
  };
  itr = std::stable_partition(steps.begin(), itr, is_add_pred);
  steps.erase(itr, steps.end());

  // Compute the total number of cycles that were overlapped
  auto overlapped_cycles = [](int accum,
                              const poplar::ProfileValue& step) -> int {
    return accum + step["cyclesOverlapped"].asInt();
  };
  int total_overlapped_cycles =
      std::accumulate(steps.begin(), steps.end(), 0, overlapped_cycles);

  // Check we overlapped enough cycles. This value was determined empirically
  ASSERT_GT(total_overlapped_cycles, 2000);
}

// Tests that poplar revisits IPUs in the expected order.
// We expect the schedule to look like
// ||A|A| | |E|E||A|A| | |E|E||A| | | |E| ||
// || |B|B|D|D|F||F|B|B|D|D|F||F|B| |D| |F||
// || | |C|C| | || | |C|C| | || | |C| | | ||
// ||  RAMP-UP  ||   REPEAT  || RAMP-DOWN ||
TEST_F(PipelineVisitorTest, TestPipelineVisitorRevisitIPUOrder) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_2 {
  param_0 = f32[] parameter(0), sharding={maximal device=2}
  const_1 = f32[] constant(1), sharding={maximal device=2}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=2}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=2}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=2}}
}

_stage_2_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_0), sharding={{maximal device=1}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_2, sharding={{maximal device=2}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=2}
  d0 = (f32[]) call(gte_c), to_apply=_stage_2_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_d = f32[] get-tuple-element(d0), index=0, sharding={maximal device=1}
  e0 = (f32[]) call(gte_d), to_apply=_stage_1_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_e = f32[] get-tuple-element(e0), index=0, sharding={maximal device=0}
  ROOT d = (f32[]) call(gte_e), to_apply=_stage_0_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(4, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 4);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d0"), 3},
      {entry_computation->GetInstructionWithName("gte_d"), 3},
      {entry_computation->GetInstructionWithName("e0"), 4},
      {entry_computation->GetInstructionWithName("gte_e"), 4},
      {entry_computation->GetInstructionWithName("d"), 5},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.6"), 0},
      // Inter-ipu-copy between stage 1 and 2
      {entry_computation->GetInstructionWithName("custom-call.7"), 1},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.8"), 2},
      // Inter-ipu-copy between stage 3 and 4
      {entry_computation->GetInstructionWithName("custom-call.9"), 3},
      // Inter-ipu-copy between stage 4 and 5
      {entry_computation->GetInstructionWithName("custom-call.10"), 4},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 2, 1, 0, 1}, stage_assignments, {}, 3, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(8).ValueOrDie();

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program);
  engine.enableExecutionProfiling();
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: 0
/custom-call.1: 1
/custom-call: 0
/custom-call.2: 2
/custom-call.1: 1
/custom-call.3: 3
/custom-call.2: 2
/custom-call.4: 4
/custom-call.3: 3
/custom-call.5: 5
/custom-call.4: 4
/custom-call: 0
/custom-call.5: 5
/custom-call.1: 1
/custom-call: 0
/custom-call.2: 2
/custom-call.1: 1
/custom-call.3: 3
/custom-call.2: 2
/custom-call.4: 4
/custom-call.3: 3
/custom-call.5: 5
/custom-call.4: 4
/custom-call: 0
/custom-call.5: 5
/custom-call.1: 1
/custom-call: 0
/custom-call.2: 2
/custom-call.1: 1
/custom-call.3: 3
/custom-call.2: 2
/custom-call.4: 4
/custom-call.3: 3
/custom-call.5: 5
/custom-call.4: 4
/custom-call: 0
/custom-call.5: 5
/custom-call.1: 1
/custom-call: 0
/custom-call.2: 2
/custom-call.1: 1
/custom-call.3: 3
/custom-call.2: 2
/custom-call.4: 4
/custom-call.3: 3
/custom-call.5: 5
/custom-call.4: 4
/custom-call.5: 5
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
// Also make sure that aliasing is preserved by the FIFO.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoValueBroadcastTuples) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  a = f32[] parameter(0), sharding={maximal device=0}
  const_0 = f32[] constant(100), sharding={maximal device=0}
  add_0 = f32[] add(a, const_0), sharding={maximal device=0}
  bcast = f32[2] broadcast(add_0), dimensions={}, sharding={maximal device=0}
  ROOT out = (f32[2]) tuple(bcast), sharding={{maximal device=0}}, backend_config="{\"isInplace\":true}"
}

_stage_1 {
  a = f32[2] parameter(0), sharding={maximal device=1}
  const_1 = f32[2] constant({1,2}), sharding={maximal device=1}
  add_1 = f32[2] add(a, const_1), sharding={maximal device=1}
  ROOT t = (f32[2]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[2] parameter(0), sharding={maximal device=1}
  const_1_bw = f32[2] constant({5,10}), sharding={maximal device=1}
  add_1_bw = f32[2] add(param_0, const_1_bw), sharding={maximal device=1}
  ROOT t = (f32[2]) tuple(add_1_bw), sharding={{maximal device=1}}
}

_stage_0_bw {
  param = f32[2] parameter(0), sharding={maximal device=0}
  param1 = f32[2] parameter(1), sharding={maximal device=0}
  add_0 = f32[2] add(param, param1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}
  ", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

ENTRY pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = (f32[2]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[2] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[2] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("custom-call.2"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("custom-call.3"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("custom-call.1"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}}}, "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(8).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
/custom-call: {206,212}
)";

  // Check the output of the stage has aliases.
  ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("_stage_0")
                  .FindTensorByName("out", 0)
                  .containsAliases());
  // Check that the fifo has aliases.
  ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("pipeline")
                  .FindTensorByName("custom-call.1", 0)
                  .containsAliases());
  ASSERT_EQ(expected, ss.str());
}

TEST_F(PipelineVisitorTest, TestPipelineVisitorMergedCopies) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  ROOT t = (f32[], f32[]) tuple(param_0, param_1), sharding={{maximal device=0}, {maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  param_1 = f32[] parameter(1), sharding={maximal device=1}
  add = f32[] add(param_0, param_1), sharding={maximal device=1}
  token_f = token[] custom-call(add), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={maximal device=1}
}

pipeline {
  p0 = f32[] parameter(0), sharding={maximal device=0}
  p1 = f32[] parameter(1), sharding={maximal device=0}
  ps0 = (f32[], f32[]) call(p0, p1), to_apply=_stage_0, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}}
  ps0_0 = f32[] get-tuple-element(ps0), index=0, sharding={maximal device=0}
  ps0_1 = f32[] get-tuple-element(ps0), index=1, sharding={maximal device=0}
  ps1 = () call(ps0_0, ps0_1), to_apply=_stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  ROOT t = () tuple()
}

ENTRY e {
  p0 = f32[] parameter(0), parameter_replication={false}, sharding={maximal device=0}
  p1 = f32[] parameter(1), parameter_replication={false}, sharding={maximal device=0}
  ROOT c = () call(p0, p1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2, 1024);
  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());
  auto size_function = [](const BufferValue& buffer) -> int64 {
    return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
  };

  IpuScheduler scheduler(
      size_function, CreateClusteringMemoryScheduler(resources->information));
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  CombineInstructions combiner;
  EXPECT_TRUE(combiner.Run(module.get()).ValueOrDie());

  auto entry_computation = module->entry_computation();
  auto pipeline = entry_computation->root_instruction();
  auto pipeline_comp = pipeline->to_apply();

  auto p0 = resources->main_graph->addConstant(poplar::FLOAT, {}, 1.f);
  resources->main_graph->setTileMapping(p0, 0);
  auto p1 = resources->main_graph->addConstant(poplar::FLOAT, {}, 2.f);
  resources->main_graph->setTileMapping(p1, 0);

  PipelineVisitor visitor(pipeline, *resources,
                          DeferredArgRBVectors{{TensorOrRemoteBuffer{p0}},
                                               {TensorOrRemoteBuffer{p1}}},
                          "visitor");
  TF_EXPECT_OK(pipeline_comp->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph, program);

  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/custom-call: 3
/custom-call: 3
/custom-call: 3
/custom-call: 3
/custom-call: 3
/custom-call: 3
)";

  ASSERT_EQ(expected, ss.str());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
