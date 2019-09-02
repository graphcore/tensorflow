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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

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

#include <sstream>

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class PipelineVisitorTest : public HloTestBase {};

std::unique_ptr<CompilerResources> GetMockResources(poplar::Device& device,
                                                    HloModule* module,
                                                    bool merge_infeeds) {
  auto resources = absl::make_unique<CompilerResources>(
      poplar::OptionFlags(), poplar::OptionFlags(), poplar::OptionFlags(),
      false, false, merge_infeeds, 1, 0, 0, 1, 64, module,
      IpuOptions::FloatingPointBehaviour());
  resources->main_graph = absl::make_unique<poplar::Graph>(
      device, 0, poplar::replication_factor(1));

  // Add 2 mock vgraphs
  resources->shard_graphs.emplace_back(
      resources->main_graph->createVirtualGraph(0, 4));
  resources->shard_graphs.emplace_back(
      resources->main_graph->createVirtualGraph(4, 8));

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

_stage_0 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
}

_stage_1 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_1_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(2), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_0_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
  ROOT result = f32[] constant(4), sharding={maximal device=0}
}

ENTRY pipeline (arg: f32[]) -> f32[] {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = f32[] call(arg), to_apply=_stage_0, sharding={maximal device=0}

  b0 = f32[] call(a0), to_apply=_stage_1, sharding={maximal device=1}

  c0 = f32[] call(b0), to_apply=_stage_1_bw, sharding={maximal device=1}

  ROOT d = f32[] call(c0), to_apply=_stage_0_bw, sharding={maximal device=0}
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false);

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
  const std::vector<std::pair<std::string, int>> stage_assignments = {
      {"arg", 0},           {"a0", 0}, {"b0", 1}, {"c0", 2}, {"d", 3},
      {"custom-call.4", 0},  // Inter-IPU-copy between stage 0 and 1
      {"custom-call.5", 2},  // Inter-IPU-copy between stage 2 and 3
  };

  for (auto stage_assignment : stage_assignments) {
    auto instruction =
        entry_computation->GetInstructionWithName(stage_assignment.first);
    resources->pipeline_stage_assignment[instruction] = stage_assignment.second;
  }

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(stage_count, 5, *resources, {{placeholder}});
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence();

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
/custom-call: 1
/custom-call.2: 4
/custom-call.1: 2
/custom-call.3: 5
/custom-call: 1
/custom-call.2: 4
/custom-call.1: 2
/custom-call.3: 5
/custom-call: 1
/custom-call.2: 4
/custom-call.1: 2
/custom-call.3: 5
/custom-call: 1
/custom-call.2: 4
/custom-call.1: 2
/custom-call.3: 5
/custom-call.2: 4
/custom-call.3: 5
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the print tensor statements get printed in the expected
// order with the same stage being executed on the same IPU.
TEST_F(PipelineVisitorTest, TestPipelineVisitorOrderDuplicated) {
  const string& hlo_string = R"(
HloModule module

_stage_0 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
}

_stage_1 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_1_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_0_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
  ROOT result = f32[] constant(4), sharding={maximal device=0}
}

ENTRY pipeline (arg: f32[]) -> f32[] {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = f32[] call(arg), to_apply=_stage_0, sharding={maximal device=0}

  b0 = f32[] call(a0), to_apply=_stage_1, sharding={maximal device=1}

  c0 = f32[] call(b0), to_apply=_stage_1_bw, sharding={maximal device=1}

  ROOT d = f32[] call(c0), to_apply=_stage_0_bw, sharding={maximal device=0}
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false);

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
  const std::vector<std::pair<std::string, int>> stage_assignments = {
      {"arg", 0},           {"a0", 0}, {"b0", 1}, {"c0", 2}, {"d", 3},
      {"custom-call.4", 0},  // Inter-IPU-copy between stage 0 and 1
      {"custom-call.5", 2},  // Inter-IPU-copy between stage 2 and 3
  };

  for (auto stage_assignment : stage_assignments) {
    auto instruction =
        entry_computation->GetInstructionWithName(stage_assignment.first);
    resources->pipeline_stage_assignment[instruction] = stage_assignment.second;
  }

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(stage_count, 5, *resources, {{placeholder}});
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence();

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
/custom-call: 1
/custom-call.1: 3
/custom-call.1: 2
/custom-call.3: 4
/custom-call: 1
/custom-call.1: 3
/custom-call.1: 2
/custom-call.3: 4
/custom-call: 1
/custom-call.1: 3
/custom-call.1: 2
/custom-call.3: 4
/custom-call: 1
/custom-call.1: 3
/custom-call.1: 2
/custom-call.3: 4
/custom-call.1: 3
/custom-call.3: 4
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program.
TEST_F(PipelineVisitorTest, TestPipelineVisitorValue) {
  const string& hlo_string = R"(
HloModule module

_stage_0 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
}

_stage_1 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_1_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_0_bw (arg_0: f32[]) -> token[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
}

ENTRY pipeline (arg: f32[]) -> token[] {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = f32[] call(arg), to_apply=_stage_0, sharding={maximal device=0}

  b0 = f32[] call(a0), to_apply=_stage_1, sharding={maximal device=1}

  c0 = f32[] call(b0), to_apply=_stage_1_bw, sharding={maximal device=1}

  ROOT d = token[] call(c0), to_apply=_stage_0_bw, sharding={maximal device=0}
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false);

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
  const std::vector<std::pair<std::string, int>> stage_assignments = {
      {"arg", 0},           {"a0", 0}, {"b0", 1}, {"c0", 2}, {"d", 3},
      {"custom-call.1", 0},  // Inter-IPU-copy between stage 0 and 1
      {"custom-call.2", 2},  // Inter-IPU-copy between stage 2 and 3
  };

  for (auto stage_assignment : stage_assignments) {
    auto instruction =
        entry_computation->GetInstructionWithName(stage_assignment.first);
    resources->pipeline_stage_assignment[instruction] = stage_assignment.second;
  }

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(stage_count, 5, *resources, {{placeholder}});
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence();

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
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoValue) {
  const string& hlo_string = R"(
HloModule module

_stage_0 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
}

_stage_1 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_1_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_0_bw (arg_0: f32[], arg_1: f32[]) -> token[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  add_0 = f32[] add(param_0, param_1), sharding={maximal device=0}
  ROOT token_f = token[] custom-call(add_0), custom_call_target="Poputil::PrintTensor", backend_config="{}\n", sharding={maximal device=0}
}

ENTRY pipeline (arg: f32[]) -> token[] {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = f32[] call(arg), to_apply=_stage_0, sharding={maximal device=0}
  a1 = f32[] custom-call(a0), custom_call_target="Poputil::Fifo", backend_config="{\"depth\":1}\n", sharding={maximal device=0}

  b0 = f32[] call(a0), to_apply=_stage_1, sharding={maximal device=1}

  c0 = f32[] call(b0), to_apply=_stage_1_bw, sharding={maximal device=1}

  ROOT d = token[] call(c0, a1), to_apply=_stage_0_bw, sharding={maximal device=0}
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false);

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
  const std::vector<std::pair<std::string, int>> stage_assignments = {
      {"arg", 0},           {"a0", 0}, {"b0", 1}, {"c0", 2}, {"d", 3},
      {"custom-call.1", 0},  // Inter-ipu-copy between stage 0 and 1
      {"custom-call.2", 2},  // Inter-ipu-copy between stage 2 and 3
      {"custom-call.3", 0},  // FIFO after stage 0
  };

  for (auto stage_assignment : stage_assignments) {
    auto instruction =
        entry_computation->GetInstructionWithName(stage_assignment.first);
    resources->pipeline_stage_assignment[instruction] = stage_assignment.second;
  }

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(stage_count, 5, *resources, {{placeholder}});
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence();

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
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that poplar OnTileExecute programs overlap sufficiently for a
// pipeline computation.
TEST_F(PipelineVisitorTest, TestPipelineVisitorFifoOverlap) {
  const string& hlo_string = R"(
HloModule module

_stage_0 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
}

_stage_1 (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_1_bw (arg_0: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  ROOT add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
}

_stage_0_bw (arg_0: f32[], arg_1: f32[]) -> f32[] {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  param_1 = f32[] parameter(1), sharding={maximal device=0}
  ROOT add_0 = f32[] add(param_0, param_1), sharding={maximal device=0}
}

ENTRY pipeline (arg: f32[]) -> f32[] {
  arg = f32[] parameter(0), sharding={maximal device=0}

  a0 = f32[] call(arg), to_apply=_stage_0, sharding={maximal device=0}
  a1 = f32[] custom-call(a0), custom_call_target="Poputil::Fifo", backend_config="{\"depth\":1}\n", sharding={maximal device=0}

  b0 = f32[] call(a0), to_apply=_stage_1, sharding={maximal device=1}

  c0 = f32[] call(b0), to_apply=_stage_1_bw, sharding={maximal device=1}

  ROOT d = f32[] call(c0, a1), to_apply=_stage_0_bw, sharding={maximal device=0}
}
)";
  auto device = createIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false);

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
  const std::vector<std::pair<std::string, int>> stage_assignments = {
      {"arg", 0},           {"a0", 0}, {"b0", 1}, {"c0", 2}, {"d", 3},
      {"custom-call", 0},    // Inter-ipu-copy between stage 0 and 1
      {"custom-call.1", 2},  // Inter-ipu-copy between stage 2 and 3
      {"custom-call.2", 0},  // FIFO after stage 0
  };

  for (auto stage_assignment : stage_assignments) {
    auto instruction =
        entry_computation->GetInstructionWithName(stage_assignment.first);
    resources->pipeline_stage_assignment[instruction] = stage_assignment.second;
  }

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);

  PipelineVisitor visitor(stage_count, 5, *resources, {{placeholder}});
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence();

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

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
