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
#include <poplar/IPUModel.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <pva/pva.hpp>
#include <sstream>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
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
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using PipelineGroupedVisitorTest = HloPoplarTestBase;

// This tests that the print tensor statements get printed in the expected
// order, given a pipeline poplar control program.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorOrder) {
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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}


  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = (f32[]) call(gte_c), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},

      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-IPU-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-IPU-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
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

  const std::string expected = R"(/print-tensor: 1
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor.3: 5
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorValue) {
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
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c), sharding={{maximal device=0}}, to_apply=_stage_0_bw, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-IPU-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-IPU-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
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

  const std::string expected = R"(/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorFifoValue) {
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
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("fifo"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
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

  const std::string expected = R"(/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorFifoValueTuples) {
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
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

pipeline {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = ((f32[2], f32[4], f32[2], f32[2])) call(arg), to_apply=_stage_0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = (f32[2], f32[4], f32[2], f32[2]) get-tuple-element(a0), index=0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"isInplace\":true}"
  a1 = (f32[2], f32[4], f32[2], f32[2]) custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("fifo"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {2});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
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

  const std::string expected = R"(/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that poplar OnTileExecute programs overlap sufficiently for a
// pipeline computation.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorFifoOverlap) {
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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}
  ROOT d = (f32[]) call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("fifo"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(8).ValueOrDie();

  // Create unique reporting directory to avoid tests overwriting
  // eachother's reports.
  TemporaryDirManager dir_manager(
      "PipelineGroupedVisitorTest.TestPipelineVisitorFifoOverlap");

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program,
                        /*options=*/
                        {{"debug.retainDebugInformation", "true"},
                         {"autoReport.all", "true"},
                         {"autoReport.directory", dir_manager.GetDirName()}});
  engine.enableExecutionProfiling();

  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  // Get the execution steps
  auto report = engine.getReport(/*reportExecution=*/true);
  std::vector<pva::ExecutionStep> steps(report.execution().steps().begin(),
                                        report.execution().steps().end());

  // Only consider the on tile execute steps
  auto is_on_tile_exec_pred = [](pva::ExecutionStep& step) -> bool {
    return step.program()->type() == pva::Program::Type::OnTileExecute;
  };
  auto itr =
      std::stable_partition(steps.begin(), steps.end(), is_on_tile_exec_pred);

  // And only consider the computation steps
  auto is_add_pred = [](pva::ExecutionStep& step) -> bool {
    return absl::StrContains(step.program()->name(), "add") &&
           !absl::StrContains(step.program()->name(), "Copy");
  };
  itr = std::stable_partition(steps.begin(), itr, is_add_pred);
  steps.erase(itr, steps.end());

  // Compute the total number of cycles that were overlapped.
  auto overlapped_cycles = [](int accum, pva::ExecutionStep& step) -> int {
    for (auto& ipu : step.ipus()) {
      accum += ipu.allCycles().from().max() - ipu.allCycles().from().min();
    }
    return accum;
  };
  // Compute the total number of cycles.
  auto cycles = [](int accum, pva::ExecutionStep& step) -> int {
    for (auto& ipu : step.ipus()) {
      accum += ipu.cycles();
    }
    return accum;
  };
  int total_overlapped_cycles =
      std::accumulate(steps.begin(), steps.end(), 0, overlapped_cycles);
  int total_cycles = std::accumulate(steps.begin(), steps.end(), 0, cycles);
  // Check we overlapped enough cycles. This value was determined empirically
  ASSERT_GT((static_cast<float>(total_overlapped_cycles)) / total_cycles, 0.49);
}

// This tests that poplar OnTileExecute programs overlap sufficiently for a
// pipeline computation.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorRevisitIPU) {
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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":4}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  b1 = f32[] custom-call(gte_b), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":4}", sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_2, sharding={{maximal device=2}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=2}, backend_config="{\"isInplace\":true}"
  d0 = (f32[]) call(gte_c), to_apply=_stage_2_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  gte_d = f32[] get-tuple-element(d0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  e0 = (f32[]) call(gte_d, b1), to_apply=_stage_1_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_e = f32[] get-tuple-element(e0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  ROOT d = (f32[]) call(gte_e, a1), to_apply=_stage_0_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(12), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(4, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 4);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
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
      {entry_computation->GetInstructionWithName("fifo"), 0},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("fifo.1"), 1},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 1 and 2
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 1},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.2"), 2},
      // Inter-ipu-copy between stage 3 and 4
      {entry_computation->GetInstructionWithName("ipu-inter-copy.3"), 3},
      // Inter-ipu-copy between stage 1 and 4
      {entry_computation->GetInstructionWithName("ipu-inter-copy.4"), 1},
      // Inter-ipu-copy between stage 4 and 5
      {entry_computation->GetInstructionWithName("ipu-inter-copy.5"), 4},
      // Inter-ipu-copy between stage 0 and 5
      {entry_computation->GetInstructionWithName("ipu-inter-copy.6"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 12);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 2, 1, 0, 1}, stage_assignments, {}, 3, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(12).ValueOrDie();

  // Create unique reporting directory to avoid tests overwriting
  // eachother's reports.
  TemporaryDirManager dir_manager(
      "PipelineGroupedVisitorTest.TestPipelineVisitorRevisitIPU");

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program,
                        /*options=*/
                        {{"debug.retainDebugInformation", "true"},
                         {"autoReport.all", "true"},
                         {"autoReport.directory", dir_manager.GetDirName()}});
  engine.enableExecutionProfiling();

  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  // Get the execution steps
  auto report = engine.getReport(/*reportExecution=*/true);
  std::vector<pva::ExecutionStep> steps(report.execution().steps().begin(),
                                        report.execution().steps().end());

  // Only consider the on tile execute steps
  auto is_on_tile_exec_pred = [](pva::ExecutionStep& step) -> bool {
    return step.program()->type() == pva::Program::Type::OnTileExecute;
  };
  auto itr =
      std::stable_partition(steps.begin(), steps.end(), is_on_tile_exec_pred);

  // And only consider the computation steps
  auto is_add_pred = [](pva::ExecutionStep& step) -> bool {
    return absl::StrContains(step.program()->name(), "add") &&
           !absl::StrContains(step.program()->name(), "Copy");
  };
  itr = std::stable_partition(steps.begin(), itr, is_add_pred);
  steps.erase(itr, steps.end());

  // Compute the total number of cycles that were overlapped.
  auto overlapped_cycles = [](int accum, pva::ExecutionStep& step) -> int {
    for (auto& ipu : step.ipus()) {
      accum += ipu.allCycles().from().max() - ipu.allCycles().from().min();
    }
    return accum;
  };
  // Compute the total number of cycles.
  auto cycles = [](int accum, pva::ExecutionStep& step) -> int {
    for (auto& ipu : step.ipus()) {
      accum += ipu.cycles();
    }
    return accum;
  };
  int total_overlapped_cycles =
      std::accumulate(steps.begin(), steps.end(), 0, overlapped_cycles);
  int total_cycles = std::accumulate(steps.begin(), steps.end(), 0, cycles);
  // Check we overlapped enough cycles. This value was determined empirically
  ASSERT_GT(((float)total_overlapped_cycles) / total_cycles, 0.49);
}

// Tests that poplar revisits IPUs in the expected order.
TEST_F(PipelineGroupedVisitorTest, TestPipelineVisitorRevisitIPUOrder) {
  const string& hlo_string = R"(
HloModule module

_stage_0 {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  temp_0 = f32[] constant(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_1 {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_2 {
  param_0 = f32[] parameter(0), sharding={maximal device=2}
  const_1 = f32[] constant(1), sharding={maximal device=2}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=2}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=2}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=2}}
}

_stage_2_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=1}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=1}}
}

_stage_1_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=0}
  const_1 = f32[] constant(1), sharding={maximal device=0}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  add_1 = f32[] add(param_0, const_1), sharding={maximal device=0}
  ROOT t = (f32[]) tuple(add_1), sharding={{maximal device=0}}
}

_stage_0_bw {
  param_0 = f32[] parameter(0), sharding={maximal device=1}
  const_1 = f32[] constant(1), sharding={maximal device=1}
  token_f = token[] custom-call(param_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=1}
  add_0 = f32[] add(param_0, const_1), sharding={maximal device=1}
  ROOT t = (f32[]) tuple(add_0), sharding={{maximal device=1}}
}

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

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

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(12), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(4, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 4);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
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
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 1 and 2
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 1},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.2"), 2},
      // Inter-ipu-copy between stage 3 and 4
      {entry_computation->GetInstructionWithName("ipu-inter-copy.3"), 3},
      // Inter-ipu-copy between stage 4 and 5
      {entry_computation->GetInstructionWithName("ipu-inter-copy.4"), 4},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 12);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 2, 1, 0, 1}, stage_assignments, {}, 3, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(12).ValueOrDie();

  // Build and run the graph
  poplar::Engine engine(*resources->main_graph, program,
                        /*options=*/{{"debug.retainDebugInformation", "true"}});
  engine.enableExecutionProfiling();
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  const std::string expected = R"(/print-tensor: 0
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.5: 5
)";

  ASSERT_EQ(expected, ss.str());
}

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
// Also make sure that aliasing is preserved by the FIFO.
TEST_F(PipelineGroupedVisitorTest,
       TestPipelineVisitorFifoValueBroadcastTuples) {
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
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[2]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[2] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[2] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={maximal device=0}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Grouped\"}}}"
}
)";
  auto device = CreateIpuModel(2, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(device, module.get(), false, 2);

  CustomOpReplacer replacer;
  EXPECT_TRUE(replacer.Run(module.get()).ValueOrDie());

  InterIpuCopyInserter inserter;
  EXPECT_TRUE(inserter.Run(module.get()).ValueOrDie());

  HloTrivialScheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());

  auto pipeline_call_computation = module->entry_computation();
  auto entry_computation =
      pipeline_call_computation->GetInstructionWithName("p")
          ->called_computations()[0];

  // Count the number of stages
  const auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  // Assign each instruction in the pipeline to a stage
  const absl::flat_hash_map<const HloInstruction*, int> stage_assignments = {
      {entry_computation->GetInstructionWithName("arg"), 0},
      {entry_computation->GetInstructionWithName("grad_acc"), 0},
      {entry_computation->GetInstructionWithName("a0"), 0},
      {entry_computation->GetInstructionWithName("gte_a"), 0},
      {entry_computation->GetInstructionWithName("b0"), 1},
      {entry_computation->GetInstructionWithName("gte_b"), 1},
      {entry_computation->GetInstructionWithName("c0"), 2},
      {entry_computation->GetInstructionWithName("gte_c"), 2},
      {entry_computation->GetInstructionWithName("d"), 3},
      // Inter-ipu-copy between stage 0 and 1
      {entry_computation->GetInstructionWithName("ipu-inter-copy"), 0},
      // Inter-ipu-copy between stage 2 and 3
      {entry_computation->GetInstructionWithName("ipu-inter-copy.1"), 2},
      // FIFO after stage 0
      {entry_computation->GetInstructionWithName("fifo"), 0},
  };

  auto placeholder = resources->main_graph->addVariable(poplar::FLOAT, {});
  resources->main_graph->setTileMapping(placeholder, 0);
  auto grad_acc_placeholder =
      resources->main_graph->addConstant(poplar::INT, {}, 8);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Grouped, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
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

  const std::string expected = R"(/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
)";

  // Check the output of the stage has aliases.
  ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("_stage_0")
                  .FindTensorByName("out", 0)
                  .containsAliases());
  // Check that the fifo has aliases.
  ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("pipeline")
                  .FindTensorByName("fifo", 0)
                  .containsAliases());
  ASSERT_EQ(expected, ss.str());
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
