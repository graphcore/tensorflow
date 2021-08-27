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
#include <pva/pva.hpp>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
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

using PConfig = PoplarBackendConfig::CallConfig::PipelineConfig;

struct TestParams {
  std::string hlo_string;
  std::vector<std::pair<std::string, int>> stage_assignments;
  int gradient_accumulation_count;
  std::vector<std::size_t> placeholder_shape;
  std::string expected;
  bool check_aliases;
  poplar::OptionFlags opts;
  std::vector<int> ipu_assigments;
  int num_v_graphs;
  PConfig::Schedule config;
  TestParams(std::string hlo_string,
             std::vector<std::pair<std::string, int>> stage_assignments,
             int gradient_accumulation_count,
             std::vector<std::size_t> placeholder_shape, std::string expected,
             bool check_aliases = false, poplar::OptionFlags opts = {},
             std::vector<int> ipu_assigments = {0, 1, 1, 0},
             int num_v_graphs = 2,
             PConfig::Schedule config = PConfig::Interleaved)
      : hlo_string(hlo_string),
        stage_assignments(stage_assignments),
        gradient_accumulation_count(gradient_accumulation_count),
        placeholder_shape(placeholder_shape),
        expected(expected),
        check_aliases(check_aliases),
        opts(opts),
        ipu_assigments(ipu_assigments),
        num_v_graphs(num_v_graphs),
        config(config) {}
};

class PipelineVisitorTestParam
    : public HloPoplarTestBase,
      public ::testing::WithParamInterface<TestParams> {};

using PipelineVisitorTest = HloPoplarTestBase;

// This tests that the print tensor statements get printed in the expected
// order, given a pipeline poplar control program.
TEST_P(PipelineVisitorTestParam, TestPipelineVisitor) {
  auto params = GetParam();
  auto device = CreateIpuModel(params.num_v_graphs, 4);

  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(params.hlo_string).ConsumeValueOrDie();
  auto resources =
      GetMockResources(device, module.get(), false, params.num_v_graphs);

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
  auto stage_count = absl::c_count_if(
      entry_computation->instructions(), [](const HloInstruction* hlo) {
        return hlo->opcode() == HloOpcode::kCall;
      });

  auto comps = module->MakeComputationPostOrder();
  auto bw_stage_count = absl::c_count_if(comps, [](const HloComputation* hlo) {
    return hlo->name().find("_bw") != std::string::npos;
  });
  // Assign each instruction in the pipeline to a stage
  absl::flat_hash_map<const HloInstruction*, int> stage_assignments;
  for (const auto& [name, stage] : params.stage_assignments) {
    stage_assignments.emplace(entry_computation->GetInstructionWithName(name),
                              stage);
  }

  auto placeholder = resources->main_graph->addVariable(
      poplar::FLOAT, params.placeholder_shape);
  auto grad_acc_placeholder = resources->main_graph->addConstant(
      poplar::INT, {}, params.gradient_accumulation_count);
  resources->main_graph->setTileMapping(placeholder, 0);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      params.config, stage_count, params.ipu_assigments, stage_assignments, {},
      bw_stage_count, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");

  // TF_EXPECT_OK(entry_computation->Accept(&visitor));
  auto status = entry_computation->Accept(&visitor);
  TF_EXPECT_OK(status);

  // Get verify program
  auto verify_program =
      visitor
          .VerifyPipelineArguments(
              pipeline_call_computation->GetInstructionWithName("const_2"),
              grad_acc_placeholder, *(resources->main_graph))
          .ValueOrDie();
  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(params.gradient_accumulation_count)
                     .ValueOrDie();

  // Compile the graph
  poplar::Engine engine(*resources->main_graph,
                        poplar::program::Sequence(verify_program, program),
                        params.opts);
  engine.enableExecutionProfiling();
  // Capture the engine output into a string stream.
  std::stringstream ss;
  engine.setPrintTensorStream(ss);

  // Run the program
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  if (params.check_aliases) {
    // Check the output of the stage has aliases.
    ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("_stage_0")
                    .FindTensorByName("out", 0)
                    .containsAliases());
    // Check that the fifo has aliases.
    ASSERT_TRUE(resources->tensor_maps.GetTensorMapForComputation("pipeline")
                    .FindTensorByName("fifo", 0)
                    .containsAliases());
  }

  ASSERT_EQ(params.expected, ss.str());
}

const string& hlo_0 = R"(
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
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}

)";

const std::vector<std::pair<std::string, int>> assignments_0 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    {"ipu-inter-copy", 0},
    {"ipu-inter-copy.1", 2}};

const string& expected_0 = R"(/print-tensor: 1
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.1: 2
/print-tensor: 1
/print-tensor.2: 4
/print-tensor.1: 2
/print-tensor.3: 5
/print-tensor.2: 4
/print-tensor: 1
/print-tensor.3: 5
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor.3: 5
)";

TestParams TestPipelineVisitorOrder(hlo_0, assignments_0, 4, {}, expected_0);

// This tests that the output value has the expected value, given a pipeline
// poplar control program.
const string& hlo_1 = R"(
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
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}
)";

const std::vector<std::pair<std::string, int>> assignments_1 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-IPU-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-IPU-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2}};

const string& expected_1 = R"(/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
)";

TestParams TestPipelineVisitorValue(hlo_1, assignments_1, 6, {}, expected_1);

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo.
const string& hlo_2 = R"(
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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}
)";

// Assign each instruction in the pipeline to a stage
const std::vector<std::pair<std::string, int>> assignments_2 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
    // FIFO after stage 0
    {"fifo", 0},
};

const string& expected_2 = R"(/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
)";

TestParams TestPipelineVisitorFifoValue{
    hlo_2, assignments_2, 4, {}, expected_2};

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
const string& hlo_3 = R"(
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

pipeline {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = ((f32[2], f32[4], f32[2], f32[2])) call(arg), to_apply=_stage_0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = (f32[2], f32[4], f32[2], f32[2]) get-tuple-element(a0), index=0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"isInplace\":true}"
  a1 = (f32[2], f32[4], f32[2], f32[2]) custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":1}", sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(8), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}


)";

// Assign each instruction in the pipeline to a stage
const std::vector<std::pair<std::string, int>> assignments_3 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
    // FIFO after stage 0
    {"fifo", 0},
};

const string& expected_3 = R"(/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
)";

TestParams TestPipelineVisitorFifoValueTuples{
    hlo_3, assignments_3, 8, {2}, expected_3};

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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}


  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}
  c0 = (f32[]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[] get-tuple-element(c0), index=0, sharding={maximal device=1}
  ROOT d = (f32[]) call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
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
      resources->main_graph->addConstant(poplar::INT, {}, 4);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Create unique reporting directory to avoid tests overwriting
  // eachother's reports.
  TemporaryDirManager dir_manager(
      "PipelineVisitorTest.TestPipelineVisitorFifoOverlap");

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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={maximal device=0}
  b0 = (f32[]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  b1 = f32[] custom-call(gte_b), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":2}", sharding={maximal device=1}
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
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
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
      resources->main_graph->addConstant(poplar::INT, {}, 4);
  resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

  ParallelPipelineVisitor visitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved, stage_count,
      {0, 1, 2, 1, 0, 1}, stage_assignments, {}, 3, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                           {TensorOrRemoteBuffer{grad_acc_placeholder}}},
      HloInstructionDescription(entry_computation->root_instruction()),
      "visitor");
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Get the pipeline program
  auto program = visitor.GetPipelineSequence(6).ValueOrDie();

  // Create unique reporting directory to avoid tests overwriting
  // eachother's reports.
  TemporaryDirManager dir_manager(
      "PipelineVisitorTest.TestPipelineVisitorRevisitIPU");

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
const string& hlo_5 = R"(
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
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}
)";

const std::vector<std::pair<std::string, int>> assignments_5 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d0", 3},
    {"gte_d", 3},
    {"e0", 4},
    {"gte_e", 4},
    {"d", 5},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 1 and 2
    {"ipu-inter-copy.1", 1},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.2", 2},
    // Inter-ipu-copy between stage 3 and 4
    {"ipu-inter-copy.3", 3},
    // Inter-ipu-copy between stage 4 and 5
    {"ipu-inter-copy.4", 4},
};

const string& expected_5 = R"(/print-tensor: 0
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor: 0
/print-tensor.5: 5
/print-tensor.1: 1
/print-tensor: 0
/print-tensor.2: 2
/print-tensor.1: 1
/print-tensor.3: 3
/print-tensor.2: 2
/print-tensor.4: 4
/print-tensor.3: 3
/print-tensor.5: 5
/print-tensor.4: 4
/print-tensor.5: 5
)";

TestParams TestPipelineVisitorRevisitIPUOrder{
    hlo_5,
    assignments_5,
    8,
    {},
    expected_5,
    false,
    {{"debug.retainDebugInformation", "true"}},
    {0, 1, 2, 1, 0, 1},
    4};

// This tests that the output value has the expected value, given a pipeline
// poplar control program with a fifo and tuples.
// Also make sure that aliasing is preserved by the FIFO.
const string& hlo_4 = R"(
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

pipeline {
  arg = f32[] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = (f32[2]) call(arg), to_apply=_stage_0, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = f32[2] get-tuple-element(a0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  a1 = f32[2] custom-call(gte_a), custom_call_target="Fifo", backend_config="{\"offload\":0,\"depth\":1}", sharding={maximal device=0}
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, a1), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}
)";

const std::vector<std::pair<std::string, int>> assignments_4 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
    // FIFO after stage 0
    {"fifo", 0},
};

const string& expected_4 = R"(/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
/print-tensor: {206,212}
)";

TestParams TestPipelineVisitorFifoValueBroadcastTuples{
    hlo_4, assignments_4, 8, {}, expected_4, true};

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
  auto device = CreateIpuModel(2, 4);

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

  ParallelPipelineVisitor visitor(
      pipeline, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{p0}},
                           {TensorOrRemoteBuffer{p1}}},
      HloInstructionDescription(entry_computation->root_instruction()),
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

  const std::string expected = R"(/print-tensor: 3
/print-tensor: 3
/print-tensor: 3
/print-tensor: 3
/print-tensor: 3
/print-tensor: 3
)";

  ASSERT_EQ(expected, ss.str());
}

//-----------------------------------------------------------------
// Sequential tests
//-----------------------------------------------------------------

const string& hlo_6 = R"(
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
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Sequential\"}}}"
}
)";

const string& expected_6 = R"(/print-tensor: 1
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor.3: 5
/print-tensor: 1
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor.3: 5
/print-tensor: 1
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor.3: 5
/print-tensor: 1
/print-tensor.1: 2
/print-tensor.2: 4
/print-tensor.3: 5
)";

const std::vector<std::pair<std::string, int>> assignments_6 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-IPU-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-IPU-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
};

// This tests that the print tensor statements get printed in the expected
// order, given a pipeline poplar control program.
TestParams TestPipelineVisitorOrderSequential(hlo_6, assignments_6, 4, {},
                                              expected_6, false, {},
                                              {0, 1, 1, 0}, 2,
                                              PConfig::Sequential);

const string& hlo_7 = R"(
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
  const_2 = s32[] constant(6), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Sequential\"}}}"
}
)";

const std::vector<std::pair<std::string, int>> assignments_7 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-IPU-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-IPU-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
};

const string& expected_7 = R"(/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
/print-tensor: 4
)";

// This tests that the output value has the expected value, given a pipeline
// poplar control program.
TestParams TestPipelineVisitorValueSequential(hlo_7, assignments_7, 6, {},
                                              expected_7, false, {},
                                              {0, 1, 1, 0}, 2,
                                              PConfig::Sequential);

const string& hlo_8 = R"(
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
  tuple = (f32[2], f32[4], f32[2], f32[2]) parameter(1), sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}
  add_1 = f32[2] get-tuple-element(tuple), index=2, sharding={maximal device=0}

  add_0 = f32[2] add(param, add_1), sharding={maximal device=0}
  token_f = token[] custom-call(add_0), custom_call_target="PrintTensor", backend_config="{}", sharding={maximal device=0}
  ROOT t = () tuple(), sharding={{maximal device=0}}
}

pipeline {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  grad_acc = s32[] parameter(1), sharding={maximal device=0}

  a0 = ((f32[2], f32[4], f32[2], f32[2])) call(arg), to_apply=_stage_0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  gte_a = (f32[2], f32[4], f32[2], f32[2]) get-tuple-element(a0), index=0, sharding={{maximal device=0},{maximal device=0},{maximal device=0},{maximal device=0}}, backend_config="{\"isInplace\":true}"
  b0 = (f32[2]) call(gte_a), to_apply=_stage_1, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_b = f32[2] get-tuple-element(b0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  c0 = (f32[2]) call(gte_b), to_apply=_stage_1_bw, sharding={{maximal device=1}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  gte_c = f32[2] get-tuple-element(c0), index=0, sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  ROOT d = () call(gte_c, gte_a), to_apply=_stage_0_bw, sharding={{maximal device=0}}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
}

ENTRY main {
  arg = f32[2] parameter(0), sharding={maximal device=0}
  const_2 = s32[] constant(8), sharding={maximal device=0}
  ROOT p = () call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Sequential\"}}}"
}

)";

const std::vector<std::pair<std::string, int>> assignments_8 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d", 3},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.1", 2},
};

const string& expected_8 = R"(/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
/print-tensor: {306,6012}
)";

// This tests that the output value has the expected value, given a pipeline
// poplar control program with tuples.
TestParams TestPipelineVisitorValueTuplesSequential(hlo_8, assignments_8, 8,
                                                    {2}, expected_8, false, {},
                                                    {0, 1, 1, 0}, 2,
                                                    PConfig::Sequential);

const string& hlo_9 = R"(
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
  const_2 = s32[] constant(4), sharding={maximal device=0}
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Sequential\"}}}"
}
)";

const std::vector<std::pair<std::string, int>> assignments_9 = {
    {"arg", 0},
    {"grad_acc", 0},
    {"a0", 0},
    {"gte_a", 0},
    {"b0", 1},
    {"gte_b", 1},
    {"c0", 2},
    {"gte_c", 2},
    {"d0", 3},
    {"gte_d", 3},
    {"e0", 4},
    {"gte_e", 4},
    {"d", 5},
    // Inter-ipu-copy between stage 0 and 1
    {"ipu-inter-copy", 0},
    // Inter-ipu-copy between stage 1 and 2
    {"ipu-inter-copy.1", 1},
    // Inter-ipu-copy between stage 2 and 3
    {"ipu-inter-copy.2", 2},
    // Inter-ipu-copy between stage 3 and 4
    {"ipu-inter-copy.3", 3},
    // Inter-ipu-copy between stage 4 and 5
    {"ipu-inter-copy.4", 4},
};

const string& expected_9 = R"(/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
/print-tensor: 0
/print-tensor.1: 1
/print-tensor.2: 2
/print-tensor.3: 3
/print-tensor.4: 4
/print-tensor.5: 5
)";

TestParams TestPipelineVisitorRevisitIPUOrderSequential{
    hlo_9,
    assignments_9,
    8,
    {},
    expected_9,
    false,
    {{"debug.retainDebugInformation", "true"}},
    {0, 1, 2, 1, 0, 1},
    4,
    PConfig::Sequential};

INSTANTIATE_TEST_CASE_P(
    PipelineVisitorTestParameter, PipelineVisitorTestParam,
    ::testing::Values(TestPipelineVisitorOrder, TestPipelineVisitorValue,
                      TestPipelineVisitorFifoValue,
                      TestPipelineVisitorFifoValueTuples,
                      TestPipelineVisitorFifoValueBroadcastTuples,
                      TestPipelineVisitorRevisitIPUOrder,
                      TestPipelineVisitorOrderSequential,
                      TestPipelineVisitorValueSequential,
                      TestPipelineVisitorValueTuplesSequential,
                      TestPipelineVisitorRevisitIPUOrderSequential));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
