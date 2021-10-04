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

class PipelineVerifierTest : public HloPoplarTestBase,
                             public ::testing::WithParamInterface<int> {};

// This tests that the print tensor statements get printed in the expected
// order, given a pipeline poplar control program.
TEST_P(PipelineVerifierTest, TestPipelineVerify) {
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
  ROOT p = (f32[]) call(arg, const_2), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"1\",\"schedule\":\"Interleaved\"}}}"
}

)";
  int count = GetParam();
  {
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
    auto stage_count = absl::c_count_if(
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
    const poplar::Tensor grad_acc_placeholder =
        resources->main_graph->addConstant(poplar::INT, {}, count);
    resources->main_graph->setTileMapping(placeholder, 0);
    resources->main_graph->setTileMapping(grad_acc_placeholder, 0);

    ParallelPipelineVisitor visitor(
        PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved,
        stage_count, {0, 1, 1, 0}, stage_assignments, {}, 2, *resources,
        DeferredArgRBVectors{{TensorOrRemoteBuffer{placeholder}},
                             {TensorOrRemoteBuffer{grad_acc_placeholder}}},
        GetInplaceDescription(entry_computation->root_instruction()),
        "visitor");

    TF_EXPECT_OK(entry_computation->Accept(&visitor));

    // Get verify program
    auto verify_program =
        visitor
            .VerifyPipelineArguments(
                pipeline_call_computation->GetInstructionWithName("arg.1"),
                grad_acc_placeholder, *(resources->main_graph))
            .ValueOrDie();
    // Get the pipeline program
    auto program = visitor.GetPipelineSequence(count).ValueOrDie();

    poplar::program::Sequence seq(verify_program, program);

    // Compile the graph
    poplar::Engine engine(*resources->main_graph, seq);

    // Run the program
    device.attach();
    engine.load(device);
    ASSERT_THROW(engine.run(0), poplar::application_runtime_error);
    device.detach();
  }
}

INSTANTIATE_TEST_CASE_P(VerifierTests, PipelineVerifierTest,
                        ::testing::Values(0, 1, 3));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
