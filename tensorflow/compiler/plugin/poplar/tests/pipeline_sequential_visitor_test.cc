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
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
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

using PipelineSeqVisitorTest = HloPoplarTestBase;

TEST_F(PipelineSeqVisitorTest, TestPipelineVisitorMergedCopies) {
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

  SequentialPipelineVisitor visitor(
      *resources->main_graph, pipeline, *resources,
      DeferredArgRBVectors{{TensorOrRemoteBuffer{p0}},
                           {TensorOrRemoteBuffer{p1}}},
      GetInplaceDescription(entry_computation->root_instruction()), "visitor");
  TF_EXPECT_OK(pipeline_comp->Accept(&visitor));

  // Get the pipeline program
  auto program =
      visitor.GetPipelineSequence(*resources->main_graph, 6).ValueOrDie();

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
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
