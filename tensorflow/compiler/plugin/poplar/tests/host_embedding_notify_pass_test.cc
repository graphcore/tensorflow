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

#include "tensorflow/compiler/plugin/poplar/driver/passes/host_embedding_notification.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using HostEmbeddingNotifyTest = HloTestBase;
// Check the pass doesn't break anything.
TEST_F(HostEmbeddingNotifyTest, NoChange) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  in2 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in0, in1, in2)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in1, in2, in3)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[] parameter(3)
  bcast = f32[1,4,4,2] broadcast(arg3), dimensions={}
  new_arg0 = f32[1,4,4,2] add(arg0, bcast)
  new_arg1 = f32[1,4,4,2] add(arg1, bcast)
  new_arg2 = f32[1,4,4,2] add(arg2, bcast)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(new_arg0, new_arg1, new_arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[] parameter(3)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(in1, in0, in4), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[] get-tuple-element(stage_0), index=2
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(stage_0_0, stage_0_1, stage_0_2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_3 = f32[] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2, stage_1_3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(gte0, gte1, gte2, in4)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[] parameter(3), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  CompilerAnnotations annotations(module.get());
  HostEmbeddingNotification prvo;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_FALSE(changed);
}

// Check that a notification is moved to the resource update.
TEST_F(HostEmbeddingNotifyTest, MoveNotify) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  in2 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in0, in1, in2)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in1, in2, in3)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  a = token[] custom-call(), custom_call_target="HostEmbeddingNotify", backend_config="{\"device_ordinal\":0,\"embedding_id\":\"foo\"}\n"
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[] parameter(3)
  bcast = f32[1,4,4,2] broadcast(arg3), dimensions={}
  new_arg0 = f32[1,4,4,2] add(arg0, bcast)
  new_arg1 = f32[1,4,4,2] add(arg1, bcast)
  new_arg2 = f32[1,4,4,2] add(arg2, bcast)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(new_arg0, new_arg1, new_arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[] parameter(3)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(in1, in0, in4), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[] get-tuple-element(stage_0), index=2
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(stage_0_0, stage_0_1, stage_0_2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_3 = f32[] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2, stage_1_3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(gte0, gte1, gte2, in4)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[] parameter(3), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  HostEmbeddingNotification prvo;
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);

  auto predicate = IsPoplarInstruction(PoplarOp::HostEmbeddingNotify);

  for (auto name : {"e", "pipeline", "stage_0_fwd", "stage_1_fwd",
                    "stage_0_bwd", "stage_1_bwd"}) {
    HloComputation* comp = module->GetComputationWithName(name);
    auto insts = comp->MakeInstructionPostOrder();

    EXPECT_EQ(0, absl::c_count_if(insts, predicate));
  }

  HloComputation* comp = module->GetComputationWithName("resource_update");
  auto insts = comp->MakeInstructionPostOrder();

  EXPECT_EQ(1, absl::c_count_if(insts, predicate));
}

// Check that only one notification is moved to the resource update and
// the duplicates are removed.
TEST_F(HostEmbeddingNotifyTest, MultiNotify) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  in2 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in0, in1, in2)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in1, in2, in3)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,2] parameter(1)
  a = token[] custom-call(), custom_call_target="HostEmbeddingNotify", backend_config="{\"device_ordinal\":0,\"embedding_id\":\"foo\"}\n"
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  a = token[] custom-call(), custom_call_target="HostEmbeddingNotify", backend_config="{\"device_ordinal\":0,\"embedding_id\":\"foo\"}\n"
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,2] parameter(2)
  arg3 = f32[] parameter(3)
  bcast = f32[1,4,4,2] broadcast(arg3), dimensions={}
  new_arg0 = f32[1,4,4,2] add(arg0, bcast)
  new_arg1 = f32[1,4,4,2] add(arg1, bcast)
  new_arg2 = f32[1,4,4,2] add(arg2, bcast)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(new_arg0, new_arg1, new_arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  in3 = f32[1,4,4,2] parameter(2)
  in4 = f32[] parameter(3)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(in1, in0, in4), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[] get-tuple-element(stage_0), index=2
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(stage_0_0, stage_0_1, stage_0_2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_3 = f32[] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2, stage_1_3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,2] get-tuple-element(call_ru), index=2
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(gte0, gte1, gte2, in4)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  e.in3 = f32[] parameter(3), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2], f32[]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  CompilerAnnotations annotations(module.get());
  HostEmbeddingNotification prvo;
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module.get()).ValueOrDie());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prvo.Run(module.get()));
  EXPECT_TRUE(changed);

  auto predicate = IsPoplarInstruction(PoplarOp::HostEmbeddingNotify);

  for (auto name : {"e", "pipeline", "stage_0_fwd", "stage_1_fwd",
                    "stage_0_bwd", "stage_1_bwd"}) {
    HloComputation* comp = module->GetComputationWithName(name);
    auto insts = comp->MakeInstructionPostOrder();

    EXPECT_EQ(0, absl::c_count_if(insts, predicate));
  }

  HloComputation* comp = module->GetComputationWithName("resource_update");
  auto insts = comp->MakeInstructionPostOrder();

  EXPECT_EQ(1, absl::c_count_if(insts, predicate));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
