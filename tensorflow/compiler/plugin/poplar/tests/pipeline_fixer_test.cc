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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fixer.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using PipelineFixerTest = HloTestBase;

bool IsPipelineOk(const PipelineStages& stages) {
  for (int64_t fwd_stage_id = 0; fwd_stage_id != stages.forward.size();
       ++fwd_stage_id) {
    HloInstruction* stage = stages.forward[fwd_stage_id];
    for (HloInstruction* operand : stage->operands()) {
      switch (operand->opcode()) {
        case HloOpcode::kParameter:
          break;
        case HloOpcode::kGetTupleElement: {
          // Make sure GTE is on the previous stage.
          if (fwd_stage_id > 0 &&
              operand->operand(0) == stages.forward.at(fwd_stage_id - 1)) {
            break;
          }
        }
        default:
          return false;
      }
    }
  }

  for (int64_t bwd_stage_id = 0; bwd_stage_id != stages.backward.size();
       ++bwd_stage_id) {
    HloInstruction* stage = stages.backward[bwd_stage_id];
    for (HloInstruction* operand : stage->operands()) {
      switch (operand->opcode()) {
        case HloOpcode::kParameter:
          break;
        case HloOpcode::kGetTupleElement: {
          // Make sure GTE is on the previous stage or on the corresponding fwd
          // stage.
          if ((bwd_stage_id < stages.backward.size() - 1 &&
               operand->operand(0) == stages.backward.at(bwd_stage_id + 1)) ||
              operand->operand(0) == stages.forward.at(bwd_stage_id)) {
            break;
          }
        }
        default:
          return false;
      }
    }
  }
  return true;
}

TEST_F(PipelineFixerTest, TestInferenceAllOk) {
  std::string hlo = R"(
HloModule main

stage_1 {
  after-all.9 = token[] after-all()
  infeed.10 = ((f32[2,4,4,2]{3,2,1,0}, f32[2,4,4,2]{3,2,1,0}), token[]) infeed(after-all.9), infeed_config="\010\001\022\005feed2\"\002\001\001(\001"
  get-tuple-element.11 = (f32[2,4,4,2]{3,2,1,0}, f32[2,4,4,2]{3,2,1,0}) get-tuple-element(infeed.10), index=0
  get-tuple-element.13 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(get-tuple-element.11), index=1
  get-tuple-element.12 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(get-tuple-element.11), index=0
  arg1.7 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  convolution.14 = f32[2,4,4,2]{3,2,1,0} convolution(get-tuple-element.12, arg1.7), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg2.8 = f32[2]{0} parameter(2)
  broadcast.15 = f32[2,4,4,2]{3,2,1,0} broadcast(arg2.8), dimensions={3}
  add.16 = f32[2,4,4,2]{3,2,1,0} add(convolution.14, broadcast.15)
  add.17 = f32[2,4,4,2]{3,2,1,0} add(get-tuple-element.13, add.16)
  arg0.6 = f32[] parameter(0)
  ROOT tuple.22 = (f32[2,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(add.17, arg0.6, arg1.7, arg2.8)
}

Sum-reduction.23 {
  x.24 = f32[] parameter(0)
  y.25 = f32[] parameter(1)
  ROOT add.26 = f32[] add(x.24, y.25)
}

stage_2 {
  arg0.28 = f32[2,4,4,2]{3,2,1,0} parameter(0)
  convert.30 = f32[2,4,4,2]{3,2,1,0} convert(arg0.28)
  constant.31 = f32[] constant(0)
  convert.32 = f32[] convert(constant.31)
  reduce.33 = f32[] reduce(convert.30, convert.32), dimensions={0,1,2,3}, to_apply=Sum-reduction.23
  convert.34 = f32[] convert(reduce.33)
  arg1.29 = f32[] parameter(1)
  add.35 = f32[] add(convert.34, arg1.29)
  after-all.36 = token[] after-all()
  outfeed.37 = token[] outfeed(add.35, after-all.36), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
  ROOT tuple.38 = () tuple()
}

pipeline_wrapper {
  arg0.40 = f32[] parameter(0)
  arg1.41 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg2.42 = f32[2]{0} parameter(2)
  call.43 = (f32[2,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.40, arg1.41, arg2.42), to_apply=stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  get-tuple-element.44 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(call.43), index=0
  get-tuple-element.45 = f32[] get-tuple-element(call.43), index=1
  call.46 = () call(get-tuple-element.44, get-tuple-element.45), to_apply=stage_2, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  ROOT tuple.51 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg1.41, arg2.42)
}

pipeline {
  arg0.40 = f32[] parameter(0)
  arg1.41 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg2.42 = f32[2]{0} parameter(2)
  call = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.40, arg1.41, arg2.42), to_apply=pipeline_wrapper
  get-tuple-element.44 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=0
  get-tuple-element.45 = f32[2]{0} get-tuple-element(call), index=1
  ROOT tuple.51 = (f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0.40, get-tuple-element.44, get-tuple-element.45)
}

ENTRY main {
  arg0.1 = f32[] parameter(0), parameter_replication={false}
  reshape.4 = f32[] reshape(arg0.1)
  arg2.3 = f32[1,1,2,2]{3,2,1,0} parameter(2), parameter_replication={false}
  arg1.2 = f32[2]{0} parameter(1), parameter_replication={false}
  call.52 = (f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(reshape.4, arg2.3, arg1.2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT tuple.53 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(IsPipelineOk(stages));
}

TEST_F(PipelineFixerTest, TestTrainingAllOk) {
  std::string hlo = R"(
HloModule Pipeline

add_float {
  x.49 = f32[] parameter(0)
  y.50 = f32[] parameter(1)
  ROOT add.51 = f32[] add(x.49, y.50)
}

stage_0 {
  arg0.14 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg2.16 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  convolution.18 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.14, arg2.16), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg3.17 = f32[2]{0} parameter(3)
  broadcast.19 = f32[1,4,4,2]{3,2,1,0} broadcast(arg3.17), dimensions={3}
  add.20 = f32[1,4,4,2]{3,2,1,0} add(convolution.18, broadcast.19)
  arg1.15 = f32[] parameter(1)
  ROOT tuple.25 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(add.20, arg1.15, arg0.14, arg2.16, arg2.16, arg0.14, arg2.16, arg3.17)
}

stage_1 {
  arg0.27 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg2.29 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  convolution.33 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.27, arg2.29), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg3.30 = f32[2]{0} parameter(3)
  broadcast.34 = f32[1,4,4,2]{3,2,1,0} broadcast(arg3.30), dimensions={3}
  add.35 = f32[1,4,4,2]{3,2,1,0} add(convolution.33, broadcast.34)
  arg4.31 = f32[1,1,2,2]{3,2,1,0} parameter(4)
  convolution.36 = f32[1,4,4,2]{3,2,1,0} convolution(add.35, arg4.31), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg5.32 = f32[2]{0} parameter(5)
  broadcast.37 = f32[1,4,4,2]{3,2,1,0} broadcast(arg5.32), dimensions={3}
  add.38 = f32[1,4,4,2]{3,2,1,0} add(convolution.36, broadcast.37)
  arg1.28 = f32[] parameter(1)
  ROOT tuple.47 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(add.38, arg1.28, add.35, arg4.31, arg4.31, add.35, arg0.27, arg2.29, arg2.29, arg0.27, arg2.29, arg3.30, arg4.31, arg5.32)
}

stage_2 {
  arg0.0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg2.0 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  convolution = f32[1,4,4,2]{3,2,1,0} convolution(arg0.0, arg2.0), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg3.0 = f32[2]{0} parameter(3)
  broadcast = f32[1,4,4,2]{3,2,1,0} broadcast(arg3.0), dimensions={3}
  add = f32[1,4,4,2]{3,2,1,0} add(convolution, broadcast)
  custom-call.2 = f32[1,3,3,2]{3,2,1,0} custom-call(add), custom_call_target="MaxPool", window={size=1x2x2x1}
  convert = f32[1,3,3,2]{3,2,1,0} convert(custom-call.2)
  constant = f32[] constant(0)
  convert.1 = f32[] convert(constant)
  reduce = f32[] reduce(convert, convert.1), dimensions={0,1,2,3}, to_apply=add_float
  convert.2 = f32[] convert(reduce)
  after-all = token[] after-all()
  outfeed = token[] outfeed(convert.2, after-all), outfeed_config="\010\001\022\005feed1\"\001\001(\001"
  arg1.0 = f32[] parameter(1)
  ROOT tuple.1 = (f32[], f32[], f32[1,3,3,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,3,3,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(convert.2, arg1.0, custom-call.2, add, custom-call.2, arg0.0, arg2.0, arg2.0, arg0.0, arg2.0, arg3.0)
}

stage_1_bwd {
  arg0.9 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg3.9 = f32[1,1,2,2]{3,2,1,0} parameter(3)
  reverse.11 = f32[1,1,2,2]{3,2,1,0} reverse(arg3.9), dimensions={0,1}
  convolution.24 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.9, reverse.11), window={size=1x1}, dim_labels=b01f_01oi->b01f
  arg5.10 = f32[1,1,2,2]{3,2,1,0} parameter(5)
  reverse.12 = f32[1,1,2,2]{3,2,1,0} reverse(arg5.10), dimensions={0,1}
  convolution.25 = f32[1,4,4,2]{3,2,1,0} convolution(convolution.24, reverse.12), window={size=1x1}, dim_labels=b01f_01oi->b01f
  arg1.9 = f32[] parameter(1)
  arg4.9 = f32[1,4,4,2]{3,2,1,0} parameter(4)
  convolution.26 = f32[1,1,2,2]{3,2,1,0} convolution(arg4.9, convolution.24), window={size=4x4}, dim_labels=f01b_i01o->01bf
  convert.25 = f32[1,4,4,2]{3,2,1,0} convert(convolution.24)
  constant.15 = f32[] constant(0)
  reduce.12 = f32[2]{0} reduce(convert.25, constant.15), dimensions={0,1,2}, to_apply=add_float
  convert.26 = f32[2]{0} convert(reduce.12)
  arg2.9 = f32[1,4,4,2]{3,2,1,0} parameter(2)
  convolution.27 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.9, arg0.9), window={size=4x4}, dim_labels=f01b_i01o->01bf
  convert.27 = f32[1,4,4,2]{3,2,1,0} convert(arg0.9)
  constant.16 = f32[] constant(0)
  reduce.13 = f32[2]{0} reduce(convert.27, constant.16), dimensions={0,1,2}, to_apply=add_float
  convert.28 = f32[2]{0} convert(reduce.13)
  arg6.4 = f32[1,1,2,2]{3,2,1,0} parameter(7)
  get-tuple-element.23 = f32[] parameter(6)
  broadcast.23 = f32[1,1,2,2]{3,2,1,0} broadcast(get-tuple-element.23), dimensions={}
  multiply.15 = f32[1,1,2,2]{3,2,1,0} multiply(broadcast.23, convolution.27)
  subtract.15 = f32[1,1,2,2]{3,2,1,0} subtract(arg6.4, multiply.15)
  arg7.3 = f32[2]{0} parameter(9)
  get-tuple-element.25 = f32[] parameter(8)
  broadcast.24 = f32[2]{0} broadcast(get-tuple-element.25), dimensions={}
  multiply.16 = f32[2]{0} multiply(broadcast.24, convert.28)
  subtract.16 = f32[2]{0} subtract(arg7.3, multiply.16)
  arg5.11 = f32[2]{0} parameter(11)
  get-tuple-element.26 = f32[] parameter(10)
  broadcast.25 = f32[2]{0} broadcast(get-tuple-element.26), dimensions={}
  multiply.17 = f32[2]{0} multiply(broadcast.25, convert.26)
  subtract.17 = f32[2]{0} subtract(arg5.11, multiply.17)
  arg4.10 = f32[1,1,2,2]{3,2,1,0} parameter(13)
  get-tuple-element.27 = f32[] parameter(12)
  broadcast.26 = f32[1,1,2,2]{3,2,1,0} broadcast(get-tuple-element.27), dimensions={}
  multiply.18 = f32[1,1,2,2]{3,2,1,0} multiply(broadcast.26, convolution.26)
  subtract.18 = f32[1,1,2,2]{3,2,1,0} subtract(arg4.10, multiply.18)
  get-tuple-element.28 = f32[] parameter(14)
  ROOT tuple.33 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[]) tuple(convolution.25, arg1.9, convolution.26, convert.26, convolution.27, convert.28, subtract.15, subtract.16, subtract.17, subtract.18, get-tuple-element.28)
}

stage_0_bwd {
  arg0.11 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg3.12 = f32[1,1,2,2]{3,2,1,0} parameter(3)
  reverse.14 = f32[1,1,2,2]{3,2,1,0} reverse(arg3.12), dimensions={0,1}
  convolution.30 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.11, reverse.14), window={size=1x1}, dim_labels=b01f_01oi->b01f
  arg1.11 = f32[] parameter(1)
  arg2.11 = f32[1,4,4,2]{3,2,1,0} parameter(2)
  convolution.31 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.11, arg0.11), window={size=4x4}, dim_labels=f01b_i01o->01bf
  convert.31 = f32[1,4,4,2]{3,2,1,0} convert(arg0.11)
  constant.18 = f32[] constant(0)
  reduce.15 = f32[2]{0} reduce(convert.31, constant.18), dimensions={0,1,2}, to_apply=add_float
  convert.32 = f32[2]{0} convert(reduce.15)
  arg3.13 = f32[2]{0} parameter(5)
  get-tuple-element.33 = f32[] parameter(4)
  broadcast.28 = f32[2]{0} broadcast(get-tuple-element.33), dimensions={}
  multiply.20 = f32[2]{0} multiply(broadcast.28, convert.32)
  subtract.20 = f32[2]{0} subtract(arg3.13, multiply.20)
  arg2.12 = f32[1,1,2,2]{3,2,1,0} parameter(7)
  get-tuple-element.34 = f32[] parameter(6)
  broadcast.29 = f32[1,1,2,2]{3,2,1,0} broadcast(get-tuple-element.34), dimensions={}
  multiply.21 = f32[1,1,2,2]{3,2,1,0} multiply(broadcast.29, convolution.31)
  subtract.21 = f32[1,1,2,2]{3,2,1,0} subtract(arg2.12, multiply.21)
  ROOT tuple.38 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(convolution.30, arg1.11, convolution.31, convert.32, subtract.20, subtract.21)
}

stage_2_bwd {
  arg3.15 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg2.14 = f32[1,3,3,2]{3,2,1,0} parameter(1)
  constant.23 = f32[] constant(0)
  broadcast.35 = f32[1,3,3,2]{3,2,1,0} broadcast(constant.23), dimensions={}
  constant.24 = f32[] constant(1)
  reshape.8 = f32[1,1,1,1]{3,2,1,0} reshape(constant.24)
  reshape.9 = f32[1]{0} reshape(reshape.8)
  broadcast.36 = f32[1,3,3,2]{3,2,1,0} broadcast(reshape.9), dimensions={0}
  add.5 = f32[1,3,3,2]{3,2,1,0} add(broadcast.35, broadcast.36)
  custom-call.7 = f32[1,4,4,2]{3,2,1,0} custom-call(arg3.15, arg2.14, add.5), custom_call_target="MaxPoolGrad", window={size=1x2x2x1}
  arg5.13 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  reverse.16 = f32[1,1,2,2]{3,2,1,0} reverse(arg5.13), dimensions={0,1}
  convolution.35 = f32[1,4,4,2]{3,2,1,0} convolution(custom-call.7, reverse.16), window={size=1x1}, dim_labels=b01f_01oi->b01f
  constant.25 = f32[] constant(0)
  arg4.12 = f32[1,4,4,2]{3,2,1,0} parameter(3)
  convolution.37 = f32[1,1,2,2]{3,2,1,0} convolution(arg4.12, custom-call.7), window={size=4x4}, dim_labels=f01b_i01o->01bf
  convert.35 = f32[1,4,4,2]{3,2,1,0} convert(custom-call.7)
  constant.26 = f32[] constant(0)
  reduce.17 = f32[2]{0} reduce(convert.35, constant.26), dimensions={0,1,2}, to_apply=add_float  convert.36 = f32[2]{0} convert(reduce.17)
  arg9.4 = f32[2]{0} parameter(4)
  get-tuple-element.184.clone.17 = f32[] parameter(5)
  broadcast.38 = f32[2]{0} broadcast(get-tuple-element.184.clone.17), dimensions={}
  multiply.24 = f32[2]{0} multiply(broadcast.38, convert.36)
  subtract.24 = f32[2]{0} subtract(arg9.4, multiply.24)
  arg8.3 = f32[1,1,2,2]{3,2,1,0} parameter(6)
  get-tuple-element.184.clone.18 = f32[] parameter(7)
  broadcast.39 = f32[1,1,2,2]{3,2,1,0} broadcast(get-tuple-element.184.clone.18), dimensions={}
  multiply.25 = f32[1,1,2,2]{3,2,1,0} multiply(broadcast.39, convolution.37)
  subtract.25 = f32[1,1,2,2]{3,2,1,0} subtract(arg8.3, multiply.25)
  get-tuple-element.184.clone.19 = f32[] parameter(8)
  ROOT tuple.53 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[]) tuple(convolution.35, constant.25, convolution.37, convert.36, subtract.24, subtract.25, get-tuple-element.184.clone.19)
}

resource_update {
  arg0 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg1 = f32[2]{0} parameter(1)
  arg2 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  arg3 = f32[2]{0} parameter(3)
  arg4 = f32[1,1,2,2]{3,2,1,0} parameter(4)
  arg5 = f32[2]{0} parameter(5)
  arg6 = f32[1,1,2,2]{3,2,1,0} parameter(6)
  arg7 = f32[2]{0} parameter(7)
  ROOT t = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
}

pipeline_wrapper {
  arg0.154 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg1.155 = f32[] parameter(1)
  arg2.156 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  arg3.157 = f32[2]{0} parameter(3)
  call.164 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.154, arg1.155, arg2.156, arg3.157), to_apply=stage_0, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  get-tuple-element.165 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.164), index=0
  get-tuple-element.166 = f32[] get-tuple-element(call.164), index=1
  arg4.158 = f32[1,1,2,2]{3,2,1,0} parameter(4)
  arg5.159 = f32[2]{0} parameter(5)
  arg6.160 = f32[1,1,2,2]{3,2,1,0} parameter(6)
  arg7.161 = f32[2]{0} parameter(7)
  call.171 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.165, get-tuple-element.166, arg4.158, arg5.159, arg6.160, arg7.161), to_apply=stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  get-tuple-element.172 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.171), index=0
  get-tuple-element.173 = f32[] get-tuple-element(call.171), index=1
  arg8.162 = f32[1,1,2,2]{3,2,1,0} parameter(8)
  arg9.163 = f32[2]{0} parameter(9)
  call = (f32[], f32[], f32[1,3,3,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,3,3,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.172, get-tuple-element.173, arg8.162, arg9.163), to_apply=stage_2, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=2}
  get-tuple-element.186 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call), index=3
  get-tuple-element.185 = f32[1,3,3,2]{3,2,1,0} get-tuple-element(call), index=2
  get-tuple-element.189 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=6
  get-tuple-element.188 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call), index=5
  get-tuple-element.184.clone = f32[] get-tuple-element(call), index=1
  get-tuple-element.184.clone.1 = f32[] get-tuple-element(call), index=1
  get-tuple-element.184.clone.5 = f32[] get-tuple-element(call), index=1
  call.12 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[]) call(get-tuple-element.186, get-tuple-element.185, get-tuple-element.189, get-tuple-element.188, arg9.163, get-tuple-element.184.clone, arg8.162, get-tuple-element.184.clone.1, get-tuple-element.184.clone.5), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  get-tuple-element.201 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.12), index=0
  get-tuple-element.202 = f32[] get-tuple-element(call.12), index=1
  get-tuple-element.174 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.171), index=2
  get-tuple-element.175 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.171), index=3
  get-tuple-element.178 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.171), index=6
  get-tuple-element.179 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.171), index=7
  get-tuple-element.3 = f32[] get-tuple-element(call.12), index=6
  get-tuple-element.4 = f32[] get-tuple-element(call.12), index=6
  get-tuple-element.2 = f32[] get-tuple-element(call.12), index=6
  get-tuple-element.5 = f32[] get-tuple-element(call.12), index=6
  get-tuple-element.6 = f32[] get-tuple-element(call.12), index=6
  call.8 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[]) call(get-tuple-element.201, get-tuple-element.202, get-tuple-element.174, get-tuple-element.175, get-tuple-element.178, get-tuple-element.179, get-tuple-element.3, arg6.160, get-tuple-element.4, arg7.161, get-tuple-element.2, arg5.159, get-tuple-element.5, arg4.158, get-tuple-element.6), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  get-tuple-element.214 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.8), index=0
  get-tuple-element.215 = f32[] get-tuple-element(call.8), index=1
  get-tuple-element.167 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.164), index=2
  get-tuple-element.168 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.164), index=3
  get-tuple-element.29 = f32[] get-tuple-element(call.8), index=10
  get-tuple-element.30 = f32[] get-tuple-element(call.8), index=10
  call.10 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) call(get-tuple-element.214, get-tuple-element.215, get-tuple-element.167, get-tuple-element.168, get-tuple-element.29, arg3.157, get-tuple-element.30, arg2.156), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  get-tuple-element.35 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.10), index=5
  get-tuple-element.32 = f32[2]{0} get-tuple-element(call.10), index=4
  get-tuple-element.21 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.8), index=9
  get-tuple-element.16 = f32[2]{0} get-tuple-element(call.8), index=8
  get-tuple-element.9 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.8), index=6
  get-tuple-element.12 = f32[2]{0} get-tuple-element(call.8), index=7
  get-tuple-element.1 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.12), index=5
  get-tuple-element = f32[2]{0} get-tuple-element(call.12), index=4
  call_ru = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.35, get-tuple-element.32, get-tuple-element.21, get-tuple-element.16, get-tuple-element.9, get-tuple-element.12, get-tuple-element.1, get-tuple-element), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,1,2,2] get-tuple-element(call_ru), index=0
  gte1 = f32[2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,1,2,2] get-tuple-element(call_ru), index=2
  gte3 = f32[2] get-tuple-element(call_ru), index=3
  gte4 = f32[1,1,2,2] get-tuple-element(call_ru), index=4
  gte5 = f32[2] get-tuple-element(call_ru), index=5
  gte6 = f32[1,1,2,2] get-tuple-element(call_ru), index=6
  gte7 = f32[2] get-tuple-element(call_ru), index=7
  ROOT tuple.265 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(gte0, gte1, gte2, gte3, gte4, gte5, gte6, gte7)
}

pipeline {
  arg0.154 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg1.155 = f32[] parameter(1)
  arg2.156 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  arg3.157 = f32[2]{0} parameter(3)
  arg4.158 = f32[1,1,2,2]{3,2,1,0} parameter(4)
  arg5.159 = f32[2]{0} parameter(5)
  arg6.160 = f32[1,1,2,2]{3,2,1,0} parameter(6)
  arg7.161 = f32[2]{0} parameter(7)
  arg8.162 = f32[1,1,2,2]{3,2,1,0} parameter(8)
  arg9.163 = f32[2]{0} parameter(9)
  call = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.154, arg1.155, arg2.156, arg3.157, arg4.158, arg5.159, arg6.160, arg7.161, arg8.162, arg9.163), to_apply=pipeline_wrapper
  gte0 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=0
  gte1 = f32[2]{0} get-tuple-element(call), index=1
  gte2 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=2
  gte3 = f32[2]{0} get-tuple-element(call), index=3
  gte4 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=4
  gte5 = f32[2]{0} get-tuple-element(call), index=5
  gte6 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=6
  gte7 = f32[2]{0} get-tuple-element(call), index=7
  ROOT tuple.265 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0.154, arg1.155, gte0, gte1, gte2, gte3, gte4, gte5, gte6, gte7)
}

ENTRY Pipeline {
  arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0), parameter_replication={false}
  reshape.11 = f32[1,4,4,2]{3,2,1,0} reshape(arg0.1)
  arg1.2 = f32[] parameter(1), parameter_replication={false}
  reshape.12 = f32[] reshape(arg1.2)
  arg4.5 = f32[1,1,2,2]{3,2,1,0} parameter(4), parameter_replication={false}
  arg2.3 = f32[2]{0} parameter(2), parameter_replication={false}
  arg7.8 = f32[1,1,2,2]{3,2,1,0} parameter(7), parameter_replication={false}
  arg6.7 = f32[2]{0} parameter(6), parameter_replication={false}
  arg9.10 = f32[1,1,2,2]{3,2,1,0} parameter(9), parameter_replication={false}
  arg8.9 = f32[2]{0} parameter(8), parameter_replication={false}
  arg5.6 = f32[1,1,2,2]{3,2,1,0} parameter(5), parameter_replication={false}
  arg3.4 = f32[2]{0} parameter(3), parameter_replication={false}
  call.266 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(reshape.11, reshape.12, arg4.5, arg2.3, arg7.8, arg6.7, arg9.10, arg8.9, arg5.6, arg3.4), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  get-tuple-element.268 = f32[2]{0} get-tuple-element(call.266), index=3
  tuple.275 = (f32[2]{0}) tuple(get-tuple-element.268)
  get-tuple-element.276 = f32[2]{0} get-tuple-element(tuple.275), index=0
  get-tuple-element.274 = f32[2]{0} get-tuple-element(call.266), index=9
  tuple.277 = (f32[2]{0}) tuple(get-tuple-element.274)
  get-tuple-element.278 = f32[2]{0} get-tuple-element(tuple.277), index=0
  get-tuple-element.267 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.266), index=2
  tuple.279 = (f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.267)
  get-tuple-element.280 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(tuple.279), index=0
  get-tuple-element.273 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.266), index=8
  tuple.281 = (f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.273)
  get-tuple-element.282 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(tuple.281), index=0
  get-tuple-element.270 = f32[2]{0} get-tuple-element(call.266), index=5
  tuple.283 = (f32[2]{0}) tuple(get-tuple-element.270)
  get-tuple-element.284 = f32[2]{0} get-tuple-element(tuple.283), index=0
  get-tuple-element.269 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.266), index=4
  tuple.285 = (f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.269)
  get-tuple-element.286 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(tuple.285), index=0
  get-tuple-element.272 = f32[2]{0} get-tuple-element(call.266), index=7
  tuple.287 = (f32[2]{0}) tuple(get-tuple-element.272)
  get-tuple-element.288 = f32[2]{0} get-tuple-element(tuple.287), index=0
  get-tuple-element.271 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.266), index=6
  tuple.289 = (f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.271)
  get-tuple-element.290 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(tuple.289), index=0
  ROOT tuple.291 = (f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.276, get-tuple-element.278, get-tuple-element.280, get-tuple-element.282, get-tuple-element.284, get-tuple-element.286, get-tuple-element.288, get-tuple-element.290)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(IsPipelineOk(stages));
}

TEST_F(PipelineFixerTest, RequiresLowering) {
  std::string hlo = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_feed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_feed), index=0
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  stage_0_fwd_lr = f32[] parameter(1)
  ROOT stage_0_fwd_tuple = (f32[], f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_0_fwd_lr, stage_0_fwd_acts_0, stage_0_fwd_input)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  stage_1_fwd_lr = f32[] parameter(2)
  ROOT stage_1_fwd_tuple = (f32[], f32[], f32[1,4,4,2]) tuple(stage_1_fwd_lr, stage_1_fwd_reduce, stage_1_fwd_acts_0)
}

stage_1_bwd {
  stage_1_bwd_reduce = f32[] parameter(0)
  stage_1_bwd_bcast1 = f32[1,4,4,2] broadcast(stage_1_bwd_reduce), dimensions={}
  stage_1_bwd_acts_0 = f32[1,4,4,2] parameter(1)
  stage_1_bwd_acts_0_bwd = f32[1,4,4,2] add(stage_1_bwd_acts_0, stage_1_bwd_bcast1)
  ROOT stage_1_bwd_tuple = (f32[1,4,4,2]) tuple(stage_1_bwd_acts_0_bwd)
}

stage_0_bwd {
  stage_0_bwd_acts_0_bwd = f32[1,4,4,2] parameter(0)
  stage_0_bwd_input = f32[1,4,4,2] parameter(1)
  stage_0_bwd_input_bwd = f32[1,4,4,2] add(stage_0_bwd_input, stage_0_bwd_acts_0_bwd)
  ROOT stage_0_bwd_tuple = (f32[1,4,4,2]) tuple(stage_0_bwd_input_bwd)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1)
}

pipeline_wrapper {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_lr = f32[] parameter(2)
  pipeline_stage_0 = (f32[], f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_lr), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_lr = f32[] get-tuple-element(pipeline_stage_0), index=0
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_input = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=2
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1, pipeline_stage_0_lr), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_lr = f32[] get-tuple-element(pipeline_stage_1), index=0
  pipeline_reduce = f32[] get-tuple-element(pipeline_stage_1), index=1
  pipeline_acts_0_local = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=2
  pipeline_stage_1_bwd = (f32[1,4,4,2]) call(pipeline_reduce, pipeline_acts_0_local), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  pipeline_acts_0_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_1_bwd), index=0
  pipeline_lr_bcast1 = f32[1,4,4,2] broadcast(pipeline_stage_1_lr), dimensions={}
  pipeline_weights1_update = f32[1,4,4,2] multiply(pipeline_acts_0_bwd, pipeline_lr_bcast1)
  pipeline_weights1_apply = f32[1,4,4,2] subtract(pipeline_weights1, pipeline_weights1_update)
  pipeline_stage_0_bwd = (f32[1,4,4,2]) call(pipeline_acts_0_bwd, pipeline_input), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  pipeline_input_bwd = f32[1,4,4,2] get-tuple-element(pipeline_stage_0_bwd), index=0
  pipeline_lr_bcast2 = f32[1,4,4,2] broadcast(pipeline_stage_1_lr), dimensions={}
  pipeline_weights0_update = f32[1,4,4,2] multiply(pipeline_input_bwd, pipeline_lr_bcast2)
  pipeline_weights0_apply = f32[1,4,4,2] subtract(pipeline_weights0, pipeline_weights0_update)
  call_ru = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0_apply, pipeline_weights1_apply), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
  }

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_lr = f32[] parameter(2)
  call = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_lr), to_apply=pipeline_wrapper
  gte0 = f32[1,4,4,2] get-tuple-element(call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(gte0, gte1, pipeline_lr)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.lr = f32[] parameter(2), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[]) call(e.weights0, e.weights1, e.lr), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(e.call), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(e.call), index=1
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(IsPipelineOk(stages));
}

TEST_F(PipelineFixerTest, TestNoPipelines) {
  std::string hlo = R"(
HloModule main

ENTRY main {
  arg0.1 = f32[] parameter(0), parameter_replication={false}
  reshape.4 = f32[] reshape(arg0.1)
  arg2.3 = f32[1,1,2,2]{3,2,1,0} parameter(2), parameter_replication={false}
  arg1.2 = f32[2]{0} parameter(1), parameter_replication={false}
  ROOT tuple.53 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineFixerTest, TestMultiplePipelines) {
  std::string hlo = R"(
HloModule main

stage_1 {
  after-all.9 = token[] after-all()
  infeed.10 = ((f32[2,4,4,2]{3,2,1,0}, f32[2,4,4,2]{3,2,1,0}), token[]) infeed(after-all.9), infeed_config="\010\001\022\005feed2\"\002\001\001(\001"
  get-tuple-element.11 = (f32[2,4,4,2]{3,2,1,0}, f32[2,4,4,2]{3,2,1,0}) get-tuple-element(infeed.10), index=0
  get-tuple-element.13 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(get-tuple-element.11), index=1
  get-tuple-element.12 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(get-tuple-element.11), index=0
  arg1.7 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  convolution.14 = f32[2,4,4,2]{3,2,1,0} convolution(get-tuple-element.12, arg1.7), window={size=1x1}, dim_labels=b01f_01io->b01f
  arg2.8 = f32[2]{0} parameter(2)
  broadcast.15 = f32[2,4,4,2]{3,2,1,0} broadcast(arg2.8), dimensions={3}
  add.16 = f32[2,4,4,2]{3,2,1,0} add(convolution.14, broadcast.15)
  add.17 = f32[2,4,4,2]{3,2,1,0} add(get-tuple-element.13, add.16)
  arg0.6 = f32[] parameter(0)
  ROOT tuple.22 = (f32[2,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(add.17, arg0.6, arg1.7, arg2.8)
}

Sum-reduction.23 {
  x.24 = f32[] parameter(0)
  y.25 = f32[] parameter(1)
  ROOT add.26 = f32[] add(x.24, y.25)
}

stage_2 {
  arg0.28 = f32[2,4,4,2]{3,2,1,0} parameter(0)
  convert.30 = f32[2,4,4,2]{3,2,1,0} convert(arg0.28)
  constant.31 = f32[] constant(0)
  convert.32 = f32[] convert(constant.31)
  reduce.33 = f32[] reduce(convert.30, convert.32), dimensions={0,1,2,3}, to_apply=Sum-reduction.23
  convert.34 = f32[] convert(reduce.33)
  arg1.29 = f32[] parameter(1)
  add.35 = f32[] add(convert.34, arg1.29)
  after-all.36 = token[] after-all()
  outfeed.37 = token[] outfeed(add.35, after-all.36), outfeed_config="\010\001\022\005feed3\"\001\001(\001"
  ROOT tuple.38 = () tuple()
}

pipeline {
  arg0.40 = f32[] parameter(0)
  arg1.41 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg2.42 = f32[2]{0} parameter(2)
  call.43 = (f32[2,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.40, arg1.41, arg2.42), to_apply=stage_1, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  get-tuple-element.44 = f32[2,4,4,2]{3,2,1,0} get-tuple-element(call.43), index=0
  get-tuple-element.45 = f32[] get-tuple-element(call.43), index=1
  call.46 = () call(get-tuple-element.44, get-tuple-element.45), to_apply=stage_2, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  ROOT tuple.51 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg1.41, arg2.42)
}

ENTRY main {
  arg0.1 = f32[] parameter(0), parameter_replication={false}
  reshape.4 = f32[] reshape(arg0.1)
  arg2.3 = f32[1,1,2,2]{3,2,1,0} parameter(2), parameter_replication={false}
  arg1.2 = f32[2]{0} parameter(1), parameter_replication={false}
  call.52 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(reshape.4, arg2.3, arg1.2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  call.51 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(reshape.4, arg2.3, arg1.2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT tuple.53 = () tuple()
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFixer fixer;
  EXPECT_FALSE(fixer.Run(module.get()).ok());
}

TEST_F(PipelineFixerTest, TestParameterModifiedByConst) {
  std::string hlo = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_infeed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_infeed), index=0
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_acts_0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  stage_1_fwd_t = token[] after-all()
  stage_1_fwd_outfeed = () outfeed(stage_1_fwd_reduce, stage_1_fwd_t)
  ROOT stage_1_fwd_tuple = () tuple()
}

pipeline_wrapper {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  one = f32[] constant(1)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = () call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  bcast = f32[1,4,4,2] broadcast(one), dimensions={}
  add = f32[1,4,4,2] add(bcast, pipeline_weights1)
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(add)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  call = (f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=pipeline_wrapper
  gte = f32[1,4,4,2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0, gte)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT gte = f32[1,4,4,2] get-tuple-element(e.call), index=1
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  EXPECT_TRUE(
      Match(stages.forward[1]->to_apply()->root_instruction(),
            m::Tuple(m::Add(m::Broadcast(m::Constant()), m::Parameter(1)))));

  EXPECT_TRUE(IsPipelineOk(stages));
}

TEST_F(PipelineFixerTest, TestClusterWithMultipleOutputs) {
  std::string hlo = R"(
HloModule top

add_float {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(f32[] x, f32[] y)
}

stage_0_fwd {
  stage_0_fwd_t = token[] after-all()
  stage_0_fwd_infeed = (f32[1,4,4,2], token[]) infeed(stage_0_fwd_t)
  stage_0_fwd_input = f32[1,4,4,2] get-tuple-element(stage_0_fwd_infeed), index=0
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_acts_0 = f32[1,4,4,2] add(stage_0_fwd_input, stage_0_fwd_weights0)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2]) tuple(stage_0_fwd_acts_0)
}

stage_1_fwd {
  stage_1_fwd_acts_0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights1 = f32[1,4,4,2] parameter(1)
  stage_1_fwd_acts_1 = f32[1,4,4,2] add(stage_1_fwd_acts_0, stage_1_fwd_weights1)
  stage_1_fwd_zero = f32[] constant(0)
  one = f32[] constant(1)
  bcast = f32[1,4,4,2] broadcast(one), dimensions={}
  add = f32[1,4,4,2] add(bcast, stage_1_fwd_weights1)
  stage_1_fwd_reduce = f32[] reduce(stage_1_fwd_acts_1, stage_1_fwd_zero), dimensions={0,1,2,3}, to_apply=add_float
  ROOT stage_1_fwd_tuple = (f32[], f32[1,4,4,2]) tuple(stage_1_fwd_reduce, add)
}

pipeline_wrapper {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_stage_0 = (f32[1,4,4,2]) call(pipeline_weights0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_acts_0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_stage_1 = (f32[], f32[1,4,4,2]) call(pipeline_acts_0, pipeline_weights1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  gte0 = f32[] get-tuple-element(pipeline_stage_1), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  outfeed_tuple = (f32[], f32[1,4,4,2]) tuple(gte0, gte1)
  outfeed_token = token[] after-all()
  outfeed = () outfeed(outfeed_tuple, outfeed_token)
  ROOT pipeline_tuple = (f32[1,4,4,2]) tuple(gte1)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  call = (f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1), to_apply=pipeline_wrapper
  gte = f32[1,4,4,2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_weights0, gte)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT gte = f32[1,4,4,2] get-tuple-element(e.call), index=1
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));

  EXPECT_TRUE(IsPipelineOk(stages));
}

TEST_F(PipelineFixerTest, TestSplitElementwiseOps) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_p0 = f32[2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[2]) tuple(stage_0_fwd_p0)
}

stage_1_fwd {
  stage_1_fwd_p0 = f32[2] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[2]) tuple(stage_1_fwd_p0)
}

stage_2_fwd {
  stage_2_fwd_p0 = f32[2] parameter(0)
  ROOT stage_2_fwd_tuple = (f32[2]) tuple(stage_2_fwd_p0)
}

stage_2_bwd {
  stage_2_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_2_bwd_p0)
  ROOT stage_2_bwd_tuple = (f32[2], f32[2]) tuple(stage_2_bwd_p0, l)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_1_bwd_p0)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, l)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  ROOT stage_0_bwd_tuple = (f32[2]) tuple(stage_0_bwd_p0)
}

resource_update {
  p0 = f32[2] parameter(0)
  ROOT tuple = (f32[2]) tuple(p0)
}

pipeline_wrapper {
  p0 = f32[2] parameter(0)
  fwd_stage_0 = (f32[2]) call(p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  fwd_stage_0.0 = f32[2] get-tuple-element(fwd_stage_0), index=0
  fwd_stage_1 = (f32[2]) call(fwd_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  fwd_stage_1.0 = f32[2] get-tuple-element(fwd_stage_1), index=0
  fwd_stage_2 = (f32[2]) call(fwd_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  fwd_stage_2.0 = f32[2] get-tuple-element(fwd_stage_2), index=0
  bwd_stage_2 = (f32[2], f32[2]) call(fwd_stage_2.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  bwd_stage_2.0 = f32[2] get-tuple-element(bwd_stage_2), index=0
  bwd_stage_2.1 = f32[2] get-tuple-element(bwd_stage_2), index=1
  bwd_stage_1 = (f32[2], f32[2]) call(bwd_stage_2.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  bwd_stage_1.0 = f32[2] get-tuple-element(bwd_stage_1), index=0
  bwd_stage_1.1 = f32[2] get-tuple-element(bwd_stage_1), index=1
  bwd_stage_0 = (f32[2]) call(bwd_stage_1.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  bwd_stage_0.0 = f32[2] get-tuple-element(bwd_stage_0), index=0

  add_grads_partial = f32[2] add(bwd_stage_1.1, bwd_stage_0.0)
  add_grads = f32[2] add(add_grads_partial, bwd_stage_2.1)
  c = f32[2] constant({10, 2})
  normalized_grads = f32[2] multiply(add_grads, c)
  ru = (f32[2]) call(normalized_grads), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru.0 = f32[2] get-tuple-element(ru), index=0

  ROOT pipeline_tuple = (f32[2]) tuple(ru.0)
}

pipeline {
  p0 = f32[2] parameter(0)
  call = (f32[2]) call(p0), to_apply=pipeline_wrapper
  gte = f32[2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(gte)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT gte = f32[2] get-tuple-element(e.call), index=0
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(IsPipelineOk(stages));

  // Note how the add has been split to occur in individual stages.
  EXPECT_TRUE(
      Match(stages.backward[2]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)), m::Constant(),
                     m::Multiply(m::Constant(), m::Log(m::Parameter(0))))));

  EXPECT_TRUE(
      Match(stages.backward[1]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)),
                     m::Multiply(m::Parameter(1), m::Log(m::Parameter(0))),
                     m::Parameter(2), m::Parameter(3))));

  EXPECT_TRUE(Match(
      stages.backward[0]->to_apply()->root_instruction(),
      m::Tuple(m::Parameter(0),
               m::Add(m::Add(m::Parameter(2),
                             m::Multiply(m::Parameter(1), m::Parameter(0))),
                      m::Parameter(3)))));
}

TEST_F(PipelineFixerTest, TestSplitElementwiseOpsDifferentShards) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_p0 = f32[2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[2]) tuple(stage_0_fwd_p0)
}

stage_1_fwd {
  stage_1_fwd_p0 = f32[2] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[2]) tuple(stage_1_fwd_p0)
}

stage_2_fwd {
  stage_2_fwd_p0 = f32[2] parameter(0)
  ROOT stage_2_fwd_tuple = (f32[2]) tuple(stage_2_fwd_p0)
}

stage_2_bwd {
  stage_2_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_2_bwd_p0)
  ROOT stage_2_bwd_tuple = (f32[2], f32[2]) tuple(stage_2_bwd_p0, l)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_1_bwd_p0)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, l)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  ROOT stage_0_bwd_tuple = (f32[2]) tuple(stage_0_bwd_p0)
}

resource_update {
  p0 = f32[2] parameter(0)
  ROOT tuple = (f32[2]) tuple(p0)
}

pipeline_wrapper {
  p0 = f32[2] parameter(0)
  fwd_stage_0 = (f32[2]) call(p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  fwd_stage_0.0 = f32[2] get-tuple-element(fwd_stage_0), index=0
  fwd_stage_1 = (f32[2]) call(fwd_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  fwd_stage_1.0 = f32[2] get-tuple-element(fwd_stage_1), index=0
  fwd_stage_2 = (f32[2]) call(fwd_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=2}
  fwd_stage_2.0 = f32[2] get-tuple-element(fwd_stage_2), index=0
  bwd_stage_2 = (f32[2], f32[2]) call(fwd_stage_2.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  bwd_stage_2.0 = f32[2] get-tuple-element(bwd_stage_2), index=0
  bwd_stage_2.1 = f32[2] get-tuple-element(bwd_stage_2), index=1
  bwd_stage_1 = (f32[2], f32[2]) call(bwd_stage_2.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  bwd_stage_1.0 = f32[2] get-tuple-element(bwd_stage_1), index=0
  bwd_stage_1.1 = f32[2] get-tuple-element(bwd_stage_1), index=1
  bwd_stage_0 = (f32[2]) call(bwd_stage_1.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  bwd_stage_0.0 = f32[2] get-tuple-element(bwd_stage_0), index=0

  add_grads_partial = f32[2] add(bwd_stage_1.1, bwd_stage_0.0)
  add_grads = f32[2] add(add_grads_partial, bwd_stage_2.1)
  c = f32[2] constant({10, 2})
  normalized_grads = f32[2] multiply(add_grads, c)
  ru = (f32[2]) call(normalized_grads), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru.0 = f32[2] get-tuple-element(ru), index=0

  ROOT pipeline_tuple = (f32[2]) tuple(ru.0)
}

pipeline {
  p0 = f32[2] parameter(0)
  call = (f32[2]) call(p0), to_apply=pipeline_wrapper
  gte = f32[2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(gte)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT gte = f32[2] get-tuple-element(e.call), index=0
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(IsPipelineOk(stages));

  // Note how the add has not been split.
  EXPECT_TRUE(Match(stages.backward[2]->to_apply()->root_instruction(),
                    m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)))));

  EXPECT_TRUE(Match(
      stages.backward[1]->to_apply()->root_instruction(),
      m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)), m::Parameter(1))));

  EXPECT_TRUE(Match(
      stages.backward[0]->to_apply()->root_instruction(),
      m::Tuple(m::Parameter(0),
               m::Multiply(m::Add(m::Add(m::Parameter(1), m::Parameter(0)),
                                  m::Parameter(2)),
                           m::Constant()))));
}

TEST_F(PipelineFixerTest, TestFixingConstantGradients) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_p0 = f32[2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[2]) tuple(stage_0_fwd_p0)
}

stage_1_fwd {
  stage_1_fwd_p0 = f32[2] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[2]) tuple(stage_1_fwd_p0)
}

stage_2_fwd {
  stage_2_fwd_p0 = f32[2] parameter(0)
  ROOT stage_2_fwd_tuple = (f32[2]) tuple(stage_2_fwd_p0)
}

stage_2_bwd {
  stage_2_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_2_bwd_p0)
  ROOT stage_2_bwd_tuple = (f32[2], f32[2]) tuple(stage_2_bwd_p0, l)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_1_bwd_p0)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, l)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  ROOT stage_0_bwd_tuple = (f32[2]) tuple(stage_0_bwd_p0)
}

resource_update {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  ROOT tuple = (f32[2]) tuple(p0)
}

pipeline_wrapper {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  p2 = s32[] parameter(2)

  fwd_stage_0 = (f32[2]) call(p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  fwd_stage_0.0 = f32[2] get-tuple-element(fwd_stage_0), index=0
  fwd_stage_1 = (f32[2]) call(fwd_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  fwd_stage_1.0 = f32[2] get-tuple-element(fwd_stage_1), index=0
  fwd_stage_2 = (f32[2]) call(fwd_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  fwd_stage_2.0 = f32[2] get-tuple-element(fwd_stage_2), index=0
  bwd_stage_2 = (f32[2], f32[2]) call(fwd_stage_2.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  bwd_stage_2.0 = f32[2] get-tuple-element(bwd_stage_2), index=0
  bwd_stage_2.1 = f32[2] get-tuple-element(bwd_stage_2), index=1
  bwd_stage_1 = (f32[2], f32[2]) call(bwd_stage_2.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  bwd_stage_1.0 = f32[2] get-tuple-element(bwd_stage_1), index=0
  bwd_stage_1.1 = f32[2] get-tuple-element(bwd_stage_1), index=1
  bwd_stage_0 = (f32[2]) call(bwd_stage_1.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  bwd_stage_0.0 = f32[2] get-tuple-element(bwd_stage_0), index=0

  add_grads_partial = f32[2] add(bwd_stage_1.1, bwd_stage_0.0)
  add_grads = f32[2] add(add_grads_partial, bwd_stage_2.1)
  c = f32[2] constant({10, 2})
  normalized_grads = f32[2] multiply(add_grads, c)

  c2 = f32[2] constant({10, 2})
  other_update = f32[2] add(p1, c2)

  acc-scale = f32[] constant(1)

  create = f32[2] custom-call(other_update), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  add = f32[2] custom-call(create, p1, acc-scale), custom_call_target="GradientAccumulatorAddWithScale", backend_config="{}"
  sink = f32[2] custom-call(add), custom_call_target="GradientAccumulatorSink", backend_config="{}"


  ru = (f32[2]) call(normalized_grads, sink), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru.0 = f32[2] get-tuple-element(ru), index=0

  ROOT pipeline_tuple = (f32[2]) tuple(ru.0)
}

pipeline {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  p2 = s32[] parameter(2)

  call = (f32[2]) call(p0, p1, p2), to_apply=pipeline_wrapper
  gte = f32[2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(gte)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  e.weights1 = f32[2] parameter(1), parameter_replication={false}
  e.c0 = s32[] constant(3)
  e.call = (f32[2]) call(e.weights0, e.weights1, e.c0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\", \"pipelineConfig\":{\"schedule\":\"Grouped\",\"batch_serialization_iterations\":5,\"offload_gradient_accumulation_buffers\":\"1\", \"gradient_accumulation_index\":\"2\"}}}"
  ROOT gte = f32[2] get-tuple-element(e.call), index=0
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline_wrapper");

  HloInstruction* pipeline_op =
      module->entry_computation()->GetInstructionWithName("e.call");

  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  ASSERT_TRUE(custom_op_replaced);
  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.TestFixConstantGradients(
                                            pipeline_op, pipeline_computation));
  ASSERT_TRUE(changed);

  auto ru = pipeline_computation->GetInstructionWithName("ru");
  ASSERT_EQ(ru->operand(1)->name(), "multiply.1");

  auto scale_fac = ru->operand(1)->operand(0)->operand(0)->operand(0);
  ASSERT_EQ(scale_fac->operand(0)->name(), "constant");
  ASSERT_EQ(scale_fac->operand(1)->name(), "p2");
}

TEST_F(PipelineFixerTest, TestSplitGradientAccumulation) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_p0 = f32[2] parameter(0)
  ROOT stage_0_fwd_tuple = (f32[2]) tuple(stage_0_fwd_p0)
}

stage_1_fwd {
  stage_1_fwd_p0 = f32[2] parameter(0)
  ROOT stage_1_fwd_tuple = (f32[2]) tuple(stage_1_fwd_p0)
}

stage_2_fwd {
  stage_2_fwd_p0 = f32[2] parameter(0)
  ROOT stage_2_fwd_tuple = (f32[2]) tuple(stage_2_fwd_p0)
}

stage_2_bwd {
  stage_2_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_2_bwd_p0)
  ROOT stage_2_bwd_tuple = (f32[2], f32[2]) tuple(stage_2_bwd_p0, l)
}

stage_1_bwd {
  stage_1_bwd_p0 = f32[2] parameter(0)
  l = f32[2] log(stage_1_bwd_p0)
  ROOT stage_1_bwd_tuple = (f32[2], f32[2]) tuple(stage_1_bwd_p0, l)
}

stage_0_bwd {
  stage_0_bwd_p0 = f32[2] parameter(0)
  ROOT stage_0_bwd_tuple = (f32[2]) tuple(stage_0_bwd_p0)
}

resource_update {
  p0 = f32[2] parameter(0)
  ROOT tuple = (f32[2]) tuple(p0)
}

pipeline_wrapper {
  p0 = f32[2] parameter(0)
  fwd_stage_0 = (f32[2]) call(p0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  fwd_stage_0.0 = f32[2] get-tuple-element(fwd_stage_0), index=0
  fwd_stage_1 = (f32[2]) call(fwd_stage_0.0), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  fwd_stage_1.0 = f32[2] get-tuple-element(fwd_stage_1), index=0
  fwd_stage_2 = (f32[2]) call(fwd_stage_1.0), to_apply=stage_2_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={maximal device=0}
  fwd_stage_2.0 = f32[2] get-tuple-element(fwd_stage_2), index=0
  bwd_stage_2 = (f32[2], f32[2]) call(fwd_stage_2.0), to_apply=stage_2_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}"
  bwd_stage_2.0 = f32[2] get-tuple-element(bwd_stage_2), index=0
  bwd_stage_2.1 = f32[2] get-tuple-element(bwd_stage_2), index=1
  bwd_stage_1 = (f32[2], f32[2]) call(bwd_stage_2.0), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
  bwd_stage_1.0 = f32[2] get-tuple-element(bwd_stage_1), index=0
  bwd_stage_1.1 = f32[2] get-tuple-element(bwd_stage_1), index=1
  bwd_stage_0 = (f32[2]) call(bwd_stage_1.0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}"
  bwd_stage_0.0 = f32[2] get-tuple-element(bwd_stage_0), index=0

  add_grads_partial = f32[2] add(bwd_stage_1.1, bwd_stage_0.0)
  add_grads = f32[2] add(add_grads_partial, bwd_stage_2.1)

  acc_scale = f32[] constant(1)
  create = f32[2] custom-call(p0), custom_call_target="GradientAccumulatorCreate", backend_config="{}"
  add = f32[2] custom-call(create, add_grads, acc_scale), custom_call_target="GradientAccumulatorAddWithScale", backend_config="{}"
  sink = f32[2] custom-call(add), custom_call_target="GradientAccumulatorSink", backend_config="{}"
  
  ru = (f32[2]) call(sink), to_apply=resource_update, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  ru.0 = f32[2] get-tuple-element(ru), index=0

  ROOT pipeline_tuple = (f32[2]) tuple(ru.0)
}

pipeline {
  p0 = f32[2] parameter(0)
  call = (f32[2]) call(p0), to_apply=pipeline_wrapper
  gte = f32[2] get-tuple-element(call), index=0
  ROOT pipeline_tuple = (f32[2]) tuple(gte)
}

ENTRY e {
  e.weights0 = f32[2] parameter(0), parameter_replication={false}
  e.call = (f32[2]) call(e.weights0), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  ROOT gte = f32[2] get-tuple-element(e.call), index=0
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  ASSERT_TRUE(custom_op_replaced);

  PipelineFixer fixer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fixer.Run(module.get()));
  EXPECT_TRUE(changed);
  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");

  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  EXPECT_TRUE(stages.resource_update);
  const HloInstruction* resource_update = *stages.resource_update;
  const HloInstruction* gradient_accumulator_sink = resource_update->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(
      gradient_accumulator_sink));
  ASSERT_EQ(gradient_accumulator_sink->operand_count(), 3);

  EXPECT_TRUE(Match(
      gradient_accumulator_sink,
      m::CustomCall(m::GetTupleElement(m::Op().Is(stages.backward[2]), 2),
                    m::GetTupleElement(m::Op().Is(stages.backward[1]), 2),
                    m::GetTupleElement(m::Op().Is(stages.backward[0]), 1))));

  EXPECT_TRUE(
      Match(stages.backward[2]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)),
                     m::CustomCall(m::Parameter(1), m::Log(m::Parameter(0)),
                                   m::ConstantScalar(1.f)))));

  EXPECT_TRUE(
      Match(stages.backward[1]->to_apply()->root_instruction(),
            m::Tuple(m::Parameter(0), m::Log(m::Parameter(0)),
                     m::CustomCall(m::Parameter(1), m::Log(m::Parameter(0)),
                                   m::ConstantScalar(1.f)))));

  EXPECT_TRUE(Match(
      stages.backward[0]->to_apply()->root_instruction(),
      m::Tuple(m::Parameter(0), m::CustomCall(m::Parameter(1), m::Parameter(0),
                                              m::ConstantScalar(1.f)))));
}

TEST_F(PipelineFixerTest, TestIsRunningMeanAccumulatorScale) {
  std::string hlo = R"(
HloModule top

ENTRY e {
  counter = s32[] custom-call(), custom_call_target="ExecutionCounter", backend_config="{\"lower_into_pipeline_stage\":true}"
  counter_f32 = f32[] convert(counter)
  one = f32[] constant(1)
  counter_plus_one = f32[] add(counter_f32, one)
  inv_counter_plus_one = f32[] divide(one, counter_plus_one)
  running_mean_scale_valid = f32[] multiply(counter_f32, inv_counter_plus_one)

  running_mean_scale_invalid = f32[] multiply(counter_f32, counter_plus_one)
  ROOT t = () tuple()
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  ASSERT_TRUE(custom_op_replaced);

  HloInstruction* running_mean_scale_valid =
      FindInstruction(module.get(), "running_mean_scale_valid");
  HloInstruction* running_mean_scale_invalid =
      FindInstruction(module.get(), "running_mean_scale_invalid");

  EXPECT_TRUE(pipeline_fixer_util::IsRunningMeanAccumulatorScale(
      running_mean_scale_valid));
  EXPECT_FALSE(pipeline_fixer_util::IsRunningMeanAccumulatorScale(
      running_mean_scale_invalid));
}

TEST_F(PipelineFixerTest, TestIsRunningMeanGradient) {
  std::string hlo = R"(
HloModule top

ENTRY e {
  p0 = f16[2] parameter(0)
  p1 = f32[2] parameter(1)

  counter = s32[] custom-call(), custom_call_target="ExecutionCounter", backend_config="{\"lower_into_pipeline_stage\":true}"
  counter_f32 = f32[] convert(counter)
  one = f32[] constant(1)
  counter_plus_one = f32[] add(counter_f32, one)
  scale = f32[] divide(one, counter_plus_one)

  convert = f16[] convert(scale)
  b_scale0 = f16[2] broadcast(convert), dimensions={}
  grad0 = f16[2] multiply(p0, b_scale0)

  b_scale1 = f32[2] broadcast(scale), dimensions={}
  grad1 = f32[2] multiply(p1, b_scale1)

  grad_invalid = f32[] add(counter_f32, counter_plus_one)
  ROOT t = () tuple()
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module.get()));
  ASSERT_TRUE(custom_op_replaced);

  HloInstruction* grad0 = FindInstruction(module.get(), "grad0");
  HloInstruction* grad1 = FindInstruction(module.get(), "grad1");
  HloInstruction* grad_invalid = FindInstruction(module.get(), "grad_invalid");

  EXPECT_TRUE(pipeline_fixer_util::IsRunningMeanGradient(grad0));
  EXPECT_TRUE(pipeline_fixer_util::IsRunningMeanGradient(grad1));
  EXPECT_FALSE(pipeline_fixer_util::IsRunningMeanGradient(grad_invalid));
}

TEST_F(PipelineFixerTest, TestGradientsNeedRescaling) {
  using pipeline_config = PoplarBackendConfig::CallConfig::PipelineConfig;
  {
    TF_ASSERT_OK_AND_ASSIGN(bool needs_rescaling,
                            pipeline_fixer_util::GradientsNeedRescaling(
                                pipeline_config::Grouped, 2, 2));
    EXPECT_FALSE(needs_rescaling);
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(bool needs_rescaling,
                            pipeline_fixer_util::GradientsNeedRescaling(
                                pipeline_config::Grouped, 2, 1));
    EXPECT_TRUE(needs_rescaling);
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(bool needs_rescaling,
                            pipeline_fixer_util::GradientsNeedRescaling(
                                pipeline_config::Sequential, 2, 1));
    EXPECT_FALSE(needs_rescaling);
  }
  {
    EXPECT_FALSE(pipeline_fixer_util::GradientsNeedRescaling(
                     pipeline_config::Interleaved, 2, 1)
                     .ok());
  }
}

TEST_F(PipelineFixerTest, TestGetGradientScale) {
  std::string hlo = R"(
HloModule top

ENTRY e {
  p0 = s32[] parameter(0)
  ROOT t = () tuple()
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * scale,
                          pipeline_fixer_util::GetGradientScale(p0, 4, 0));

  EXPECT_TRUE(
      Match(scale,
            m::CustomCall(m::Select(
                m::Compare(m::Subtract(m::Parameter(0), m::ConstantScalar(4)),
                           m::CustomCall())
                    .WithComparisonDirection(ComparisonDirection::kLe),
                m::Convert(m::Parameter(0)),
                m::Add(m::Convert(m::CustomCall()), m::ConstantScalar(4.f))))));
}

TEST_F(PipelineFixerTest, TestRescaleGradient) {
  std::string hlo = R"(
HloModule top

ENTRY e {
  p0 = f16[2] parameter(0)
  p1 = f32[2] parameter(1)
  p2 = f32[] parameter(2)

  new_scale = f32[] constant(10)

  convert = f16[] convert(p2)
  b_scale0 = f16[2] broadcast(convert), dimensions={}
  grad0 = f16[2] multiply(p0, b_scale0)

  b_scale1 = f32[2] broadcast(p2), dimensions={}
  grad1 = f32[2] multiply(p1, b_scale1)

  ROOT t = () tuple()
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloInstruction* grad0 = FindInstruction(module.get(), "grad0");
  HloInstruction* grad1 = FindInstruction(module.get(), "grad1");
  HloInstruction* new_scale = FindInstruction(module.get(), "new_scale");

  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * new_grad0,
      pipeline_fixer_util::RescaleGradient(grad0, new_scale));
  EXPECT_TRUE(Match(
      new_grad0, m::Multiply(m::Parameter(0),
                             m::Broadcast(m::Convert(m::ConstantScalar(10))))));
  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * new_grad1,
      pipeline_fixer_util::RescaleGradient(grad1, new_scale));
  EXPECT_TRUE(
      Match(new_grad1,
            m::Multiply(m::Parameter(1), m::Broadcast(m::ConstantScalar(10)))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
