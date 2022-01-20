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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation_stage_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
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

using PipelineGroupedRecomputationTest = HloTestBase;

TEST_F(PipelineGroupedRecomputationTest, TestRecomputation1) {
  std::string hlo = R"(
HloModule cluster

_pop_op_conv_biasadd {
  arg_0.5 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.5 = f32[2]{0} parameter(1)
  broadcast.17.clone = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.5), dimensions={3}
  ROOT add.18.clone = f32[1,4,4,2]{3,2,1,0} add(arg_0.5, broadcast.17.clone)
}

pipeline_stage_0_func_11_rewritten__.11 {
  arg0.12 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg2.14 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  convolution.16 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.12, arg2.14), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  arg3.15 = f32[2]{0} parameter(3), sharding={maximal device=0}
  fusion.5 = f32[1,4,4,2]{3,2,1,0} fusion(convolution.16, arg3.15), kind=kCustom, calls=_pop_op_conv_biasadd, sharding={maximal device=0}
  arg1.13 = f32[] parameter(1), sharding={maximal device=0}
  ROOT tuple.50 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}) tuple(fusion.5, arg1.13, arg0.12), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
}

_pop_op_conv_biasadd.1 {
  arg_0.6 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.6 = f32[2]{0} parameter(1)
  broadcast.30.clone = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.6), dimensions={3}
  ROOT add.31.clone = f32[1,4,4,2]{3,2,1,0} add(arg_0.6, broadcast.30.clone)
}

pipeline_stage_1_func_34_rewritten__.24 {
  arg0.25 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=1}
  arg2.27 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  convolution.29 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.25, arg2.27), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  arg3.28 = f32[2]{0} parameter(3), sharding={maximal device=1}
  fusion.6 = f32[1,4,4,2]{3,2,1,0} fusion(convolution.29, arg3.28), kind=kCustom, calls=_pop_op_conv_biasadd.1, sharding={maximal device=1}
  arg1.26 = f32[] parameter(1), sharding={maximal device=1}
  ROOT tuple.48 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(fusion.6, arg1.26, arg0.25, arg2.27), sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
}

Sum-reduction.37 {
  x.38 = f32[] parameter(0)
  y.39 = f32[] parameter(1)
  ROOT add.40 = f32[] add(x.38, y.39)
}

_pop_op_implicit_binary_inplace {
  arg_0.20 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.17 = f32[2]{0} parameter(1)
  broadcast.111 = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.17), dimensions={3}
  ROOT add.24 = f32[1,4,4,2]{3,2,1,0} add(arg_0.20, broadcast.111)
}

pipeline_stage_2_func_57_rewritten__.0 {
  arg0.0 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=2}
  arg2.0 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=2}
  convolution = f32[1,4,4,2]{3,2,1,0} convolution(arg0.0, arg2.0), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=2}
  arg3.0 = f32[2]{0} parameter(3), sharding={maximal device=2}
  fusion.20 = f32[1,4,4,2]{3,2,1,0} fusion(convolution, arg3.0), kind=kCustom, calls=_pop_op_implicit_binary_inplace, sharding={maximal device=2}
  constant = f32[] constant(0), sharding={maximal device=2}
  reduce = f32[] reduce(fusion.20, constant), dimensions={0,1,2,3}, to_apply=Sum-reduction.37, sharding={maximal device=2}
  after-all = token[] after-all(), sharding={maximal device=2}
  outfeed = token[] outfeed(reduce, after-all), outfeed_config="\010\001\022\005feed0\"\001\001(\001", sharding={maximal device=2}
  arg1.0 = f32[] parameter(1), sharding={maximal device=2}
  ROOT tuple.47 = (f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(arg1.0, arg0.0, arg2.0), sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}}
}

_pop_op_scaled_inplace {
  arg_0.7 = f32[2]{0} parameter(0)
  arg_1.7 = f32[2]{0} parameter(1)
  constant.72.clone = f32[] constant(0.1)
  broadcast.87.clone = f32[2]{0} broadcast(constant.72.clone), dimensions={}
  multiply.31.clone = f32[2]{0} multiply(arg_1.7, broadcast.87.clone)
  ROOT add.21.clone = f32[2]{0} add(arg_0.7, multiply.31.clone)
}

_pop_op_scaled_inplace.1 {
  arg_0.8 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.8 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  constant.72.clone.1 = f32[] constant(0.1)
  broadcast.83.clone = f32[1,1,2,2]{3,2,1,0} broadcast(constant.72.clone.1), dimensions={}
  multiply.29.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.8, broadcast.83.clone)
  ROOT add.20.clone = f32[1,1,2,2]{3,2,1,0} add(arg_0.8, multiply.29.clone)
}

_pop_op_scaled_inplace.2 {
  arg_0.9 = f32[2]{0} parameter(0)
  arg_1.9 = f32[2]{0} parameter(1)
  arg_2 = f32[] parameter(2)
  broadcast.89.clone = f32[2]{0} broadcast(arg_2), dimensions={}
  multiply.32.clone = f32[2]{0} multiply(arg_1.9, broadcast.89.clone)
  ROOT subtract.18.clone = f32[2]{0} subtract(arg_0.9, multiply.32.clone)
}

_pop_op_scaled_inplace.3 {
  arg_0.10 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.10 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.1 = f32[] parameter(2)
  broadcast.85.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.1), dimensions={}
  multiply.30.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.10, broadcast.85.clone)
  ROOT subtract.17.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.10, multiply.30.clone)
}

_pop_op_implicit_ternary.1 {
  constant.71.clone = f32[] constant(-1)
  broadcast.107 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.71.clone), dimensions={}
  arg_0.18 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.74.clone = f32[] constant(1)
  broadcast.108 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.74.clone), dimensions={}
  ROOT clamp.13 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.107, arg_0.18, broadcast.108)
}

_pop_op_implicit_ternary.4 {
  constant.71.clone.1 = f32[] constant(-1)
  broadcast.114 = f32[2]{0} broadcast(constant.71.clone.1), dimensions={}
  arg_0.22 = f32[2]{0} parameter(0)
  constant.74.clone.1 = f32[] constant(1)
  broadcast.115 = f32[2]{0} broadcast(constant.74.clone.1), dimensions={}
  ROOT clamp.16 = f32[2]{0} clamp(broadcast.114, arg_0.22, broadcast.115)
}

pipeline_stage_0_func_11_grad_174__.2 {
  arg2.15 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  arg2.13 = f32[1,4,4,2]{3,2,1,0} parameter(1), sharding={maximal device=0}
  arg0.11 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  convolution.22 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.13, arg0.11), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=0}
  fusion.8 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.22, arg2.15), kind=kCustom, calls=_pop_op_scaled_inplace.1, sharding={maximal device=0}
  fusion.18 = f32[1,1,2,2]{3,2,1,0} fusion(fusion.8), kind=kCustom, calls=_pop_op_implicit_ternary.1, sharding={maximal device=0}
  get-tuple-element.23 = f32[] parameter(3), sharding={maximal device=0}
  fusion.10 = f32[1,1,2,2]{3,2,1,0} fusion(arg2.15, fusion.18, get-tuple-element.23), kind=kCustom, calls=_pop_op_scaled_inplace.3, sharding={maximal device=0}
  arg3.13 = f32[2]{0} parameter(4), sharding={maximal device=0}
  constant.75 = f32[] constant(0), sharding={maximal device=0}
  reduce.11 = f32[2]{0} reduce(arg0.11, constant.75), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=0}
  fusion.7 = f32[2]{0} fusion(reduce.11, arg3.13), kind=kCustom, calls=_pop_op_scaled_inplace, sharding={maximal device=0}
  fusion.22 = f32[2]{0} fusion(fusion.7), kind=kCustom, calls=_pop_op_implicit_ternary.4, sharding={maximal device=0}
  fusion.9 = f32[2]{0} fusion(arg3.13, fusion.22, get-tuple-element.23), kind=kCustom, calls=_pop_op_scaled_inplace.2, sharding={maximal device=0}
  ROOT tuple.42 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(fusion.10, fusion.9), sharding={{maximal device=0}, {maximal device=0}}
}

_pop_op_conv_with_reverse.clone {
  arg_0.3 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.3 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  reverse.10 = f32[1,1,2,2]{3,2,1,0} reverse(arg_1.3), dimensions={0,1}
  ROOT convolution.23 = f32[1,4,4,2]{3,2,1,0} convolution(arg_0.3, reverse.10), window={size=1x1}, dim_labels=b01f_01oi->b01f
}

_pop_op_scaled_inplace.4 {
  arg_0.11 = f32[2]{0} parameter(0)
  arg_1.11 = f32[2]{0} parameter(1)
  arg_2.2 = f32[] parameter(2)
  broadcast.95.clone = f32[2]{0} broadcast(arg_2.2), dimensions={}
  multiply.34.clone = f32[2]{0} multiply(arg_1.11, broadcast.95.clone)
  ROOT subtract.20.clone = f32[2]{0} subtract(arg_0.11, multiply.34.clone)
}

_pop_op_scaled_inplace.5 {
  arg_0.12 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.12 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.3 = f32[] parameter(2)
  broadcast.92.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.3), dimensions={}
  multiply.33.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.12, broadcast.92.clone)
  ROOT subtract.19.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.12, multiply.33.clone)
}

_pop_op_implicit_ternary.2 {
  constant.76.clone = f32[] constant(-1)
  broadcast.109 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.76.clone), dimensions={}
  arg_0.19 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.77.clone = f32[] constant(1)
  broadcast.110 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.77.clone), dimensions={}
  ROOT clamp.14 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.109, arg_0.19, broadcast.110)
}

_pop_op_implicit_ternary.3 {
  constant.76.clone.1 = f32[] constant(-1)
  broadcast.112 = f32[2]{0} broadcast(constant.76.clone.1), dimensions={}
  arg_0.21 = f32[2]{0} parameter(0)
  constant.77.clone.1 = f32[] constant(1)
  broadcast.113 = f32[2]{0} broadcast(constant.77.clone.1), dimensions={}
  ROOT clamp.15 = f32[2]{0} clamp(broadcast.112, arg_0.21, broadcast.113)
}

pipeline_stage_1_func_34_grad_139__.3 {
  arg0.13 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=1}
  arg3.14 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  fusion.3 = f32[1,4,4,2]{3,2,1,0} fusion(arg0.13, arg3.14), kind=kCustom, calls=_pop_op_conv_with_reverse.clone, sharding={maximal device=1}
  arg4.9 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=1}
  arg2.16 = f32[1,4,4,2]{3,2,1,0} parameter(1), sharding={maximal device=1}
  convolution.24 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.16, arg0.13), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=1}
  fusion.19 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.24), kind=kCustom, calls=_pop_op_implicit_ternary.2, sharding={maximal device=1}
  get-tuple-element.24 = f32[] parameter(3), sharding={maximal device=1}
  fusion.12 = f32[1,1,2,2]{3,2,1,0} fusion(arg4.9, fusion.19, get-tuple-element.24), kind=kCustom, calls=_pop_op_scaled_inplace.5, sharding={maximal device=1}
  arg5.2 = f32[2]{0} parameter(5), sharding={maximal device=1}
  constant.78 = f32[] constant(0), sharding={maximal device=1}
  reduce.12 = f32[2]{0} reduce(arg0.13, constant.78), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=1}
  fusion.21 = f32[2]{0} fusion(reduce.12), kind=kCustom, calls=_pop_op_implicit_ternary.3, sharding={maximal device=1}
  fusion.11 = f32[2]{0} fusion(arg5.2, fusion.21, get-tuple-element.24), kind=kCustom, calls=_pop_op_scaled_inplace.4, sharding={maximal device=1}
  ROOT tuple.44 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) tuple(fusion.3, fusion.12, fusion.11, get-tuple-element.24), sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
}

_pop_op_conv_with_reverse.2.clone {
  arg_0.4 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.4 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  reverse.11 = f32[1,1,2,2]{3,2,1,0} reverse(arg_1.4), dimensions={0,1}
  ROOT convolution.25 = f32[1,4,4,2]{3,2,1,0} convolution(arg_0.4, reverse.11), window={size=1x1}, dim_labels=b01f_01oi->b01f
}

_pop_op_scaled_inplace.6 {
  arg_0.13 = f32[2]{0} parameter(0)
  arg_1.13 = f32[2]{0} parameter(1)
  constant.82.clone = f32[] constant(0.1)
  broadcast.102.clone = f32[2]{0} broadcast(constant.82.clone), dimensions={}
  multiply.37.clone = f32[2]{0} multiply(arg_1.13, broadcast.102.clone)
  ROOT add.23.clone = f32[2]{0} add(arg_0.13, multiply.37.clone)
}

_pop_op_scaled_inplace.7 {
  arg_0.14 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.14 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  constant.82.clone.1 = f32[] constant(0.1)
  broadcast.98.clone = f32[1,1,2,2]{3,2,1,0} broadcast(constant.82.clone.1), dimensions={}
  multiply.35.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.14, broadcast.98.clone)
  ROOT add.22.clone = f32[1,1,2,2]{3,2,1,0} add(arg_0.14, multiply.35.clone)
}

_pop_op_scaled_inplace.8 {
  arg_0.15 = f32[2]{0} parameter(0)
  arg_1.15 = f32[2]{0} parameter(1)
  arg_2.4 = f32[] parameter(2)
  broadcast.104.clone = f32[2]{0} broadcast(arg_2.4), dimensions={}
  multiply.38.clone = f32[2]{0} multiply(arg_1.15, broadcast.104.clone)
  ROOT subtract.22.clone = f32[2]{0} subtract(arg_0.15, multiply.38.clone)
}

_pop_op_scaled_inplace.9 {
  arg_0.16 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.16 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.5 = f32[] parameter(2)
  broadcast.100.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.5), dimensions={}
  multiply.36.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.16, broadcast.100.clone)
  ROOT subtract.21.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.16, multiply.36.clone)
}

_pop_op_implicit_ternary {
  constant.81.clone = f32[] constant(-1)
  broadcast.105 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.81.clone), dimensions={}
  arg_0.17 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.83.clone = f32[] constant(1)
  broadcast.106 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.83.clone), dimensions={}
  ROOT clamp.12 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.105, arg_0.17, broadcast.106)
}

_pop_op_implicit_ternary.5 {
  constant.81.clone.1 = f32[] constant(-1)
  broadcast.116 = f32[2]{0} broadcast(constant.81.clone.1), dimensions={}
  arg_0.23 = f32[2]{0} parameter(0)
  constant.83.clone.1 = f32[] constant(1)
  broadcast.117 = f32[2]{0} broadcast(constant.83.clone.1), dimensions={}
  ROOT clamp.17 = f32[2]{0} clamp(broadcast.116, arg_0.23, broadcast.117)
}

pipeline_stage_2_func_57_grad_98__.5 {
  constant.80 = f32[1]{0} constant({1}), sharding={maximal device=2}
  broadcast.96 = f32[1,4,4,2]{3,2,1,0} broadcast(constant.80), dimensions={0}, sharding={maximal device=2}
  arg4.10 = f32[1,1,2,2]{3,2,1,0} parameter(1), sharding={maximal device=2}
  fusion.4 = f32[1,4,4,2]{3,2,1,0} fusion(broadcast.96, arg4.10), kind=kCustom, calls=_pop_op_conv_with_reverse.2.clone, sharding={maximal device=2}
  arg6.5 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=2}
  arg3.16 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=2}
  convolution.26 = f32[1,1,2,2]{3,2,1,0} convolution(arg3.16, broadcast.96), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=2}
  fusion.14 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.26, arg6.5), kind=kCustom, calls=_pop_op_scaled_inplace.7, sharding={maximal device=2}
  fusion.17 = f32[1,1,2,2]{3,2,1,0} fusion(fusion.14), kind=kCustom, calls=_pop_op_implicit_ternary, sharding={maximal device=2}
  get-tuple-element.161.clone.18 = f32[] parameter(3), sharding={maximal device=2}
  fusion.16 = f32[1,1,2,2]{3,2,1,0} fusion(arg6.5, fusion.17, get-tuple-element.161.clone.18), kind=kCustom, calls=_pop_op_scaled_inplace.9, sharding={maximal device=2}
  arg7.4 = f32[2]{0} parameter(4), sharding={maximal device=2}
  constant.84 = f32[] constant(0), sharding={maximal device=2}
  reduce.13 = f32[2]{0} reduce(broadcast.96, constant.84), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=2}
  fusion.13 = f32[2]{0} fusion(reduce.13, arg7.4), kind=kCustom, calls=_pop_op_scaled_inplace.6, sharding={maximal device=2}
  fusion.23 = f32[2]{0} fusion(fusion.13), kind=kCustom, calls=_pop_op_implicit_ternary.5, sharding={maximal device=2}
  fusion.15 = f32[2]{0} fusion(arg7.4, fusion.23, get-tuple-element.161.clone.18), kind=kCustom, calls=_pop_op_scaled_inplace.8, sharding={maximal device=2}
  ROOT tuple.46 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) tuple(fusion.4, fusion.16, fusion.15, get-tuple-element.161.clone.18), sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}, {maximal device=2}}
}

resource_update {
  arg0 = f32[1,1,2,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg1 = f32[2]{0} parameter(1), sharding={maximal device=0}
  arg2 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  arg3 = f32[2]{0} parameter(3), sharding={maximal device=1}
  arg4 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=2}
  arg5 = f32[2]{0} parameter(5), sharding={maximal device=2}
  ROOT t = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0, arg1, arg2, arg3, arg4, arg5)
}

pipeline {
  arg0.125 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg1.126 = f32[] parameter(1), sharding={maximal device=0}
  arg2.127 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  arg3.128 = f32[2]{0} parameter(3), sharding={maximal device=0}
  call.145 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}) call(arg0.125, arg1.126, arg2.127, arg3.128), to_apply=pipeline_stage_0_func_11_rewritten__.11, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  get-tuple-element.146 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.145), index=0, sharding={maximal device=0}
  get-tuple-element.147 = f32[] get-tuple-element(call.145), index=1, sharding={maximal device=0}
  arg4.129 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=1}
  arg5.130 = f32[2]{0} parameter(5), sharding={maximal device=1}
  call.152 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) call(get-tuple-element.146, get-tuple-element.147, arg4.129, arg5.130), to_apply=pipeline_stage_1_func_34_rewritten__.24, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
  get-tuple-element.153 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.152), index=0, sharding={maximal device=1}
  get-tuple-element.154 = f32[] get-tuple-element(call.152), index=1, sharding={maximal device=1}
  arg6.131 = f32[1,1,2,2]{3,2,1,0} parameter(6), sharding={maximal device=2}
  arg7.132 = f32[2]{0} parameter(7), sharding={maximal device=2}
  call = (f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) call(get-tuple-element.153, get-tuple-element.154, arg6.131, arg7.132), to_apply=pipeline_stage_2_func_57_rewritten__.0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}}
  get-tuple-element.25 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call), index=1, sharding={maximal device=2}
  get-tuple-element.26 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=2, sharding={maximal device=2}
  get-tuple-element.161.clone = f32[] get-tuple-element(call), index=0, sharding={maximal device=2}
  call.13 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) call(get-tuple-element.25, get-tuple-element.26, arg6.131, get-tuple-element.161.clone, arg7.132), to_apply=pipeline_stage_2_func_57_grad_98__.5, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}, {maximal device=2}}
  get-tuple-element.27 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.152), index=2, sharding={maximal device=1}
  get-tuple-element.28 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.152), index=3, sharding={maximal device=1}
  get-tuple-element.5 = f32[] get-tuple-element(call.13), index=3, sharding={maximal device=1}
  get-tuple-element.176 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.13), index=0, sharding={maximal device=2}
  call.12 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) call(get-tuple-element.176, get-tuple-element.27, get-tuple-element.28, get-tuple-element.5, arg4.129, arg5.130), to_apply=pipeline_stage_1_func_34_grad_139__.3, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
  get-tuple-element.203 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.12), index=0, sharding={maximal device=1}
  get-tuple-element.14 = f32[] get-tuple-element(call.12), index=3, sharding={maximal device=1}
  get-tuple-element.29 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.145), index=2, sharding={maximal device=0}
  call.11 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.203, get-tuple-element.29, arg2.127, get-tuple-element.14, arg3.128), to_apply=pipeline_stage_0_func_11_grad_174__.2, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}}
  get-tuple-element.17 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.11), index=0, sharding={maximal device=0}
  get-tuple-element.21 = f32[2]{0} get-tuple-element(call.11), index=1, sharding={maximal device=0}
  get-tuple-element.7 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.12), index=1, sharding={maximal device=1}
  get-tuple-element.10 = f32[2]{0} get-tuple-element(call.12), index=2, sharding={maximal device=1}
  get-tuple-element = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.13), index=1, sharding={maximal device=2}
  get-tuple-element.1 = f32[2]{0} get-tuple-element(call.13), index=2, sharding={maximal device=2}
  call_ru = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.17, get-tuple-element.21, get-tuple-element.7, get-tuple-element.10, get-tuple-element, get-tuple-element.1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}", sharding={{maximal device=0}, {maximal device=0}, {maximal device=1}, {maximal device=1}, {maximal device=2}, {maximal device=2}}
  gte0 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=0, sharding={maximal device=0}
  gte1 = f32[2]{0} get-tuple-element(call_ru), index=1, sharding={maximal device=0}
  gte2 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=2, sharding={maximal device=1}
  gte3 = f32[2]{0} get-tuple-element(call_ru), index=3, sharding={maximal device=1}
  gte4 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=4, sharding={maximal device=2}
  gte5 = f32[2]{0} get-tuple-element(call_ru), index=5, sharding={maximal device=2}
  ROOT tuple.266 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0.125, arg1.126, gte0, gte1, gte2, gte3, gte4, gte5)
}

ENTRY cluster {
  arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0), parameter_replication={false}
  arg1.2 = f32[] parameter(1), parameter_replication={false}
  arg4.5 = f32[1,1,2,2]{3,2,1,0} parameter(4), parameter_replication={false}
  arg2.3 = f32[2]{0} parameter(2), parameter_replication={false}
  arg7.8 = f32[1,1,2,2]{3,2,1,0} parameter(7), parameter_replication={false}
  arg6.7 = f32[2]{0} parameter(6), parameter_replication={false}
  arg5.6 = f32[1,1,2,2]{3,2,1,0} parameter(5), parameter_replication={false}
  arg3.4 = f32[2]{0} parameter(3), parameter_replication={false}
  call.267 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.1, arg1.2, arg4.5, arg2.3, arg7.8, arg6.7, arg5.6, arg3.4), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  get-tuple-element.269 = f32[2]{0} get-tuple-element(call.267), index=3
  get-tuple-element.273 = f32[2]{0} get-tuple-element(call.267), index=7
  get-tuple-element.268 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=2
  get-tuple-element.272 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=6
  get-tuple-element.271 = f32[2]{0} get-tuple-element(call.267), index=5
  get-tuple-element.270 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=4
  ROOT tuple.286 = (f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.269, get-tuple-element.273, get-tuple-element.268, get-tuple-element.272, get-tuple-element.271, get-tuple-element.270)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFIFOInserter inserter(false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  PipelineRecomputationStageInserter recomputation(true, false);
  TF_ASSERT_OK_AND_ASSIGN(changed, recomputation.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_computation =
      FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_computation));
  // There are three stages, but we do not recompute the last stage.
  EXPECT_THAT(stages.recomputation.size(), 2);
  EXPECT_TRUE(stages.recomputation.contains(0));
  EXPECT_TRUE(stages.recomputation.contains(1));
  for (auto pair : stages.recomputation) {
    // Given a recomputation stage, go through the corresponding forward stage.
    // Expect a fifo for each operand of the forward stage which is not a
    // parameter.
    auto stage_id = pair.first;
    auto recomp_stage = pair.second;
    auto fwd_stage = stages.forward[stage_id];
    for (int64 op_idx = 0; op_idx != fwd_stage->operand_count(); ++op_idx) {
      auto operand = fwd_stage->operand(op_idx);
      if (operand->opcode() == HloOpcode::kParameter) {
        EXPECT_THAT(operand, recomp_stage->operand(op_idx));
      } else {
        EXPECT_TRUE(
            IsPoplarInstruction(PoplarOp::Fifo)(recomp_stage->operand(op_idx)));
        EXPECT_THAT(operand, recomp_stage->operand(op_idx)->operand(0));
      }
    }

    // Expect that all users of the recomputation stage are
    // GTE->PipelineStageBackward with the same stage_id.
    for (auto user : recomp_stage->users()) {
      EXPECT_THAT(user->opcode(), HloOpcode::kGetTupleElement);
      for (auto user_of_user : user->users()) {
        EXPECT_THAT(user_of_user, stages.backward[stage_id]);
      }
    }

    // Expect that the recomputation stage has a control predecessor that
    // is the forward pass.
    ASSERT_EQ(1, recomp_stage->control_predecessors().size());
    EXPECT_EQ(fwd_stage, recomp_stage->control_predecessors()[0]);
  }
}

TEST_F(PipelineGroupedRecomputationTest, TestNoBackwardStages) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  stage_0_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_0_fwd_weights1 = f32[1,4,4,2] parameter(1)
  log = f32[1,4,4,2] log(stage_0_fwd_weights1), backend_config="{\"isInplace\":true}"
  add = f32[1,4,4,2] add(stage_0_fwd_weights0, log), backend_config="{\"isInplace\":true}"
  stage_0_fwd_weights2 = f32[1,4,4,2] parameter(2)
  ROOT stage_0_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(add, stage_0_fwd_weights2)
}

stage_1_fwd {
  stage_1_fwd_weights0 = f32[1,4,4,2] parameter(0)
  stage_1_fwd_weights2 = f32[1,4,4,2] parameter(1)
  ROOT stage_1_fwd_tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(stage_1_fwd_weights0, stage_1_fwd_weights2)
}

pipeline {
  pipeline_weights0 = f32[1,4,4,2] parameter(0)
  pipeline_weights1 = f32[1,4,4,2] parameter(1)
  pipeline_weights2 = f32[1,4,4,2] parameter(2)
  pipeline_stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_weights0, pipeline_weights1, pipeline_weights2), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  pipeline_stage_0_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=0
  pipeline_stage_0_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_0), index=1
  pipeline_stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(pipeline_stage_0_w0, pipeline_stage_0_w2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  pipeline_stage_1_w0 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=0
  pipeline_stage_1_w2 = f32[1,4,4,2] get-tuple-element(pipeline_stage_1), index=1
  ROOT pipeline_tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(pipeline_stage_1_w0, pipeline_weights1, pipeline_stage_1_w2)
}

ENTRY e {
  e.weights0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.weights1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  e.weights2 = f32[1,4,4,2] parameter(2), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(e.weights0, e.weights1, e.weights2), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  InterIpuCopyInserter inter_inserter;
  TF_ASSERT_OK_AND_ASSIGN(changed, inter_inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  PipelineRecomputationStageInserter recomputation(true, false);
  TF_ASSERT_OK_AND_ASSIGN(changed, recomputation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineGroupedRecomputationTest, TestRecomputationDisabled) {
  std::string hlo = R"(
HloModule cluster

_pop_op_conv_biasadd {
  arg_0.5 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.5 = f32[2]{0} parameter(1)
  broadcast.17.clone = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.5), dimensions={3}
  ROOT add.18.clone = f32[1,4,4,2]{3,2,1,0} add(arg_0.5, broadcast.17.clone)
}

pipeline_stage_0_func_11_rewritten__.11 {
  arg0.12 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg2.14 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  convolution.16 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.12, arg2.14), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  arg3.15 = f32[2]{0} parameter(3), sharding={maximal device=0}
  fusion.5 = f32[1,4,4,2]{3,2,1,0} fusion(convolution.16, arg3.15), kind=kCustom, calls=_pop_op_conv_biasadd, sharding={maximal device=0}
  arg1.13 = f32[] parameter(1), sharding={maximal device=0}
  ROOT tuple.50 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}) tuple(fusion.5, arg1.13, arg0.12), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
}

_pop_op_conv_biasadd.1 {
  arg_0.6 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.6 = f32[2]{0} parameter(1)
  broadcast.30.clone = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.6), dimensions={3}
  ROOT add.31.clone = f32[1,4,4,2]{3,2,1,0} add(arg_0.6, broadcast.30.clone)
}

pipeline_stage_1_func_34_rewritten__.24 {
  arg0.25 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=1}
  arg2.27 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  convolution.29 = f32[1,4,4,2]{3,2,1,0} convolution(arg0.25, arg2.27), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  arg3.28 = f32[2]{0} parameter(3), sharding={maximal device=1}
  fusion.6 = f32[1,4,4,2]{3,2,1,0} fusion(convolution.29, arg3.28), kind=kCustom, calls=_pop_op_conv_biasadd.1, sharding={maximal device=1}
  arg1.26 = f32[] parameter(1), sharding={maximal device=1}
  ROOT tuple.48 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(fusion.6, arg1.26, arg0.25, arg2.27), sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
}

Sum-reduction.37 {
  x.38 = f32[] parameter(0)
  y.39 = f32[] parameter(1)
  ROOT add.40 = f32[] add(x.38, y.39)
}

_pop_op_implicit_binary_inplace {
  arg_0.20 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.17 = f32[2]{0} parameter(1)
  broadcast.111 = f32[1,4,4,2]{3,2,1,0} broadcast(arg_1.17), dimensions={3}
  ROOT add.24 = f32[1,4,4,2]{3,2,1,0} add(arg_0.20, broadcast.111)
}

pipeline_stage_2_func_57_rewritten__.0 {
  arg0.0 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=2}
  arg2.0 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=2}
  convolution = f32[1,4,4,2]{3,2,1,0} convolution(arg0.0, arg2.0), window={size=1x1}, dim_labels=b01f_01io->b01f, sharding={maximal device=2}
  arg3.0 = f32[2]{0} parameter(3), sharding={maximal device=2}
  fusion.20 = f32[1,4,4,2]{3,2,1,0} fusion(convolution, arg3.0), kind=kCustom, calls=_pop_op_implicit_binary_inplace, sharding={maximal device=2}
  constant = f32[] constant(0), sharding={maximal device=2}
  reduce = f32[] reduce(fusion.20, constant), dimensions={0,1,2,3}, to_apply=Sum-reduction.37, sharding={maximal device=2}
  after-all = token[] after-all(), sharding={maximal device=2}
  outfeed = token[] outfeed(reduce, after-all), outfeed_config="\010\001\022\005feed0\"\001\001(\001", sharding={maximal device=2}
  arg1.0 = f32[] parameter(1), sharding={maximal device=2}
  ROOT tuple.47 = (f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(arg1.0, arg0.0, arg2.0), sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}}
}

_pop_op_scaled_inplace {
  arg_0.7 = f32[2]{0} parameter(0)
  arg_1.7 = f32[2]{0} parameter(1)
  constant.72.clone = f32[] constant(0.1)
  broadcast.87.clone = f32[2]{0} broadcast(constant.72.clone), dimensions={}
  multiply.31.clone = f32[2]{0} multiply(arg_1.7, broadcast.87.clone)
  ROOT add.21.clone = f32[2]{0} add(arg_0.7, multiply.31.clone)
}

_pop_op_scaled_inplace.1 {
  arg_0.8 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.8 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  constant.72.clone.1 = f32[] constant(0.1)
  broadcast.83.clone = f32[1,1,2,2]{3,2,1,0} broadcast(constant.72.clone.1), dimensions={}
  multiply.29.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.8, broadcast.83.clone)
  ROOT add.20.clone = f32[1,1,2,2]{3,2,1,0} add(arg_0.8, multiply.29.clone)
}

_pop_op_scaled_inplace.2 {
  arg_0.9 = f32[2]{0} parameter(0)
  arg_1.9 = f32[2]{0} parameter(1)
  arg_2 = f32[] parameter(2)
  broadcast.89.clone = f32[2]{0} broadcast(arg_2), dimensions={}
  multiply.32.clone = f32[2]{0} multiply(arg_1.9, broadcast.89.clone)
  ROOT subtract.18.clone = f32[2]{0} subtract(arg_0.9, multiply.32.clone)
}

_pop_op_scaled_inplace.3 {
  arg_0.10 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.10 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.1 = f32[] parameter(2)
  broadcast.85.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.1), dimensions={}
  multiply.30.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.10, broadcast.85.clone)
  ROOT subtract.17.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.10, multiply.30.clone)
}

_pop_op_implicit_ternary.1 {
  constant.71.clone = f32[] constant(-1)
  broadcast.107 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.71.clone), dimensions={}
  arg_0.18 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.74.clone = f32[] constant(1)
  broadcast.108 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.74.clone), dimensions={}
  ROOT clamp.13 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.107, arg_0.18, broadcast.108)
}

_pop_op_implicit_ternary.4 {
  constant.71.clone.1 = f32[] constant(-1)
  broadcast.114 = f32[2]{0} broadcast(constant.71.clone.1), dimensions={}
  arg_0.22 = f32[2]{0} parameter(0)
  constant.74.clone.1 = f32[] constant(1)
  broadcast.115 = f32[2]{0} broadcast(constant.74.clone.1), dimensions={}
  ROOT clamp.16 = f32[2]{0} clamp(broadcast.114, arg_0.22, broadcast.115)
}

pipeline_stage_0_func_11_grad_174__.2 {
  arg2.15 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  arg2.13 = f32[1,4,4,2]{3,2,1,0} parameter(1), sharding={maximal device=0}
  arg0.11 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  convolution.22 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.13, arg0.11), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=0}
  fusion.8 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.22, arg2.15), kind=kCustom, calls=_pop_op_scaled_inplace.1, sharding={maximal device=0}
  fusion.18 = f32[1,1,2,2]{3,2,1,0} fusion(fusion.8), kind=kCustom, calls=_pop_op_implicit_ternary.1, sharding={maximal device=0}
  get-tuple-element.23 = f32[] parameter(3), sharding={maximal device=0}
  fusion.10 = f32[1,1,2,2]{3,2,1,0} fusion(arg2.15, fusion.18, get-tuple-element.23), kind=kCustom, calls=_pop_op_scaled_inplace.3, sharding={maximal device=0}
  arg3.13 = f32[2]{0} parameter(4), sharding={maximal device=0}
  constant.75 = f32[] constant(0), sharding={maximal device=0}
  reduce.11 = f32[2]{0} reduce(arg0.11, constant.75), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=0}
  fusion.7 = f32[2]{0} fusion(reduce.11, arg3.13), kind=kCustom, calls=_pop_op_scaled_inplace, sharding={maximal device=0}
  fusion.22 = f32[2]{0} fusion(fusion.7), kind=kCustom, calls=_pop_op_implicit_ternary.4, sharding={maximal device=0}
  fusion.9 = f32[2]{0} fusion(arg3.13, fusion.22, get-tuple-element.23), kind=kCustom, calls=_pop_op_scaled_inplace.2, sharding={maximal device=0}
  ROOT tuple.42 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(fusion.10, fusion.9), sharding={{maximal device=0}, {maximal device=0}}
}

_pop_op_conv_with_reverse.clone {
  arg_0.3 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.3 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  reverse.10 = f32[1,1,2,2]{3,2,1,0} reverse(arg_1.3), dimensions={0,1}
  ROOT convolution.23 = f32[1,4,4,2]{3,2,1,0} convolution(arg_0.3, reverse.10), window={size=1x1}, dim_labels=b01f_01oi->b01f
}

_pop_op_scaled_inplace.4 {
  arg_0.11 = f32[2]{0} parameter(0)
  arg_1.11 = f32[2]{0} parameter(1)
  arg_2.2 = f32[] parameter(2)
  broadcast.95.clone = f32[2]{0} broadcast(arg_2.2), dimensions={}
  multiply.34.clone = f32[2]{0} multiply(arg_1.11, broadcast.95.clone)
  ROOT subtract.20.clone = f32[2]{0} subtract(arg_0.11, multiply.34.clone)
}

_pop_op_scaled_inplace.5 {
  arg_0.12 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.12 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.3 = f32[] parameter(2)
  broadcast.92.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.3), dimensions={}
  multiply.33.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.12, broadcast.92.clone)
  ROOT subtract.19.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.12, multiply.33.clone)
}

_pop_op_implicit_ternary.2 {
  constant.76.clone = f32[] constant(-1)
  broadcast.109 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.76.clone), dimensions={}
  arg_0.19 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.77.clone = f32[] constant(1)
  broadcast.110 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.77.clone), dimensions={}
  ROOT clamp.14 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.109, arg_0.19, broadcast.110)
}

_pop_op_implicit_ternary.3 {
  constant.76.clone.1 = f32[] constant(-1)
  broadcast.112 = f32[2]{0} broadcast(constant.76.clone.1), dimensions={}
  arg_0.21 = f32[2]{0} parameter(0)
  constant.77.clone.1 = f32[] constant(1)
  broadcast.113 = f32[2]{0} broadcast(constant.77.clone.1), dimensions={}
  ROOT clamp.15 = f32[2]{0} clamp(broadcast.112, arg_0.21, broadcast.113)
}

pipeline_stage_1_func_34_grad_139__.3 {
  arg0.13 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=1}
  arg3.14 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  fusion.3 = f32[1,4,4,2]{3,2,1,0} fusion(arg0.13, arg3.14), kind=kCustom, calls=_pop_op_conv_with_reverse.clone, sharding={maximal device=1}
  arg4.9 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=1}
  arg2.16 = f32[1,4,4,2]{3,2,1,0} parameter(1), sharding={maximal device=1}
  convolution.24 = f32[1,1,2,2]{3,2,1,0} convolution(arg2.16, arg0.13), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=1}
  fusion.19 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.24), kind=kCustom, calls=_pop_op_implicit_ternary.2, sharding={maximal device=1}
  get-tuple-element.24 = f32[] parameter(3), sharding={maximal device=1}
  fusion.12 = f32[1,1,2,2]{3,2,1,0} fusion(arg4.9, fusion.19, get-tuple-element.24), kind=kCustom, calls=_pop_op_scaled_inplace.5, sharding={maximal device=1}
  arg5.2 = f32[2]{0} parameter(5), sharding={maximal device=1}
  constant.78 = f32[] constant(0), sharding={maximal device=1}
  reduce.12 = f32[2]{0} reduce(arg0.13, constant.78), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=1}
  fusion.21 = f32[2]{0} fusion(reduce.12), kind=kCustom, calls=_pop_op_implicit_ternary.3, sharding={maximal device=1}
  fusion.11 = f32[2]{0} fusion(arg5.2, fusion.21, get-tuple-element.24), kind=kCustom, calls=_pop_op_scaled_inplace.4, sharding={maximal device=1}
  ROOT tuple.44 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) tuple(fusion.3, fusion.12, fusion.11, get-tuple-element.24), sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
}

_pop_op_conv_with_reverse.2.clone {
  arg_0.4 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  arg_1.4 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  reverse.11 = f32[1,1,2,2]{3,2,1,0} reverse(arg_1.4), dimensions={0,1}
  ROOT convolution.25 = f32[1,4,4,2]{3,2,1,0} convolution(arg_0.4, reverse.11), window={size=1x1}, dim_labels=b01f_01oi->b01f
}

_pop_op_scaled_inplace.6 {
  arg_0.13 = f32[2]{0} parameter(0)
  arg_1.13 = f32[2]{0} parameter(1)
  constant.82.clone = f32[] constant(0.1)
  broadcast.102.clone = f32[2]{0} broadcast(constant.82.clone), dimensions={}
  multiply.37.clone = f32[2]{0} multiply(arg_1.13, broadcast.102.clone)
  ROOT add.23.clone = f32[2]{0} add(arg_0.13, multiply.37.clone)
}

_pop_op_scaled_inplace.7 {
  arg_0.14 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.14 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  constant.82.clone.1 = f32[] constant(0.1)
  broadcast.98.clone = f32[1,1,2,2]{3,2,1,0} broadcast(constant.82.clone.1), dimensions={}
  multiply.35.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.14, broadcast.98.clone)
  ROOT add.22.clone = f32[1,1,2,2]{3,2,1,0} add(arg_0.14, multiply.35.clone)
}

_pop_op_scaled_inplace.8 {
  arg_0.15 = f32[2]{0} parameter(0)
  arg_1.15 = f32[2]{0} parameter(1)
  arg_2.4 = f32[] parameter(2)
  broadcast.104.clone = f32[2]{0} broadcast(arg_2.4), dimensions={}
  multiply.38.clone = f32[2]{0} multiply(arg_1.15, broadcast.104.clone)
  ROOT subtract.22.clone = f32[2]{0} subtract(arg_0.15, multiply.38.clone)
}

_pop_op_scaled_inplace.9 {
  arg_0.16 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  arg_1.16 = f32[1,1,2,2]{3,2,1,0} parameter(1)
  arg_2.5 = f32[] parameter(2)
  broadcast.100.clone = f32[1,1,2,2]{3,2,1,0} broadcast(arg_2.5), dimensions={}
  multiply.36.clone = f32[1,1,2,2]{3,2,1,0} multiply(arg_1.16, broadcast.100.clone)
  ROOT subtract.21.clone = f32[1,1,2,2]{3,2,1,0} subtract(arg_0.16, multiply.36.clone)
}

_pop_op_implicit_ternary {
  constant.81.clone = f32[] constant(-1)
  broadcast.105 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.81.clone), dimensions={}
  arg_0.17 = f32[1,1,2,2]{3,2,1,0} parameter(0)
  constant.83.clone = f32[] constant(1)
  broadcast.106 = f32[1,1,2,2]{3,2,1,0} broadcast(constant.83.clone), dimensions={}
  ROOT clamp.12 = f32[1,1,2,2]{3,2,1,0} clamp(broadcast.105, arg_0.17, broadcast.106)
}

_pop_op_implicit_ternary.5 {
  constant.81.clone.1 = f32[] constant(-1)
  broadcast.116 = f32[2]{0} broadcast(constant.81.clone.1), dimensions={}
  arg_0.23 = f32[2]{0} parameter(0)
  constant.83.clone.1 = f32[] constant(1)
  broadcast.117 = f32[2]{0} broadcast(constant.83.clone.1), dimensions={}
  ROOT clamp.17 = f32[2]{0} clamp(broadcast.116, arg_0.23, broadcast.117)
}

pipeline_stage_2_func_57_grad_98__.5 {
  constant.80 = f32[1]{0} constant({1}), sharding={maximal device=2}
  broadcast.96 = f32[1,4,4,2]{3,2,1,0} broadcast(constant.80), dimensions={0}, sharding={maximal device=2}
  arg4.10 = f32[1,1,2,2]{3,2,1,0} parameter(1), sharding={maximal device=2}
  fusion.4 = f32[1,4,4,2]{3,2,1,0} fusion(broadcast.96, arg4.10), kind=kCustom, calls=_pop_op_conv_with_reverse.2.clone, sharding={maximal device=2}
  arg6.5 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=2}
  arg3.16 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=2}
  convolution.26 = f32[1,1,2,2]{3,2,1,0} convolution(arg3.16, broadcast.96), window={size=4x4}, dim_labels=f01b_i01o->01bf, sharding={maximal device=2}
  fusion.14 = f32[1,1,2,2]{3,2,1,0} fusion(convolution.26, arg6.5), kind=kCustom, calls=_pop_op_scaled_inplace.7, sharding={maximal device=2}
  fusion.17 = f32[1,1,2,2]{3,2,1,0} fusion(fusion.14), kind=kCustom, calls=_pop_op_implicit_ternary, sharding={maximal device=2}
  get-tuple-element.161.clone.18 = f32[] parameter(3), sharding={maximal device=2}
  fusion.16 = f32[1,1,2,2]{3,2,1,0} fusion(arg6.5, fusion.17, get-tuple-element.161.clone.18), kind=kCustom, calls=_pop_op_scaled_inplace.9, sharding={maximal device=2}
  arg7.4 = f32[2]{0} parameter(4), sharding={maximal device=2}
  constant.84 = f32[] constant(0), sharding={maximal device=2}
  reduce.13 = f32[2]{0} reduce(broadcast.96, constant.84), dimensions={0,1,2}, to_apply=Sum-reduction.37, sharding={maximal device=2}
  fusion.13 = f32[2]{0} fusion(reduce.13, arg7.4), kind=kCustom, calls=_pop_op_scaled_inplace.6, sharding={maximal device=2}
  fusion.23 = f32[2]{0} fusion(fusion.13), kind=kCustom, calls=_pop_op_implicit_ternary.5, sharding={maximal device=2}
  fusion.15 = f32[2]{0} fusion(arg7.4, fusion.23, get-tuple-element.161.clone.18), kind=kCustom, calls=_pop_op_scaled_inplace.8, sharding={maximal device=2}
  ROOT tuple.46 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) tuple(fusion.4, fusion.16, fusion.15, get-tuple-element.161.clone.18), sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}, {maximal device=2}}
}

resource_update {
  arg0 = f32[1,1,2,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg1 = f32[2]{0} parameter(1), sharding={maximal device=0}
  arg2 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=1}
  arg3 = f32[2]{0} parameter(3), sharding={maximal device=1}
  arg4 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=2}
  arg5 = f32[2]{0} parameter(5), sharding={maximal device=2}
  ROOT t = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(arg0, arg1, arg2, arg3, arg4, arg5)
}

pipeline {
  arg0.125 = f32[1,4,4,2]{3,2,1,0} parameter(0), sharding={maximal device=0}
  arg1.126 = f32[] parameter(1), sharding={maximal device=0}
  arg2.127 = f32[1,1,2,2]{3,2,1,0} parameter(2), sharding={maximal device=0}
  arg3.128 = f32[2]{0} parameter(3), sharding={maximal device=0}
  call.145 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}) call(arg0.125, arg1.126, arg2.127, arg3.128), to_apply=pipeline_stage_0_func_11_rewritten__.11, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  get-tuple-element.146 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.145), index=0, sharding={maximal device=0}
  get-tuple-element.147 = f32[] get-tuple-element(call.145), index=1, sharding={maximal device=0}
  arg4.129 = f32[1,1,2,2]{3,2,1,0} parameter(4), sharding={maximal device=1}
  arg5.130 = f32[2]{0} parameter(5), sharding={maximal device=1}
  call.152 = (f32[1,4,4,2]{3,2,1,0}, f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) call(get-tuple-element.146, get-tuple-element.147, arg4.129, arg5.130), to_apply=pipeline_stage_1_func_34_rewritten__.24, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
  get-tuple-element.153 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.152), index=0, sharding={maximal device=1}
  get-tuple-element.154 = f32[] get-tuple-element(call.152), index=1, sharding={maximal device=1}
  arg6.131 = f32[1,1,2,2]{3,2,1,0} parameter(6), sharding={maximal device=2}
  arg7.132 = f32[2]{0} parameter(7), sharding={maximal device=2}
  call = (f32[], f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) call(get-tuple-element.153, get-tuple-element.154, arg6.131, arg7.132), to_apply=pipeline_stage_2_func_57_rewritten__.0, frontend_attributes={CALL_CONFIG_TYPE=PipelineStage}, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}}
  get-tuple-element.25 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call), index=1, sharding={maximal device=2}
  get-tuple-element.26 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call), index=2, sharding={maximal device=2}
  get-tuple-element.161.clone = f32[] get-tuple-element(call), index=0, sharding={maximal device=2}
  call.13 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) call(get-tuple-element.25, get-tuple-element.26, arg6.131, get-tuple-element.161.clone, arg7.132), to_apply=pipeline_stage_2_func_57_grad_98__.5, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"2\"}}}", sharding={{maximal device=2}, {maximal device=2}, {maximal device=2}, {maximal device=2}}
  get-tuple-element.27 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.152), index=2, sharding={maximal device=1}
  get-tuple-element.28 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.152), index=3, sharding={maximal device=1}
  get-tuple-element.5 = f32[] get-tuple-element(call.13), index=3, sharding={maximal device=1}
  get-tuple-element.176 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.13), index=0, sharding={maximal device=2}
  call.12 = (f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[]) call(get-tuple-element.176, get-tuple-element.27, get-tuple-element.28, get-tuple-element.5, arg4.129, arg5.130), to_apply=pipeline_stage_1_func_34_grad_139__.3, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={{maximal device=1}, {maximal device=1}, {maximal device=1}, {maximal device=1}}
  get-tuple-element.203 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.12), index=0, sharding={maximal device=1}
  get-tuple-element.14 = f32[] get-tuple-element(call.12), index=3, sharding={maximal device=1}
  get-tuple-element.29 = f32[1,4,4,2]{3,2,1,0} get-tuple-element(call.145), index=2, sharding={maximal device=0}
  call.11 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.203, get-tuple-element.29, arg2.127, get-tuple-element.14, arg3.128), to_apply=pipeline_stage_0_func_11_grad_174__.2, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={{maximal device=0}, {maximal device=0}}
  get-tuple-element.17 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.11), index=0, sharding={maximal device=0}
  get-tuple-element.21 = f32[2]{0} get-tuple-element(call.11), index=1, sharding={maximal device=0}
  get-tuple-element.7 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.12), index=1, sharding={maximal device=1}
  get-tuple-element.10 = f32[2]{0} get-tuple-element(call.12), index=2, sharding={maximal device=1}
  get-tuple-element = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.13), index=1, sharding={maximal device=2}
  get-tuple-element.1 = f32[2]{0} get-tuple-element(call.13), index=2, sharding={maximal device=2}
  call_ru = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(get-tuple-element.17, get-tuple-element.21, get-tuple-element.7, get-tuple-element.10, get-tuple-element, get-tuple-element.1), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}", sharding={{maximal device=0}, {maximal device=0}, {maximal device=1}, {maximal device=1}, {maximal device=2}, {maximal device=2}}
  gte0 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=0, sharding={maximal device=0}
  gte1 = f32[2]{0} get-tuple-element(call_ru), index=1, sharding={maximal device=0}
  gte2 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=2, sharding={maximal device=1}
  gte3 = f32[2]{0} get-tuple-element(call_ru), index=3, sharding={maximal device=1}
  gte4 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call_ru), index=4, sharding={maximal device=2}
  gte5 = f32[2]{0} get-tuple-element(call_ru), index=5, sharding={maximal device=2}
  ROOT tuple.266 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) tuple(gte0, gte1, gte2, gte3, gte4, gte5)
}

ENTRY cluster {
  arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0), parameter_replication={false}
  arg1.2 = f32[] parameter(1), parameter_replication={false}
  arg4.5 = f32[1,1,2,2]{3,2,1,0} parameter(4), parameter_replication={false}
  arg2.3 = f32[2]{0} parameter(2), parameter_replication={false}
  arg7.8 = f32[1,1,2,2]{3,2,1,0} parameter(7), parameter_replication={false}
  arg6.7 = f32[2]{0} parameter(6), parameter_replication={false}
  arg5.6 = f32[1,1,2,2]{3,2,1,0} parameter(5), parameter_replication={false}
  arg3.4 = f32[2]{0} parameter(3), parameter_replication={false}
  call.267 = (f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}) call(arg0.1, arg1.2, arg4.5, arg2.3, arg7.8, arg6.7, arg5.6, arg3.4), to_apply=pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
  get-tuple-element.269 = f32[2]{0} get-tuple-element(call.267), index=1
  get-tuple-element.273 = f32[2]{0} get-tuple-element(call.267), index=5
  get-tuple-element.268 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=0
  get-tuple-element.272 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=4
  get-tuple-element.271 = f32[2]{0} get-tuple-element(call.267), index=3
  get-tuple-element.270 = f32[1,1,2,2]{3,2,1,0} get-tuple-element(call.267), index=2
  ROOT tuple.286 = (f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(get-tuple-element.269, get-tuple-element.273, get-tuple-element.268, get-tuple-element.272, get-tuple-element.271, get-tuple-element.270)
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  PipelineFIFOInserter inserter(false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  PipelineRecomputationStageInserter recomputation(false, false);
  TF_ASSERT_OK_AND_ASSIGN(changed, recomputation.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(PipelineGroupedRecomputationTest, TestControlDependencies) {
  std::string hlo = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0, in1)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1, in2)
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
  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) tuple(arg0, arg1, arg2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,2], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,2] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,2] parameter(1)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,2]) call(in1, in0), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,2] get-tuple-element(stage_0), index=1
  stage_1 = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_0, stage_0_1), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,2] get-tuple-element(stage_1), index=1
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,2]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(gte0, gte1)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,2] parameter(1), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2]) call(e.in0, e.in1), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";
  auto config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ShardingPass sharding;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, sharding.Run(module.get()));
  EXPECT_TRUE(changed);

  PipelineFIFOInserter inserter(false);
  TF_ASSERT_OK_AND_ASSIGN(changed, inserter.Run(module.get()));
  EXPECT_TRUE(changed);

  CompilerAnnotations annotations{module.get()};
  InplaceFinder inplace_finder(annotations);
  TF_ASSERT_OK_AND_ASSIGN(changed, inplace_finder.Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* pipeline_comp = FindComputation(module.get(), "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));
  auto fifo = LookThroughCopies(stages.backward[0]->operand(1));
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Fifo)(fifo));
  EXPECT_THAT(fifo->control_successors(),
              ::testing::ElementsAre(stages.forward[1]));

  PipelineRecomputationStageInserter recomputation(true, false);
  TF_ASSERT_OK_AND_ASSIGN(changed, recomputation.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(stages, GetPipelineStages(pipeline_comp));

  fifo = stages.recomputation.at(0)->operand(1);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Fifo)(fifo));
  EXPECT_THAT(fifo->control_successors(),
              ::testing::ElementsAre(stages.forward[0]));
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
