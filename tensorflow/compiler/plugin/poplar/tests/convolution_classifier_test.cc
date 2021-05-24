/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ConvolutionClassifierTest = HloTestBase;

// Check basic parameter matching
TEST_F(ConvolutionClassifierTest, Training1) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(arg_0, broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(arg_0.1, broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(constant.19.10.clone.3), dimensions={}
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] custom-call(arg_0.3, arg_1.3), custom_call_target="ConvWithReverse", window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(arg_0.4, arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(arg_0.5, arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(arg_0.6, arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(arg_0.7, arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

_arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(arg_1.8, arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(divide.19.48.clone, arg_0.8)
}

_cluster_1  {
  arg2.19.2 = f16[4] parameter(2)
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  arg1.19.1 = f16[1,1024] parameter(1)
  convert.19.11 = f32[1,1024] convert(arg1.19.1)
  arg0.19.0 = f16[1,16,16,4] parameter(0)
  arg5.19.5 = f16[7,7,4,64] parameter(5)
  call.9 = f16[1,16,16,64] call(arg0.19.0, arg5.19.5), to_apply=pop_convolution.1
  arg4.19.4 = f16[64] parameter(4)
  call.1 = f16[1,16,16,64] fusion(call.9, arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  arg3.19.3 = f16[5,5,64,4] parameter(3)
  call.8 = f16[1,16,16,4] call(call.1, arg3.19.3), to_apply=pop_convolution
  call = f16[1,16,16,4] fusion(call.8, arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(call)
  reshape = f32[1,1024] reshape(convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(reshape, broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] fusion(convert.19.11, exponential.19.33, broadcast.19.47), kind=kCustom, calls=_arithmetic_expression
  convert.19.50 = f16[1,1024] convert(call.12)
  convert.1 = f32[1,1024] convert(convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(convert.1)
  reduce.19.61 = f32[4] reduce(reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(reduce.19.61)
  multiply.19.64 = f16[4] multiply(call.3, convert.19.62)
  subtract.19.65 = f16[4] subtract(arg2.19.2, multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(convert.1)
  reshape.10 = f16[1,16,16,4] reshape(convert.3)
  call.10 = f16[5,5,64,4] call(call.1, reshape.10), to_apply=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(call.4, call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(arg3.19.3, multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] call(reshape.10, arg3.19.3), to_apply=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(call.7)
  reduce.19.81 = f32[64] reduce(convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(reduce.19.81)
  multiply.19.84 = f16[64] multiply(call.5, convert.19.82)
  subtract.19.85 = f16[64] subtract(arg4.19.4, multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(convert.19.78)
  call.11 = f16[7,7,4,64] call(arg0.19.0, convert.2), to_apply=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(call.6, call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(arg5.19.5, multiply.19.88)
  ROOT tuple.19.98 = (f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(subtract.19.65, subtract.19.71, subtract.19.85, subtract.19.89)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  CustomOpReplacer replacer;
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(replacer.Run(module).ValueOrDie());
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "conv-with-reverse") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing convolutions";
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInRepeat) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(arg_0, broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(arg_0.1, broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(constant.19.10.clone.3), dimensions={}
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] custom-call(arg_0.3, arg_1.3), custom_call_target="ConvWithReverse", window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(arg_0.4, arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(arg_0.5, arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(arg_0.6, arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(arg_0.7, arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

_arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(arg_1.8, arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(divide.19.48.clone, arg_0.8)
}

_loop_body {
  p = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(0)
  counter = s32[] get-tuple-element(p), index=0
  arg0.19.0 = f16[1,16,16,4] get-tuple-element(p), index=1
  arg1.19.1 = f16[1,1024] get-tuple-element(p), index=2
  arg2.19.2 = f16[4] get-tuple-element(p), index=3
  arg3.19.3 = f16[5,5,64,4] get-tuple-element(p), index=4
  arg4.19.4 = f16[64] get-tuple-element(p), index=5
  arg5.19.5 = f16[7,7,4,64] get-tuple-element(p), index=6
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  convert.19.11 = f32[1,1024] convert(arg1.19.1)
  call.9 = f16[1,16,16,64] call(arg0.19.0, arg5.19.5), to_apply=pop_convolution.1
  call.1 = f16[1,16,16,64] fusion(call.9, arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  call.8 = f16[1,16,16,4] call(call.1, arg3.19.3), to_apply=pop_convolution
  call = f16[1,16,16,4] fusion(call.8, arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(call)
  reshape = f32[1,1024] reshape(convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(reshape, broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] fusion(convert.19.11, exponential.19.33, broadcast.19.47), kind=kCustom, calls=_arithmetic_expression
  convert.19.50 = f16[1,1024] convert(call.12)
  convert.1 = f32[1,1024] convert(convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(convert.1)
  reduce.19.61 = f32[4] reduce(reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(reduce.19.61)
  multiply.19.64 = f16[4] multiply(call.3, convert.19.62)
  subtract.19.65 = f16[4] subtract(arg2.19.2, multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(convert.1)
  reshape.10 = f16[1,16,16,4] reshape(convert.3)
  call.10 = f16[5,5,64,4] call(call.1, reshape.10), to_apply=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(call.4, call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(arg3.19.3, multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] call(reshape.10, arg3.19.3), to_apply=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(call.7)
  reduce.19.81 = f32[64] reduce(convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(reduce.19.81)
  multiply.19.84 = f16[64] multiply(call.5, convert.19.82)
  subtract.19.85 = f16[64] subtract(arg4.19.4, multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(convert.19.78)
  call.11 = f16[7,7,4,64] call(arg0.19.0, convert.2), to_apply=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(call.6, call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(arg5.19.5, multiply.19.88)
  ROOT tuple.19.98 = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(counter, arg0.19.0, arg1.19.1, subtract.19.65, subtract.19.71, subtract.19.85, subtract.19.89)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(1)
  ROOT call = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(input_tuple), to_apply=_loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f16[1,16,16,4] parameter(0)
  p1 = f16[1,1024] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[5,5,64,4] parameter(3)
  p4 = f16[64] parameter(4)
  p5 = f16[7,7,4,64] parameter(5)
  in = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(c, p0, p1, p2, p3, p4, p5)
  ROOT r0 = (s32[], f16[1,16,16,4], f16[1,1024], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  CustomOpReplacer replacer;
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(replacer.Run(module).ValueOrDie());
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "conv-with-reverse") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing convolutions";
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInfeedInRepeat) {
  std::string hlo_string = R"(
HloModule top

add14 {
  x.14.0 = f32[] parameter(0)
  y.14.1 = f32[] parameter(1)
  ROOT add.14.2 = f32[] add(x.14.0, y.14.1)
}

max13 {
  x.13.0 = f32[] parameter(0)
  y.13.1 = f32[] parameter(1)
  ROOT maximum.13.2 = f32[] maximum(x.13.0, y.13.1)
}

_pop_op_biasadd {
  arg_0 = f16[1,16,16,4] parameter(0)
  arg_1 = f16[4] parameter(1)
  broadcast.19.18.clone = f16[1,16,16,4] broadcast(arg_1), dimensions={3}
  ROOT add.19.19.clone = f16[1,16,16,4] add(arg_0, broadcast.19.18.clone)
}

_pop_op_biasadd.1 {
  arg_0.1 = f16[1,16,16,64] parameter(0)
  arg_1.1 = f16[64] parameter(1)
  broadcast.19.15.clone = f16[1,16,16,64] broadcast(arg_1.1), dimensions={3}
  ROOT add.19.16.clone = f16[1,16,16,64] add(arg_0.1, broadcast.19.15.clone)
}

_pop_op_wide_const {
  constant.19.10.clone = f16[] constant(0.0100021362)
  ROOT broadcast.19.63.clone = f16[4] broadcast(constant.19.10.clone), dimensions={}
}

_pop_op_wide_const.1  {
  constant.19.10.clone.1 = f16[] constant(0.0100021362)
  ROOT broadcast.19.69.clone = f16[5,5,64,4] broadcast(constant.19.10.clone.1), dimensions={}
}

_pop_op_wide_const.2  {
  constant.19.10.clone.2 = f16[] constant(0.0100021362)
  ROOT broadcast.19.83.clone = f16[64] broadcast(constant.19.10.clone.2), dimensions={}
}

_pop_op_wide_const.3 {
  constant.19.10.clone.3 = f16[] constant(0.0100021362)
  ROOT broadcast.19.87.clone = f16[7,7,4,64] broadcast(constant.19.10.clone.3), dimensions={}
}

pop_backprop_conv {
  arg_0.3 = f16[1,16,16,4] parameter(0)
  arg_1.3 = f16[5,5,64,4] parameter(1)
  ROOT call.2.clone = f16[1,16,16,64] custom-call(arg_0.3, arg_1.3), custom_call_target="ConvWithReverse", window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f
}

pop_convolution {
  arg_0.4 = f16[1,16,16,64] parameter(0)
  arg_1.4 = f16[5,5,64,4] parameter(1)
  ROOT convolution.19.17.clone = f16[1,16,16,4] convolution(arg_0.4, arg_1.4), window={size=5x5 pad=2_2x2_2}, dim_labels=b01f_01io->b01f
}

pop_convolution.1 {
  arg_0.5 = f16[1,16,16,4] parameter(0)
  arg_1.5 = f16[7,7,4,64] parameter(1)
  ROOT convolution.19.14.clone = f16[1,16,16,64] convolution(arg_0.5, arg_1.5), window={size=7x7 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
}

pop_convolution.2 {
  arg_0.6 = f16[1,16,16,64] parameter(0)
  arg_1.6 = f16[1,16,16,4] parameter(1)
  ROOT convolution.19.66.clone = f16[5,5,64,4] convolution(arg_0.6, arg_1.6), window={size=16x16 pad=2_2x2_2}, dim_labels=f01b_i01o->01bf
}

pop_convolution.3 {
  arg_0.7 = f16[1,16,16,4] parameter(0)
  arg_1.7 = f16[1,16,16,64] parameter(1)
  ROOT convolution.19.86.clone = f16[7,7,4,64] convolution(arg_0.7, arg_1.7), window={size=16x16 pad=3_3x3_3}, dim_labels=f01b_i01o->01bf
}

_arithmetic_expression {
  arg_1.8 = f32[1,1024] parameter(1)
  arg_2 = f32[1,1024] parameter(2)
  divide.19.48.clone = f32[1,1024] divide(arg_1.8, arg_2)
  arg_0.8 = f32[1,1024] parameter(0)
  ROOT subtract.19.49.clone = f32[1,1024] subtract(divide.19.48.clone, arg_0.8)
}

_loop_body {
  p = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(0)
  counter = s32[] get-tuple-element(p), index=0
  arg2.19.2 = f16[4] get-tuple-element(p), index=1
  arg3.19.3 = f16[5,5,64,4] get-tuple-element(p), index=2
  arg4.19.4 = f16[64] get-tuple-element(p), index=3
  arg5.19.5 = f16[7,7,4,64] get-tuple-element(p), index=4
  after-all = token[] after-all()
  infeed = ((f16[1,16,16,4], f16[1,1024]), token[]) infeed(after-all), infeed_config="01234567"
  infeed_tuple = (f16[1,16,16,4], f16[1,1024]) get-tuple-element(infeed), index=0
  arg0.19.0 = f16[1,16,16,4] get-tuple-element(infeed_tuple), index=0
  arg1.19.1 = f16[1,1024] get-tuple-element(infeed_tuple), index=1
  call.3 = f16[4] fusion(), kind=kCustom, calls=_pop_op_wide_const
  convert.19.11 = f32[1,1024] convert(arg1.19.1)
  call.9 = f16[1,16,16,64] call(arg0.19.0, arg5.19.5), to_apply=pop_convolution.1
  call.1 = f16[1,16,16,64] fusion(call.9, arg4.19.4), kind=kCustom, calls=_pop_op_biasadd.1
  call.8 = f16[1,16,16,4] call(call.1, arg3.19.3), to_apply=pop_convolution
  call = f16[1,16,16,4] fusion(call.8, arg2.19.2), kind=kCustom, calls=_pop_op_biasadd
  convert = f32[1,16,16,4] convert(call)
  reshape = f32[1,1024] reshape(convert)
  constant.19.29 = f32[] constant(-inf)
  reduce = f32[1] reduce(convert, constant.19.29), dimensions={1,2,3}, to_apply=max13
  broadcast.19.31 = f32[1,1024] broadcast(reduce), dimensions={0}
  subtract.19.32 = f32[1,1024] subtract(reshape, broadcast.19.31)
  exponential.19.33 = f32[1,1024] exponential(subtract.19.32)
  constant.19.35 = f32[] constant(0)
  reduce.19.36 = f32[1] reduce(exponential.19.33, constant.19.35), dimensions={1}, to_apply=add14
  broadcast.19.47 = f32[1,1024] broadcast(reduce.19.36), dimensions={0}
  call.12 = f32[1,1024] fusion(convert.19.11, exponential.19.33, broadcast.19.47), kind=kCustom, calls=_arithmetic_expression
  convert.19.50 = f16[1,1024] convert(call.12)
  convert.1 = f32[1,1024] convert(convert.19.50)
  reshape.1 = f32[1,16,16,4] reshape(convert.1)
  reduce.19.61 = f32[4] reduce(reshape.1, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.62 = f16[4] convert(reduce.19.61)
  multiply.19.64 = f16[4] multiply(call.3, convert.19.62)
  subtract.19.65 = f16[4] subtract(arg2.19.2, multiply.19.64)
  call.4 = f16[5,5,64,4] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  convert.3 = f16[1,1024] convert(convert.1)
  reshape.10 = f16[1,16,16,4] reshape(convert.3)
  call.10 = f16[5,5,64,4] call(call.1, reshape.10), to_apply=pop_convolution.2
  multiply.19.70 = f16[5,5,64,4] multiply(call.4, call.10)
  subtract.19.71 = f16[5,5,64,4] subtract(arg3.19.3, multiply.19.70)
  call.5 = f16[64] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
  call.7 = f16[1,16,16,64] call(reshape.10, arg3.19.3), to_apply=pop_backprop_conv
  convert.19.78 = f32[1,16,16,64] convert(call.7)
  reduce.19.81 = f32[64] reduce(convert.19.78, constant.19.35), dimensions={0,1,2}, to_apply=add14
  convert.19.82 = f16[64] convert(reduce.19.81)
  multiply.19.84 = f16[64] multiply(call.5, convert.19.82)
  subtract.19.85 = f16[64] subtract(arg4.19.4, multiply.19.84)
  call.6 = f16[7,7,4,64] fusion(), kind=kCustom, calls=_pop_op_wide_const.3
  convert.2 = f16[1,16,16,64] convert(convert.19.78)
  call.11 = f16[7,7,4,64] call(arg0.19.0, convert.2), to_apply=pop_convolution.3
  multiply.19.88 = f16[7,7,4,64] multiply(call.6, call.11)
  subtract.19.89 = f16[7,7,4,64] subtract(arg5.19.5, multiply.19.88)
  ROOT tuple.19.98 = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(counter, subtract.19.65, subtract.19.71, subtract.19.85, subtract.19.89)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) parameter(1)
  ROOT call = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(input_tuple), to_apply=_loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f16[4] parameter(0)
  p1 = f16[5,5,64,4] parameter(1)
  p2 = f16[64] parameter(2)
  p3 = f16[7,7,4,64] parameter(3)
  in = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) tuple(c, p0, p1, p2, p3)
  ROOT r0 = (s32[], f16[4], f16[5,5,64,4], f16[64], f16[7,7,4,64]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3, 4, 5});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  CustomOpReplacer replacer;
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(replacer.Run(module).ValueOrDie());
  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "conv-with-reverse") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "convolution.19.17.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.14.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "convolution.19.66.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "convolution.19.86.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing convolutions";
    }
  }
}

TEST_F(ConvolutionClassifierTest, SingleConvTraining) {
  std::string hlo_string = R"(
  HloModule top

  Mean-reduction4 {
    x.4.0 = f32[] parameter(0)
    y.4.1 = f32[] parameter(1)
    ROOT add.4.2 = f32[] add(x.4.0, y.4.1)
  }

  max5 {
    x.5.0 = f32[] parameter(0)
    y.5.1 = f32[] parameter(1)
    ROOT maximum.5.2 = f32[] maximum(x.5.0, y.5.1)
  }

  _pop_op_wide_const {
    constant.7.6.clone = f32[] constant(0.01)
    ROOT broadcast.7.52.clone = f32[3,3,4,12] broadcast(constant.7.6.clone), dimensions={}
  }

  _pop_op_wide_const.1 {
    constant.7.19.clone = f32[] constant(576)
    ROOT broadcast.7.20.clone = f32[1,12] broadcast(constant.7.19.clone), dimensions={}
  }

  _pop_op_wide_const.2 {
    constant.7.5.clone = f32[] constant(0.00173611112)
    ROOT broadcast.7.49.clone = f32[1,24,24,12] broadcast(constant.7.5.clone), dimensions={}
  }

  pop_convolution {
    arg_0 = f32[1,24,24,4] parameter(0)
    arg_1 = f32[1,24,24,12] parameter(1)
    ROOT convolution.7.51.clone = f32[3,3,4,12] convolution(arg_0, arg_1), window={size=24x24 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf
  }

  pop_convolution.1 {
    arg_0.1 = f32[1,24,24,4] parameter(0)
    arg_1.1 = f32[3,3,4,12] parameter(1)
    ROOT convolution.7.13.clone = f32[1,24,24,12] convolution(arg_0.1, arg_1.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  }

  _arithmetic_expression {
    arg_1.2 = f32[1,12] parameter(1)
    arg_0.2 = f32[1,12] parameter(0)
    divide.7.41.clone = f32[1,12] divide(arg_1.2, arg_0.2)
    arg_2 = f32[1,12] parameter(2)
    ROOT subtract.7.42.clone = f32[1,12] subtract(divide.7.41.clone, arg_2)
  }

  ENTRY cluster_1 {
    arg2.7.2 = f32[3,3,4,12] parameter(2)
    call = f32[3,3,4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
    arg0.7.0 = f32[1,24,24,4] parameter(0)
    call.4 = f32[1,24,24,12] call(arg0.7.0, arg2.7.2), to_apply=pop_convolution.1
    constant.7.15 = f32[] constant(0)
    reduce.7.17 = f32[1,12] reduce(call.4, constant.7.15), dimensions={1,2}, to_apply=Mean-reduction4
    call.1 = f32[1,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
    divide.7.21 = f32[1,12] divide(reduce.7.17, call.1)
    constant.7.22 = f32[] constant(-inf)
    reduce.7.23 = f32[1] reduce(divide.7.21, constant.7.22), dimensions={1}, to_apply=max5
    broadcast.7.24 = f32[1,12] broadcast(reduce.7.23), dimensions={0}
    subtract.7.25 = f32[1,12] subtract(divide.7.21, broadcast.7.24)
    exponential.7.26 = f32[1,12] exponential(subtract.7.25)
    reduce.7.29 = f32[1] reduce(exponential.7.26, constant.7.15), dimensions={1}, to_apply=Mean-reduction4
    broadcast.7.40 = f32[1,12] broadcast(reduce.7.29), dimensions={0}
    arg1.7.1 = f32[1,12] parameter(1)
    call.5 = f32[1,12] fusion(broadcast.7.40, exponential.7.26, arg1.7.1), kind=kCustom, calls=_arithmetic_expression
    broadcast = f32[1,24,24,12] broadcast(call.5), dimensions={0,3}
    call.2 = f32[1,24,24,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.2
    multiply.7.50 = f32[1,24,24,12] multiply(broadcast, call.2)
    call.3 = f32[3,3,4,12] call(arg0.7.0, multiply.7.50), to_apply=pop_convolution
    multiply.7.53 = f32[3,3,4,12] multiply(call, call.3)
    subtract.7.54 = f32[3,3,4,12] subtract(arg2.7.2, multiply.7.53)
    ROOT tuple.7.57 = (f32[3,3,4,12]) tuple(subtract.7.54)
  }
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2});
  config.set_resource_input_count(1);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 2);

  for (auto it : all_classifications) {
    if (it.first->name() == "convolution.7.51.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "convolution.7.13.clone") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else {
      FAIL() << "We should not have any missing convolutions";
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingMatMul) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2)
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  arg0.9.0 = f32[1,4] parameter(0)
  arg3.9.3 = f32[4,12] parameter(3)
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  arg1.9.1 = f32[1,12] parameter(1)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50)
}

)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  config.set_resource_input_count(2);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingMatMulInRepeat) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

loop_body {
  p0 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) parameter(0)
  counter = s32[] get-tuple-element(p0), index=0
  arg0.9.0 = f32[1,4] get-tuple-element(p0), index=1
  arg1.9.1 = f32[1,12] get-tuple-element(p0), index=2
  arg2.9.2 = f32[12,12] get-tuple-element(p0), index=3
  arg3.9.3 = f32[4,12] get-tuple-element(p0), index=4
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) tuple(counter, arg0.9.0, arg1.9.1, subtract.9.41, subtract.9.50)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) parameter(1)
  ROOT call = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f32[1,4] parameter(0)
  p1 = f32[1,12] parameter(1)
  p2 = f32[12,12] parameter(2)
  p3 = f32[4,12] parameter(3)
  in = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) tuple(c, p0, p1, p2, p3)
  ROOT r0 = (s32[], f32[1,4], f32[1,12], f32[12,12], f32[4,12]) call(c, in), to_apply=__repeat
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  config.set_resource_input_count(2);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, TrainingInfeedMatMulInRepeat) {
  std::string hlo_string = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

loop_body {
  p0 = (s32[], f32[12,12], f32[4,12]) parameter(0)
  counter = s32[] get-tuple-element(p0), index=0
  arg2.9.2 = f32[12,12] get-tuple-element(p0), index=1
  arg3.9.3 = f32[4,12] get-tuple-element(p0), index=2
  after-all = token[] after-all()
  infeed = ((f32[1,4], f32[1,12]), token[]) infeed(after-all), infeed_config="01234567"
  infeed_tuple = (f32[1,4], f32[1,12]) get-tuple-element(infeed), index=0
  arg0.9.0 = f32[1,4] get-tuple-element(infeed_tuple), index=0
  arg1.9.1 = f32[1,12] get-tuple-element(infeed_tuple), index=1
  call.2 = f32[12,12] fusion(), kind=kCustom, calls=_pop_op_wide_const
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] fusion(dot.9.9), kind=kCustom, calls=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] fusion(), kind=kCustom, calls=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] fusion(call, dot.9.38), kind=kCustom, calls=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (s32[], f32[12,12], f32[4,12]) tuple(counter, subtract.9.41, subtract.9.50)
}

__repeat {
  repeat_count = s32[] parameter(0)
  input_tuple = (s32[], f32[12,12], f32[4,12]) parameter(1)
  ROOT call = (s32[], f32[12,12], f32[4,12]) call(input_tuple), to_apply=loop_body
}

ENTRY in {
  c = s32[] constant(10)
  p0 = f32[12,12] parameter(0)
  p1 = f32[4,12] parameter(1)
  in = (s32[], f32[12,12], f32[4,12]) tuple(c, p0, p1)
  ROOT r0 = (s32[], f32[12,12], f32[4,12]) call(c, in), to_apply=__repeat
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({2, 3});
  config.set_resource_input_count(2);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 5);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot.9.9") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.13") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD);
    } else if (it.first->name() == "dot.9.36") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else if (it.first->name() == "dot.9.38") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD);
    } else if (it.first->name() == "dot.9.47") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU);
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, InferenceMatMul) {
  std::string hlo_string = R"(
HloModule top

_pop_op_relu {
  constant.17.9.clone = f32[] constant(0)
  broadcast.17.10.clone = f32[32,32] broadcast(constant.17.9.clone), dimensions={}
  arg_0 = f32[32,32] parameter(0)
  ROOT maximum.17.11.clone = f32[32,32] maximum(broadcast.17.10.clone, arg_0)
}

_pop_op_sigmoid {
  constant.17.15.clone = f32[] constant(0.5)
  broadcast.17.21.clone = f32[32,1] broadcast(constant.17.15.clone), dimensions={}
  arg_0.1 = f32[32,1] parameter(0)
  multiply.17.17.clone = f32[32,1] multiply(broadcast.17.21.clone, arg_0.1)
  tanh.17.18.clone = f32[32,1] tanh(multiply.17.17.clone)
  multiply.17.20.clone = f32[32,1] multiply(broadcast.17.21.clone, tanh.17.18.clone)
  ROOT add.17.22.clone = f32[32,1] add(broadcast.17.21.clone, multiply.17.20.clone)
}

ENTRY cluster_9 {
  arg0.17.0 = f32[32,100] parameter(0)
  arg4.17.4 = f32[100,32] parameter(4)
  dot.17.6 = f32[32,32] dot(arg0.17.0, arg4.17.4), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg3.17.3 = f32[32] parameter(3)
  broadcast.17.7 = f32[32,32] broadcast(arg3.17.3), dimensions={1}
  add.17.8 = f32[32,32] add(dot.17.6, broadcast.17.7)
  call = f32[32,32] fusion(add.17.8), kind=kCustom, calls=_pop_op_relu
  arg2.17.2 = f32[32,1] parameter(2)
  dot.17.12 = f32[32,1] dot(call, arg2.17.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg1.17.1 = f32[1] parameter(1)
  broadcast.17.13 = f32[32,1] broadcast(arg1.17.1), dimensions={1}
  add.17.14 = f32[32,1] add(dot.17.12, broadcast.17.13)
  call.1 = f32[32,1] fusion(add.17.14), kind=kCustom, calls=_pop_op_sigmoid
  ROOT tuple.17.24 = (f32[32,1]) tuple(call.1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 2);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot.17.12") {
      EXPECT_EQ(it.second, MLType::INFERENCE_FWD);
    } else if (it.first->name() == "dot.17.6") {
      EXPECT_EQ(it.second, MLType::INFERENCE_FWD);
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, MatMulWithCastWeights) {
  std::string hlo_string = R"(HloModule module

max_half_.50 {
  x.51 = f16[] parameter(0)
  y.52 = f16[] parameter(1)
  ROOT maximum.53 = f16[] maximum(x.51, y.52), backend_config="{\"isInplace\":true}"
}

add_float_.60 {
  x.61 = f32[] parameter(0)
  y.62 = f32[] parameter(1)
  ROOT add.63 = f32[] add(x.61, y.62), backend_config="{\"isInplace\":true}"
}

_pop_op_matmul_biasadd {
  arg_0 = f16[1024,3000] parameter(0)
  arg_1 = f16[3000] parameter(1)
  broadcast.23.clone = f16[1024,3000] broadcast(arg_1), dimensions={1}
  ROOT add.24.clone = f16[1024,3000] add(arg_0, broadcast.23.clone)
}

_pop_op_matmul_biasadd.1 {
  arg_0.1 = f16[1024,3000] parameter(0)
  arg_1.1 = f16[3000] parameter(1)
  broadcast.18.clone = f16[1024,3000] broadcast(arg_1.1), dimensions={1}
  ROOT add.19.clone = f16[1024,3000] add(arg_0.1, broadcast.18.clone)
}

_pop_op_scaled_inplace {
  arg_0.2 = f32[128,3000] parameter(0)
  arg_1.2 = f32[128,3000] parameter(1)
  constant.103.clone = f32[] constant(0.001)
  broadcast.138.clone = f32[128,3000] broadcast(constant.103.clone), dimensions={}
  multiply.175.clone = f32[128,3000] multiply(arg_1.2, broadcast.138.clone)
  ROOT subtract.176.clone = f32[128,3000] subtract(arg_0.2, multiply.175.clone)
}

_pop_op_scaled_inplace.1 {
  arg_0.3 = f32[3000] parameter(0)
  arg_1.3 = f32[3000] parameter(1)
  constant.103.clone.1 = f32[] constant(0.001)
  broadcast.123.clone = f32[3000] broadcast(constant.103.clone.1), dimensions={}
  multiply.160.clone = f32[3000] multiply(arg_1.3, broadcast.123.clone)
  ROOT subtract.161.clone = f32[3000] subtract(arg_0.3, multiply.160.clone)
}

_pop_op_scaled_inplace.2 {
  arg_0.4 = f32[128,3000] parameter(0)
  arg_1.4 = f32[128,3000] parameter(1)
  constant.103.clone.2 = f32[] constant(0.001)
  broadcast.138.clone.1 = f32[128,3000] broadcast(constant.103.clone.2), dimensions={}
  multiply.139.clone = f32[128,3000] multiply(arg_1.4, broadcast.138.clone.1)
  ROOT subtract.140.clone = f32[128,3000] subtract(arg_0.4, multiply.139.clone)
}

_pop_op_scaled_inplace.3 {
  arg_0.5 = f32[3000] parameter(0)
  arg_1.5 = f32[3000] parameter(1)
  constant.103.clone.3 = f32[] constant(0.001)
  broadcast.123.clone.1 = f32[3000] broadcast(constant.103.clone.3), dimensions={}
  multiply.124.clone = f32[3000] multiply(arg_1.5, broadcast.123.clone.1)
  ROOT subtract.125.clone = f32[3000] subtract(arg_0.5, multiply.124.clone)
}

_pop_op_scaled_inplace.4 {
  arg_0.6 = f32[1024,1024] parameter(0)
  arg_1.6 = f32[1024,1024] parameter(1)
  constant.103.clone.4 = f32[] constant(0.001)
  broadcast.104.clone = f32[1024,1024] broadcast(constant.103.clone.4), dimensions={}
  multiply.105.clone = f32[1024,1024] multiply(arg_1.6, broadcast.104.clone)
  ROOT subtract.106.clone = f32[1024,1024] subtract(arg_0.6, multiply.105.clone)
}

_pop_op_implicit_binary {
  constant.42.clone = s32[] constant(0)
  broadcast.3 = s32[1024] broadcast(constant.42.clone), dimensions={}
  arg_0.7 = s32[1024] parameter(0)
  ROOT compare = pred[1024] compare(broadcast.3, arg_0.7), direction=LE
}

_pop_op_implicit_binary_inplace {
  arg_0.8 = f16[1024,3000] parameter(0)
  arg_1.7 = f16[1024] parameter(1)
  broadcast.4 = f16[1024,3000] broadcast(arg_1.7), dimensions={0}
  ROOT subtract = f16[1024,3000] subtract(arg_0.8, broadcast.4)
}

_pop_op_implicit_binary_inplace.1 {
  arg_0.9 = f16[1024,3000] parameter(0)
  arg_1.8 = f16[1024] parameter(1)
  broadcast.5 = f16[1024,3000] broadcast(arg_1.8), dimensions={0}
  ROOT divide = f16[1024,3000] divide(arg_0.9, broadcast.5)
}

_pop_op_implicit_binary_inplace.2 {
  arg_0.10 = f16[1024,3000] parameter(0)
  arg_1.9 = f16[1024] parameter(1)
  broadcast.6 = f16[1024,3000] broadcast(arg_1.9), dimensions={0}
  ROOT subtract.1 = f16[1024,3000] subtract(arg_0.10, broadcast.6)
}

_pop_op_implicit_binary_inplace.3 {
  arg_0.11 = f16[1024,3000] parameter(0)
  arg_1.10 = f16[1024] parameter(1)
  broadcast.7 = f16[1024,3000] broadcast(arg_1.10), dimensions={0}
  ROOT add = f16[1024,3000] add(arg_0.11, broadcast.7)
}

_pop_op_implicit_binary.1 {
  arg_0.12 = s32[1024] parameter(0)
  broadcast.8 = s32[1024,3000] broadcast(arg_0.12), dimensions={0}
  arg_1.11 = s32[1024,3000] parameter(1)
  ROOT compare.1 = pred[1024,3000] compare(broadcast.8, arg_1.11), direction=EQ
}

_pop_op_implicit_binary.2 {
  arg_0.13 = s32[1024] parameter(0)
  constant.39.clone = s32[] constant(3000)
  broadcast.9 = s32[1024] broadcast(constant.39.clone), dimensions={}
  ROOT compare.2 = pred[1024] compare(arg_0.13, broadcast.9), direction=LT
}

_pop_op_implicit_ternary {
  arg_0.14 = pred[1024] parameter(0)
  constant.8.clone = f16[] constant(0)
  broadcast.10 = f16[1024] broadcast(constant.8.clone), dimensions={}
  constant.10.clone = f16[] constant(nan)
  broadcast.11 = f16[1024] broadcast(constant.10.clone), dimensions={}
  ROOT select = f16[1024] select(arg_0.14, broadcast.10, broadcast.11)
}

_pop_op_implicit_ternary.1 {
  arg_0.15 = pred[1024,3000] parameter(0)
  constant.7.clone = f16[] constant(1)
  broadcast.12 = f16[1024,3000] broadcast(constant.7.clone), dimensions={}
  constant.8.clone.1 = f16[] constant(0)
  broadcast.13 = f16[1024,3000] broadcast(constant.8.clone.1), dimensions={}
  ROOT select.1 = f16[1024,3000] select(arg_0.15, broadcast.12, broadcast.13)
}

ENTRY cluster {
  arg3.4 = f32[1024,1024] parameter(3), parameter_replication={false}
  convert.12 = f16[1024,1024] convert(arg3.4)
  arg1.2 = f16[1024,128] parameter(1), parameter_replication={false}
  arg5.6 = f32[128,3000] parameter(5), parameter_replication={false}
  convert.16 = f16[128,3000] convert(arg5.6)
  dot.17 = f16[1024,3000] dot(arg1.2, convert.16), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg4.5 = f32[3000] parameter(4), parameter_replication={false}, control-predecessors={dot.17}
  convert.15 = f16[3000] convert(arg4.5)
  fusion.1 = f16[1024,3000] fusion(dot.17, convert.15), kind=kCustom, calls=_pop_op_matmul_biasadd.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  tanh.20 = f16[1024,3000] tanh(fusion.1), backend_config="{\"isInplace\":true}"
  dot.21 = f16[1024,3000] dot(convert.12, tanh.20), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg2.3 = f16[1024,128] parameter(2), parameter_replication={false}
  arg7.8 = f32[128,3000] parameter(7), parameter_replication={false}
  convert.14 = f16[128,3000] convert(arg7.8)
  dot.22 = f16[1024,3000] dot(arg2.3, convert.14), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  arg6.7 = f32[3000] parameter(6), parameter_replication={false}, control-predecessors={dot.22}
  convert.13 = f16[3000] convert(arg6.7)
  fusion = f16[1024,3000] fusion(dot.22, convert.13), kind=kCustom, calls=_pop_op_matmul_biasadd, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  tanh.25 = f16[1024,3000] tanh(fusion), backend_config="{\"isInplace\":true}"
  add.26 = f16[1024,3000] add(dot.21, tanh.25), backend_config="{\"isInplace\":true}"
  constant.11 = f16[] constant(-inf)
  reduce.54 = f16[1024] reduce(add.26, constant.11), dimensions={1}, to_apply=max_half_.50
  fusion.8 = f16[1024,3000] fusion(add.26, reduce.54), kind=kCustom, calls=_pop_op_implicit_binary_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  exponential.57 = f16[1024,3000] exponential(fusion.8)
  constant.8 = f16[] constant(0)
  reduce.64.clone = f16[1024] reduce(exponential.57, constant.8), dimensions={1}, to_apply=add_float_.60
  fusion.9 = f16[1024,3000] fusion(exponential.57, reduce.64.clone), kind=kCustom, calls=_pop_op_implicit_binary_inplace.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  arg0.1 = s32[1024] parameter(0), parameter_replication={false}
  iota.31 = s32[1024,3000] iota(), iota_dimension=1
  fusion.12 = pred[1024,3000] fusion(arg0.1, iota.31), kind=kCustom, calls=_pop_op_implicit_binary.1, backend_config="{\"fusionConfig\":{}}"
  fusion.15 = f16[1024,3000] fusion(fusion.12), kind=kCustom, calls=_pop_op_implicit_ternary.1, backend_config="{\"fusionConfig\":{}}"
  fusion.7 = pred[1024] fusion(arg0.1), kind=kCustom, calls=_pop_op_implicit_binary, backend_config="{\"fusionConfig\":{}}"
  fusion.13 = pred[1024] fusion(arg0.1), kind=kCustom, calls=_pop_op_implicit_binary.2, backend_config="{\"fusionConfig\":{}}"
  and.45 = pred[1024] and(fusion.7, fusion.13), backend_config="{\"isInplace\":true}"
  fusion.14 = f16[1024] fusion(and.45), kind=kCustom, calls=_pop_op_implicit_ternary, backend_config="{\"fusionConfig\":{}}"
  fusion.11 = f16[1024,3000] fusion(fusion.15, fusion.14), kind=kCustom, calls=_pop_op_implicit_binary_inplace.3, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  subtract.81 = f16[1024,3000] subtract(fusion.9, fusion.11), backend_config="{\"isInplace\":true}"
  negate.69 = f16[1024,3000] negate(fusion.11), control-predecessors={subtract.81}, backend_config="{\"isInplace\":true}"
  log.66 = f16[1024] log(reduce.64.clone), control-predecessors={fusion.9}, backend_config="{\"isInplace\":true}"
  fusion.10 = f16[1024,3000] fusion(fusion.8, log.66), kind=kCustom, calls=_pop_op_implicit_binary_inplace.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  multiply.70 = f16[1024,3000] multiply(negate.69, fusion.10), backend_config="{\"isInplace\":true}"
  reduce = f16[] reduce(multiply.70, constant.8), dimensions={0,1}, to_apply=add_float_.60
  transpose.95 = f16[3000,1024] transpose(tanh.20), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.96 = f16[1024,1024] dot(subtract.81, transpose.95), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  convert.102 = f32[1024,1024] convert(dot.96)
  fusion.6 = f32[1024,1024] fusion(arg3.4, convert.102), kind=kCustom, calls=_pop_op_scaled_inplace.4, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  transpose.98 = f16[1024,1024] transpose(convert.12), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.99 = f16[1024,3000] dot(transpose.98, subtract.81), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  custom-call = f16[1024,3000] custom-call(tanh.20, dot.99), custom_call_target="TanhGrad"
  reduce.117.clone = f16[3000] reduce(custom-call, constant.8), dimensions={0}, to_apply=add_float_.60
  convert.121 = f32[3000] convert(reduce.117.clone)
  fusion.5 = f32[3000] fusion(arg4.5, convert.121), kind=kCustom, calls=_pop_op_scaled_inplace.3, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  transpose.132 = f16[128,1024] transpose(arg1.2), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.133 = f16[128,3000] dot(transpose.132, custom-call), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  convert.136 = f32[128,3000] convert(dot.133)
  fusion.4 = f32[128,3000] fusion(arg5.6, convert.136), kind=kCustom, calls=_pop_op_scaled_inplace.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  custom-call.1 = f16[1024,3000] custom-call(tanh.25, subtract.81), custom_call_target="TanhGrad"
  reduce.153.clone = f16[3000] reduce(custom-call.1, constant.8), dimensions={0}, to_apply=add_float_.60
  convert.157 = f32[3000] convert(reduce.153.clone)
  fusion.3 = f32[3000] fusion(arg6.7, convert.157), kind=kCustom, calls=_pop_op_scaled_inplace.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  transpose.168 = f16[128,1024] transpose(arg2.3), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.169 = f16[128,3000] dot(transpose.168, custom-call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  convert.172 = f32[128,3000] convert(dot.169)
  fusion.2 = f32[128,3000] fusion(arg7.8, convert.172), kind=kCustom, calls=_pop_op_scaled_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  ROOT tuple.190 = (f16[], f32[1024,1024], f32[3000], f32[128,3000], f32[3000], f32[128,3000]) tuple(reduce, fusion.6, fusion.5, fusion.4, fusion.3, fusion.2), backend_config="{\"isInplace\":true}"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 7);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot.17" || it.first->name() == "dot.21" ||
        it.first->name() == "dot.22") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD) << it.first->name();
    } else if (it.first->name() == "dot.96") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD) << it.first->name();
    } else if (it.first->name() == "dot.99" || it.first->name() == "dot.133" ||
               it.first->name() == "dot.169") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU) << it.first->name();
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, MatMulBatchsize1) {
  std::string hlo_string = R"(HloModule module

max_half_.26 {
  x.27 = f16[] parameter(0)
  y.28 = f16[] parameter(1)
  ROOT maximum.29 = f16[] maximum(x.27, y.28), backend_config="{\"isInplace\":true}"
}

add_float_.36 {
  x.37 = f32[] parameter(0)
  y.38 = f32[] parameter(1)
  ROOT add.39 = f32[] add(x.37, y.38), backend_config="{\"isInplace\":true}"
}

_pop_op_scaled_inplace {
  arg_0 = f16[64,64] parameter(0)
  arg_1 = f16[64,64] parameter(1)
  constant.4.clone = f16[] constant(0.0100021)
  broadcast.87.clone = f16[64,64] broadcast(constant.4.clone), dimensions={}
  multiply.88.clone = f16[64,64] multiply(arg_1, broadcast.87.clone)
  ROOT subtract.89.clone = f16[64,64] subtract(arg_0, multiply.88.clone)
}

_pop_op_scaled_inplace.1 {
  arg_0.1 = f16[32,64] parameter(0)
  arg_1.1 = f16[32,64] parameter(1)
  constant.4.clone.1 = f16[] constant(0.0100021)
  broadcast.116.clone = f16[32,64] broadcast(constant.4.clone.1), dimensions={}
  multiply.117.clone = f16[32,64] multiply(arg_1.1, broadcast.116.clone)
  ROOT subtract.118.clone = f16[32,64] subtract(arg_0.1, multiply.117.clone)
}

_pop_op_scaled_inplace.2 {
  arg_0.2 = f16[50176,32] parameter(0)
  arg_1.2 = f16[50176,32] parameter(1)
  constant.4.clone.2 = f16[] constant(0.0100021)
  broadcast.140.clone = f16[50176,32] broadcast(constant.4.clone.2), dimensions={}
  multiply.141.clone = f16[50176,32] multiply(arg_1.2, broadcast.140.clone)
  ROOT subtract.142.clone = f16[50176,32] subtract(arg_0.2, multiply.141.clone)
}

_pop_op_implicit_binary_inplace {
  arg_0.3 = f16[1,64] parameter(0)
  arg_1.3 = f16[1] parameter(1)
  broadcast.3 = f16[1,64] broadcast(arg_1.3), dimensions={0}
  ROOT subtract = f16[1,64] subtract(arg_0.3, broadcast.3)
}

_pop_op_implicit_binary {
  constant.4.clone.3 = f16[] constant(0.0100021)
  broadcast.4 = f16[1,64] broadcast(constant.4.clone.3), dimensions={}
  arg_0.4 = f16[1,64] parameter(0)
  ROOT multiply.3 = f16[1,64] multiply(broadcast.4, arg_0.4)
}

_pop_op_implicit_binary.1 {
  constant.4.clone.4 = f16[] constant(0.0100021)
  broadcast.5 = f16[1,64] broadcast(constant.4.clone.4), dimensions={}
  arg_0.5 = f16[1,64] parameter(0)
  ROOT multiply.4 = f16[1,64] multiply(broadcast.5, arg_0.5)
}

_pop_op_implicit_binary.2 {
  constant.4.clone.5 = f16[] constant(0.0100021)
  broadcast.6 = f16[1,32] broadcast(constant.4.clone.5), dimensions={}
  arg_0.6 = f16[1,32] parameter(0)
  ROOT multiply.5 = f16[1,32] multiply(broadcast.6, arg_0.6)
}

_pop_op_implicit_binary_inplace.1 {
  arg_0.7 = f16[1,64] parameter(0)
  arg_1.4 = f16[1] parameter(1)
  broadcast.7 = f16[1,64] broadcast(arg_1.4), dimensions={0}
  ROOT divide = f16[1,64] divide(arg_0.7, broadcast.7)
}

ENTRY cluster {
  arg2.3 = f16[50176,32] parameter(2), parameter_replication={false}
  arg0.1 = f16[1,50176] parameter(0), parameter_replication={false}
  reshape = f16[50176] reshape(arg0.1), backend_config="{\"isInplace\":true}"
  dot = f16[32] dot(reshape, arg2.3), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg3.4 = f16[32] parameter(3), parameter_replication={false}, control-predecessors={dot}
  add = f16[32] add(dot, arg3.4), backend_config="{\"isInplace\":true}"
  reshape.32 = f16[1,32] reshape(add), backend_config="{\"isInplace\":true}"
  custom-call = f16[1,32] custom-call(reshape.32), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  reshape.3 = f16[32] reshape(custom-call), backend_config="{\"isInplace\":true}"
  arg4.5 = f16[32,64] parameter(4), parameter_replication={false}
  dot.1 = f16[64] dot(reshape.3, arg4.5), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg5.6 = f16[64] parameter(5), parameter_replication={false}, control-predecessors={dot.1}
  add.1 = f16[64] add(dot.1, arg5.6), backend_config="{\"isInplace\":true}"
  reshape.33 = f16[1,64] reshape(add.1), backend_config="{\"isInplace\":true}"
  custom-call.1 = f16[1,64] custom-call(reshape.33), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  reshape.6 = f16[64] reshape(custom-call.1), backend_config="{\"isInplace\":true}"
  arg6.7 = f16[64,64] parameter(6), parameter_replication={false}
  dot.2 = f16[64] dot(reshape.6, arg6.7), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg7.8 = f16[64] parameter(7), parameter_replication={false}, control-predecessors={dot.2}
  add.2 = f16[64] add(dot.2, arg7.8), backend_config="{\"isInplace\":true}"
  reshape.34 = f16[1,64] reshape(add.2), backend_config="{\"isInplace\":true}"
  custom-call.2 = f16[1,64] custom-call(reshape.34), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  constant.3 = f16[] constant(-inf)
  reduce.30 = f16[1] reduce(custom-call.2, constant.3), dimensions={1}, to_apply=max_half_.26
  fusion.3 = f16[1,64] fusion(custom-call.2, reduce.30), kind=kCustom, calls=_pop_op_implicit_binary_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
  exponential.33 = f16[1,64] exponential(fusion.3), backend_config="{\"isInplace\":true}"
  constant.9 = f16[] constant(0)
  reduce.40.clone = f16[1] reduce(exponential.33, constant.9), dimensions={1}, to_apply=add_float_.36
  fusion.7 = f16[1,64] fusion(exponential.33, reduce.40.clone), kind=kCustom, calls=_pop_op_implicit_binary_inplace.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  arg1.2 = f16[1,64] parameter(1), parameter_replication={false}
  subtract.57 = f16[1,64] subtract(fusion.7, arg1.2), backend_config="{\"isInplace\":true}"
  custom-call.3 = f16[1,64] custom-call(custom-call.2, subtract.57), custom_call_target="ReluGrad"
  reshape.11 = f16[64] reshape(custom-call.3), backend_config="{\"isInplace\":true}"
  transpose.79 = f16[64,64] transpose(arg6.7), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.3 = f16[64] dot(reshape.11, transpose.79), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  reshape.12 = f16[1,64] reshape(dot.3), backend_config="{\"isInplace\":true}"
  custom-call.4 = f16[1,64] custom-call(custom-call.1, reshape.12), custom_call_target="ReluGrad"
  reshape.13 = f16[64] reshape(custom-call.4), backend_config="{\"isInplace\":true}"
  transpose.108 = f16[64,32] transpose(arg4.5), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.4 = f16[32] dot(reshape.13, transpose.108), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  reshape.14 = f16[1,32] reshape(dot.4), backend_config="{\"isInplace\":true}"
  custom-call.5 = f16[1,32] custom-call(custom-call, reshape.14), custom_call_target="ReluGrad"
  reshape.17 = f16[32] reshape(custom-call.5), backend_config="{\"isInplace\":true}"
  dot.5 = f16[50176,32] dot(reshape, reshape.17), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion.2 = f16[50176,32] fusion(arg2.3, dot.5), kind=kCustom, calls=_pop_op_scaled_inplace.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  fusion.6 = f16[1,32] fusion(custom-call.5), kind=kCustom, calls=_pop_op_implicit_binary.2, backend_config="{\"fusionConfig\":{}}"
  reshape.40 = f16[32] reshape(fusion.6), backend_config="{\"isInplace\":true}"
  subtract.134 = f16[32] subtract(arg3.4, reshape.40), backend_config="{\"isInplace\":true}"
  dot.6 = f16[32,64] dot(reshape.3, reshape.13), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion.1 = f16[32,64] fusion(arg4.5, dot.6), kind=kCustom, calls=_pop_op_scaled_inplace.1, control-predecessors={transpose.108, dot.4}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  fusion.5 = f16[1,64] fusion(custom-call.4), kind=kCustom, calls=_pop_op_implicit_binary.1, backend_config="{\"fusionConfig\":{}}"
  reshape.39 = f16[64] reshape(fusion.5), backend_config="{\"isInplace\":true}"
  subtract.105 = f16[64] subtract(arg5.6, reshape.39), backend_config="{\"isInplace\":true}"
  dot.7 = f16[64,64] dot(reshape.6, reshape.11), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion = f16[64,64] fusion(arg6.7, dot.7), kind=kCustom, calls=_pop_op_scaled_inplace, control-predecessors={transpose.79, dot.3}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  fusion.4 = f16[1,64] fusion(custom-call.3), kind=kCustom, calls=_pop_op_implicit_binary, backend_config="{\"fusionConfig\":{}}"
  reshape.38 = f16[64] reshape(fusion.4), backend_config="{\"isInplace\":true}"
  subtract.76 = f16[64] subtract(arg7.8, reshape.38), backend_config="{\"isInplace\":true}"
  ROOT tuple.156 = (f16[50176,32], f16[32], f16[32,64], f16[64], f16[64,64], f16[64]) tuple(fusion.2, subtract.134, fusion.1, subtract.105, fusion, subtract.76), backend_config="{\"isInplace\":true}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 8);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot" || it.first->name() == "dot.1" ||
        it.first->name() == "dot.2") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD) << it.first->name();
    } else if (it.first->name() == "dot.3" || it.first->name() == "dot.4") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD) << it.first->name();
    } else if (it.first->name() == "dot.5" || it.first->name() == "dot.6" ||
               it.first->name() == "dot.7") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU) << it.first->name();
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, SharedMatmulWeights) {
  std::string hlo_string = R"(HloModule module

max_float_.45 {
  x.46 = f32[] parameter(0)
  y.47 = f32[] parameter(1)
  ROOT maximum.48 = f32[] maximum(x.46, y.47), backend_config="{\"isInplace\":true}"
}

add_float_.55 {
  x.56 = f32[] parameter(0)
  y.57 = f32[] parameter(1)
  ROOT add.58 = f32[] add(x.56, y.57), backend_config="{\"isInplace\":true}"
}

_pop_op_scaled_inplace {
  arg_0 = f32[4,4] parameter(0)
  arg_1 = f32[4,4] parameter(1)
  constant.90.clone = f32[] constant(0.1)
  broadcast.160.clone = f32[4,4] broadcast(constant.90.clone), dimensions={}
  multiply.161.clone = f32[4,4] multiply(arg_1, broadcast.160.clone)
  ROOT subtract.162.clone = f32[4,4] subtract(arg_0, multiply.161.clone)
}

_pop_op_implicit_binary_inplace {
  arg_0.1 = f32[1,4] parameter(0)
  arg_1.1 = f32[1] parameter(1)
  broadcast.3 = f32[1,4] broadcast(arg_1.1), dimensions={0}
  ROOT divide = f32[1,4] divide(arg_0.1, broadcast.3)
}

_pop_op_implicit_binary_inplace.1 {
  arg_0.2 = f32[1,4] parameter(0)
  arg_1.2 = f32[1] parameter(1)
  broadcast.4 = f32[1,4] broadcast(arg_1.2), dimensions={0}
  ROOT subtract = f32[1,4] subtract(arg_0.2, broadcast.4)
}

_pop_op_implicit_binary_inplace.2 {
  arg_0.3 = f32[1,4] parameter(0)
  constant.90.clone.1 = f32[] constant(0.1)
  broadcast.5 = f32[1,4] broadcast(constant.90.clone.1), dimensions={}
  ROOT multiply.3 = f32[1,4] multiply(arg_0.3, broadcast.5)
}

_pop_op_implicit_binary_inplace.3 {
  arg_0.4 = f32[1,4] parameter(0)
  constant.90.clone.2 = f32[] constant(0.1)
  broadcast.6 = f32[1,4] broadcast(constant.90.clone.2), dimensions={}
  ROOT multiply.4 = f32[1,4] multiply(arg_0.4, broadcast.6)
}

_pop_op_implicit_binary_inplace.4 {
  arg_0.5 = f32[1,4] parameter(0)
  constant.90.clone.3 = f32[] constant(0.1)
  broadcast.7 = f32[1,4] broadcast(constant.90.clone.3), dimensions={}
  ROOT multiply.5 = f32[1,4] multiply(arg_0.5, broadcast.7)
}

ENTRY cluster {
  arg0.1 = f32[1,4] parameter(0), parameter_replication={false}
  reshape = f32[4] reshape(arg0.1), backend_config="{\"isInplace\":true}"
  arg5.6 = f32[4,4] parameter(5), parameter_replication={false}
  dot = f32[4] dot(reshape, arg5.6), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg2.3 = f32[4] parameter(2), parameter_replication={false}, control-predecessors={dot}
  add = f32[4] add(dot, arg2.3), backend_config="{\"isInplace\":true}"
  reshape.31 = f32[1,4] reshape(add), backend_config="{\"isInplace\":true}"
  custom-call = f32[1,4] custom-call(reshape.31), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  reshape.3 = f32[4] reshape(custom-call), backend_config="{\"isInplace\":true}"
  dot.1 = f32[4] dot(reshape.3, arg5.6), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg3.4 = f32[4] parameter(3), parameter_replication={false}, control-predecessors={dot.1}
  add.1 = f32[4] add(dot.1, arg3.4), backend_config="{\"isInplace\":true}"
  reshape.32 = f32[1,4] reshape(add.1), backend_config="{\"isInplace\":true}"
  custom-call.1 = f32[1,4] custom-call(reshape.32), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  reshape.6 = f32[4] reshape(custom-call.1), backend_config="{\"isInplace\":true}"
  dot.2 = f32[4] dot(reshape.6, arg5.6), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  arg4.5 = f32[4] parameter(4), parameter_replication={false}, control-predecessors={dot.2}
  add.2 = f32[4] add(dot.2, arg4.5), backend_config="{\"isInplace\":true}"
  reshape.33 = f32[1,4] reshape(add.2), backend_config="{\"isInplace\":true}"
  custom-call.2 = f32[1,4] custom-call(reshape.33), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  constant.44 = f32[] constant(-inf)
  reduce.49 = f32[1] reduce(custom-call.2, constant.44), dimensions={1}, to_apply=max_float_.45
  fusion.2 = f32[1,4] fusion(custom-call.2, reduce.49), kind=kCustom, calls=_pop_op_implicit_binary_inplace.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]}}"
  exponential.52 = f32[1,4] exponential(fusion.2), backend_config="{\"isInplace\":true}"
  constant.54 = f32[] constant(0)
  reduce.59 = f32[1] reduce(exponential.52, constant.54), dimensions={1}, to_apply=add_float_.55
  fusion.1 = f32[1,4] fusion(exponential.52, reduce.59), kind=kCustom, calls=_pop_op_implicit_binary_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  arg1.2 = f32[4] parameter(1), parameter_replication={false}
  reshape.9 = f32[1,4] reshape(arg1.2), inferred_dimension=0, backend_config="{\"isInplace\":true}"
  subtract.76 = f32[1,4] subtract(fusion.1, reshape.9), backend_config="{\"isInplace\":true}"
  custom-call.3 = f32[1,4] custom-call(custom-call.2, subtract.76), custom_call_target="ReluGrad"
  reshape.12 = f32[4] reshape(custom-call.3), backend_config="{\"isInplace\":true}"
  transpose.102 = f32[4,4] transpose(arg5.6), dimensions={1,0}, backend_config="{\"isInplace\":true}"
  dot.3 = f32[4] dot(reshape.12, transpose.102), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  reshape.13 = f32[1,4] reshape(dot.3), backend_config="{\"isInplace\":true}"
  custom-call.4 = f32[1,4] custom-call(custom-call.1, reshape.13), custom_call_target="ReluGrad"
  reshape.14 = f32[4] reshape(custom-call.4), backend_config="{\"isInplace\":true}"
  dot.4 = f32[4] dot(reshape.14, transpose.102), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  reshape.15 = f32[1,4] reshape(dot.4), backend_config="{\"isInplace\":true}"
  custom-call.5 = f32[1,4] custom-call(custom-call, reshape.15), custom_call_target="ReluGrad"
  reshape.27 = f32[4] reshape(custom-call.5), backend_config="{\"isInplace\":true}"
  dot.7 = f32[4,4] dot(reshape, reshape.27), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion.5 = f32[1,4] fusion(custom-call.5), kind=kCustom, calls=_pop_op_implicit_binary_inplace.4, control-predecessors={reshape.27, dot.7}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.36 = f32[4] reshape(fusion.5), backend_config="{\"isInplace\":true}"
  subtract.144 = f32[4] subtract(arg2.3, reshape.36), backend_config="{\"isInplace\":true}"
  dot.5 = f32[4,4] dot(reshape.3, reshape.14), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion.4 = f32[1,4] fusion(custom-call.4), kind=kCustom, calls=_pop_op_implicit_binary_inplace.3, control-predecessors={reshape.14, dot.5, dot.4}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.35 = f32[4] reshape(fusion.4), backend_config="{\"isInplace\":true}"
  subtract.120 = f32[4] subtract(arg3.4, reshape.35), backend_config="{\"isInplace\":true}"
  dot.6 = f32[4,4] dot(reshape.6, reshape.12), lhs_contracting_dims={}, rhs_contracting_dims={}
  fusion.3 = f32[1,4] fusion(custom-call.3), kind=kCustom, calls=_pop_op_implicit_binary_inplace.2, control-predecessors={reshape.12, dot.3, dot.6}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.34 = f32[4] reshape(fusion.3), backend_config="{\"isInplace\":true}"
  subtract.93 = f32[4] subtract(arg4.5, reshape.34), backend_config="{\"isInplace\":true}"
  add.157 = f32[4,4] add(dot.5, dot.6), backend_config="{\"isInplace\":true}"
  add.158 = f32[4,4] add(add.157, dot.7), backend_config="{\"isInplace\":true}"
  fusion = f32[4,4] fusion(arg5.6, add.158), kind=kCustom, calls=_pop_op_scaled_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  ROOT tuple.172 = (f32[4], f32[4], f32[4], f32[4,4]) tuple(subtract.144, subtract.120, subtract.93, fusion), backend_config="{\"isInplace\":true}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 8);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot" || it.first->name() == "dot.1" ||
        it.first->name() == "dot.2") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD) << it.first->name();
    } else if (it.first->name() == "dot.3" || it.first->name() == "dot.4") {
      EXPECT_EQ(it.second, MLType::TRAINING_BWD) << it.first->name();
    } else if (it.first->name() == "dot.5" || it.first->name() == "dot.6" ||
               it.first->name() == "dot.7") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU) << it.first->name();
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

TEST_F(ConvolutionClassifierTest, SharedMatmulInputs) {
  std::string hlo_string = R"(HloModule module

max_float_.50 {
  x.51 = f32[] parameter(0), backend_config="{}"
  y.52 = f32[] parameter(1), backend_config="{}"
  ROOT maximum.53 = f32[] maximum(x.51, y.52), backend_config="{\"isInplace\":true}"
}

add_float_.60 {
  x.61 = f32[] parameter(0), backend_config="{}"
  y.62 = f32[] parameter(1), backend_config="{}"
  ROOT add.63 = f32[] add(x.61, y.62), backend_config="{\"isInplace\":true}"
}

_pop_op_scaled_inplace {
  arg_0 = f32[4,4] parameter(0)
  arg_1 = f32[4,4] parameter(1)
  constant.95.clone = f32[] constant(0.1), backend_config="{}"
  broadcast.140.clone = f32[4,4] broadcast(constant.95.clone), dimensions={}, backend_config="{}"
  multiply.174.clone = f32[4,4] multiply(arg_1, broadcast.140.clone), backend_config="{}"
  ROOT subtract.175.clone = f32[4,4] subtract(arg_0, multiply.174.clone), backend_config="{}"
}

_pop_op_scaled_inplace.1 {
  arg_0.1 = f32[4,4] parameter(0)
  arg_1.1 = f32[4,4] parameter(1)
  constant.95.clone.1 = f32[] constant(0.1), backend_config="{}"
  broadcast.140.clone.1 = f32[4,4] broadcast(constant.95.clone.1), dimensions={}, backend_config="{}"
  multiply.111.clone = f32[4,4] multiply(arg_1.1, broadcast.140.clone.1), backend_config="{}"
  ROOT subtract.112.clone = f32[4,4] subtract(arg_0.1, multiply.111.clone), backend_config="{}"
}

_pop_op_scaled_inplace.2 {
  arg_0.2 = f32[4,4] parameter(0)
  arg_1.2 = f32[4,4] parameter(1)
  constant.95.clone.2 = f32[] constant(0.1), backend_config="{}"
  broadcast.140.clone.2 = f32[4,4] broadcast(constant.95.clone.2), dimensions={}, backend_config="{}"
  multiply.141.clone = f32[4,4] multiply(arg_1.2, broadcast.140.clone.2), backend_config="{}"
  ROOT subtract.142.clone = f32[4,4] subtract(arg_0.2, multiply.141.clone), backend_config="{}"
}

_pop_op_implicit_binary_inplace {
  arg_0.3 = f32[1,4] parameter(0)
  arg_1.3 = f32[1] parameter(1)
  broadcast.3 = f32[1,4] broadcast(arg_1.3), dimensions={0}, backend_config="{}"
  ROOT subtract = f32[1,4] subtract(arg_0.3, broadcast.3), backend_config="{}"
}

_pop_op_implicit_binary_inplace.1 {
  arg_0.4 = f32[1,4] parameter(0)
  constant.95.clone.3 = f32[] constant(0.1), backend_config="{}"
  broadcast.4 = f32[1,4] broadcast(constant.95.clone.3), dimensions={}, backend_config="{}"
  ROOT multiply.3 = f32[1,4] multiply(arg_0.4, broadcast.4), backend_config="{}"
}

_pop_op_implicit_binary_inplace.2 {
  arg_0.5 = f32[1,4] parameter(0)
  constant.95.clone.4 = f32[] constant(0.1), backend_config="{}"
  broadcast.5 = f32[1,4] broadcast(constant.95.clone.4), dimensions={}, backend_config="{}"
  ROOT multiply.4 = f32[1,4] multiply(arg_0.5, broadcast.5), backend_config="{}"
}

_pop_op_implicit_binary_inplace.3 {
  arg_0.6 = f32[1,4] parameter(0)
  constant.95.clone.5 = f32[] constant(0.1), backend_config="{}"
  broadcast.6 = f32[1,4] broadcast(constant.95.clone.5), dimensions={}, backend_config="{}"
  ROOT multiply.5 = f32[1,4] multiply(arg_0.6, broadcast.6), backend_config="{}"
}

_pop_op_implicit_binary_inplace.4 {
  arg_0.7 = f32[1,4] parameter(0)
  arg_1.4 = f32[1] parameter(1)
  broadcast.7 = f32[1,4] broadcast(arg_1.4), dimensions={0}, backend_config="{}"
  ROOT divide = f32[1,4] divide(arg_0.7, broadcast.7), backend_config="{}"
}

ENTRY cluster {
  arg0.1 = f32[1,4] parameter(0), parameter_replication={false}, backend_config="{}"
  reshape = f32[4] reshape(arg0.1), backend_config="{\"isInplace\":true}"
  arg5.6 = f32[4,4] parameter(5), parameter_replication={false}, backend_config="{}"
  dot = f32[4] dot(reshape, arg5.6), lhs_contracting_dims={0}, rhs_contracting_dims={0}, backend_config="{}"
  arg2.3 = f32[4] parameter(2), parameter_replication={false}, control-predecessors={dot}, backend_config="{}"
  add = f32[4] add(dot, arg2.3), backend_config="{\"isInplace\":true}"
  reshape.25 = f32[1,4] reshape(add), backend_config="{\"isInplace\":true}"
  custom-call = f32[1,4] custom-call(reshape.25), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  arg7.8 = f32[4,4] parameter(7), parameter_replication={false}, backend_config="{}"
  dot.1 = f32[4] dot(reshape, arg7.8), lhs_contracting_dims={0}, rhs_contracting_dims={0}, backend_config="{}"
  arg4.5 = f32[4] parameter(4), parameter_replication={false}, control-predecessors={dot.1}, backend_config="{}"
  add.2 = f32[4] add(dot.1, arg4.5), backend_config="{\"isInplace\":true}"
  reshape.28 = f32[1,4] reshape(add.2), backend_config="{\"isInplace\":true}"
  custom-call.2 = f32[1,4] custom-call(reshape.28), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  arg6.7 = f32[4,4] parameter(6), parameter_replication={false}, backend_config="{}"
  dot.2 = f32[4] dot(reshape, arg6.7), lhs_contracting_dims={0}, rhs_contracting_dims={0}, backend_config="{}"
  arg3.4 = f32[4] parameter(3), parameter_replication={false}, control-predecessors={dot.2}, backend_config="{}"
  add.1 = f32[4] add(dot.2, arg3.4), backend_config="{\"isInplace\":true}"
  reshape.27 = f32[1,4] reshape(add.1), backend_config="{\"isInplace\":true}"
  custom-call.1 = f32[1,4] custom-call(reshape.27), custom_call_target="Relu", backend_config="{\"isInplace\":true}"
  add.20 = f32[1,4] add(custom-call, custom-call.1), backend_config="{}"
  add.25 = f32[1,4] add(custom-call.2, add.20), backend_config="{}"
  constant.49 = f32[] constant(-inf), backend_config="{}"
  reduce.54 = f32[1] reduce(add.25, constant.49), dimensions={1}, to_apply=max_float_.50, backend_config="{}"
  fusion.3 = f32[1,4] fusion(add.25, reduce.54), kind=kCustom, calls=_pop_op_implicit_binary_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  exponential.57 = f32[1,4] exponential(fusion.3), backend_config="{\"isInplace\":true}"
  constant.59 = f32[] constant(0), backend_config="{}"
  reduce.64 = f32[1] reduce(exponential.57, constant.59), dimensions={1}, to_apply=add_float_.60, backend_config="{}"
  fusion.7 = f32[1,4] fusion(exponential.57, reduce.64), kind=kCustom, calls=_pop_op_implicit_binary_inplace.4, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  arg1.2 = f32[4] parameter(1), parameter_replication={false}, backend_config="{}"
  reshape.11 = f32[1,4] reshape(arg1.2), inferred_dimension=0, backend_config="{\"isInplace\":true}"
  subtract.81 = f32[1,4] subtract(fusion.7, reshape.11), backend_config="{\"isInplace\":true}"
  custom-call.4 = f32[1,4] custom-call(custom-call, subtract.81), custom_call_target="ReluGrad", backend_config="{}"
  reshape.17 = f32[4] reshape(custom-call.4), backend_config="{\"isInplace\":true}"
  dot.3 = f32[4,4] dot(reshape, reshape.17), lhs_contracting_dims={}, rhs_contracting_dims={}, backend_config="{}"
  fusion.4 = f32[1,4] fusion(custom-call.4), kind=kCustom, calls=_pop_op_implicit_binary_inplace.1, control-predecessors={reshape.17, dot.3}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.30 = f32[4] reshape(fusion.4), backend_config="{\"isInplace\":true}"
  subtract.128 = f32[4] subtract(arg2.3, reshape.30), backend_config="{\"isInplace\":true}"
  custom-call.3 = f32[1,4] custom-call(custom-call.1, subtract.81), custom_call_target="ReluGrad", backend_config="{}"
  reshape.19 = f32[4] reshape(custom-call.3), backend_config="{\"isInplace\":true}"
  dot.4 = f32[4,4] dot(reshape, reshape.19), lhs_contracting_dims={}, rhs_contracting_dims={}, backend_config="{}"
  fusion.5 = f32[1,4] fusion(custom-call.3), kind=kCustom, calls=_pop_op_implicit_binary_inplace.2, control-predecessors={reshape.19, dot.4}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.29 = f32[4] reshape(fusion.5), backend_config="{\"isInplace\":true}"
  subtract.98 = f32[4] subtract(arg3.4, reshape.29), backend_config="{\"isInplace\":true}"
  custom-call.5 = f32[1,4] custom-call(custom-call.2, subtract.81), custom_call_target="ReluGrad", backend_config="{}"
  reshape.21 = f32[4] reshape(custom-call.5), backend_config="{\"isInplace\":true}"
  dot.5 = f32[4,4] dot(reshape, reshape.21), lhs_contracting_dims={}, rhs_contracting_dims={}, backend_config="{}"
  fusion.6 = f32[1,4] fusion(custom-call.5), kind=kCustom, calls=_pop_op_implicit_binary_inplace.3, control-predecessors={reshape.21, dot.5}, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  reshape.31 = f32[4] reshape(fusion.6), backend_config="{\"isInplace\":true}"
  subtract.161 = f32[4] subtract(arg4.5, reshape.31), backend_config="{\"isInplace\":true}"
  fusion.2 = f32[4,4] fusion(arg5.6, dot.3), kind=kCustom, calls=_pop_op_scaled_inplace.2, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  fusion.1 = f32[4,4] fusion(arg6.7, dot.4), kind=kCustom, calls=_pop_op_scaled_inplace.1, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  fusion = f32[4,4] fusion(arg7.8, dot.5), kind=kCustom, calls=_pop_op_scaled_inplace, backend_config="{\"fusionConfig\":{\"inplaceDescriptions\":[{\"kind\":\"USE_ALIAS_READ_WRITE\"}]},\"isInplace\":true}"
  ROOT tuple.189 = (f32[4], f32[4], f32[4], f32[4,4], f32[4,4], f32[4,4]) tuple(subtract.128, subtract.98, subtract.161, fusion.2, fusion.1, fusion), backend_config="{\"isInplace\":true}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({});
  config.set_resource_input_count(4);
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  auto all_classifications_or_status = GetAllNotNoneMlTypes(module);
  EXPECT_TRUE(all_classifications_or_status.ok());
  auto all_classifications = all_classifications_or_status.ValueOrDie();
  EXPECT_EQ(all_classifications.size(), 6);

  for (auto it : all_classifications) {
    if (it.first->name() == "dot" || it.first->name() == "dot.1" ||
        it.first->name() == "dot.2") {
      EXPECT_EQ(it.second, MLType::TRAINING_FWD) << it.first->name();
    } else if (it.first->name() == "dot.3" || it.first->name() == "dot.4" ||
               it.first->name() == "dot.5") {
      EXPECT_EQ(it.second, MLType::TRAINING_WU) << it.first->name();
    } else {
      FAIL() << "We should not have any missing matmuls";
    }
  }
}

// TODO:
// - convolutions which share common weights
// - convolutions which share common inputs
// - double check with transformer models

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
