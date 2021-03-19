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
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using NormInputRecomputationTest = HloTestBase;

const std::string hlo_string = R"(
 HloModule top

 Sum-reduction.38 (x.39: f32[], y.40: f32[]) -> f32[] {
   y.40 = f32[] parameter(1)
   x.39 = f32[] parameter(0)
   ROOT add.41 = f32[] add(f32[] x.39, f32[] y.40)
 }

 _pop_op_conv_with_reverse (arg_0: f32[1,4,4,2], arg_1: f32[1,1,2,2]) -> f32[1,4,4,2] {
   arg_0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   arg_1 = f32[1,1,2,2]{3,2,1,0} parameter(1)
   reverse.64.clone = f32[1,1,2,2]{3,2,1,0} reverse(f32[1,1,2,2]{3,2,1,0} arg_1), dimensions={0,1}
   ROOT convolution.65.clone = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_0, f32[1,1,2,2]{3,2,1,0} reverse.64.clone), window={size=1x1}, dim_labels=b01f_01oi->b01f
 }

 _pop_op_relu (arg_0.1: f32[1,4,4,2]) -> f32[1,4,4,2] {
   arg_0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   constant.20.clone = f32[] constant(0)
   broadcast.21.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.20.clone), dimensions={}
   ROOT maximum.34.clone = f32[1,4,4,2]{3,2,1,0} maximum(f32[1,4,4,2]{3,2,1,0} arg_0.1, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone)
 }

 _pop_op_relu.1 (arg_0.2: f32[1,4,4,2]) -> f32[1,4,4,2] {
   arg_0.2 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   constant.20.clone.1 = f32[] constant(0)
   broadcast.21.clone.1 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.20.clone.1), dimensions={}
   ROOT maximum.22.clone = f32[1,4,4,2]{3,2,1,0} maximum(f32[1,4,4,2]{3,2,1,0} arg_0.2, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone.1)
 }

 _pop_op_relugrad (arg_0.3: f32[1,4,4,2], arg_1.1: f32[1,4,4,2]) -> f32[1,4,4,2] {
   arg_0.3 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   constant.20.clone.2 = f32[] constant(0)
   broadcast.21.clone.2 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.20.clone.2), dimensions={}
   compare.72.clone = pred[1,4,4,2]{3,2,1,0} compare(f32[1,4,4,2]{3,2,1,0} arg_0.3, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone.2), direction=GT
   arg_1.1 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   ROOT select.73.clone = f32[1,4,4,2]{3,2,1,0} select(pred[1,4,4,2]{3,2,1,0} compare.72.clone, f32[1,4,4,2]{3,2,1,0} arg_1.1, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone.2)
 }

 _pop_op_relugrad.1 (arg_0.4: f32[1,4,4,2], arg_1.2: f32[1,4,4,2]) -> f32[1,4,4,2] {
   arg_0.4 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   constant.20.clone.3 = f32[] constant(0)
   broadcast.21.clone.3 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.20.clone.3), dimensions={}
   compare.46.clone = pred[1,4,4,2]{3,2,1,0} compare(f32[1,4,4,2]{3,2,1,0} arg_0.4, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone.3), direction=GT
   arg_1.2 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   ROOT select.47.clone = f32[1,4,4,2]{3,2,1,0} select(pred[1,4,4,2]{3,2,1,0} compare.46.clone, f32[1,4,4,2]{3,2,1,0} arg_1.2, f32[1,4,4,2]{3,2,1,0} broadcast.21.clone.3)
 }

 _pop_op_conv_scaled_inplace (arg_0.5: f32[1,1,2,2], arg_1.3: f32[1,4,4,2], arg_2: f32[1,4,4,2]) -> f32[1,1,2,2] {
   arg_0.5 = f32[1,1,2,2]{3,2,1,0} parameter(0)
   arg_1.3 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   arg_2 = f32[1,4,4,2]{3,2,1,0} parameter(2)
   convolution.89.clone = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_1.3, f32[1,4,4,2]{3,2,1,0} arg_2), window={size=4x4}, dim_labels=f01b_i01o->01bf
   constant.55.clone = f32[] constant(0.1)
   broadcast.67.clone = f32[1,1,2,2]{3,2,1,0} broadcast(f32[] constant.55.clone), dimensions={}
   multiply.92.clone = f32[1,1,2,2]{3,2,1,0} multiply(f32[1,1,2,2]{3,2,1,0} convolution.89.clone, f32[1,1,2,2]{3,2,1,0} broadcast.67.clone)
   ROOT subtract.93.clone = f32[1,1,2,2]{3,2,1,0} subtract(f32[1,1,2,2]{3,2,1,0} arg_0.5, f32[1,1,2,2]{3,2,1,0} multiply.92.clone)
 }

 _pop_op_conv_scaled_inplace.1 (arg_0.6: f32[1,1,2,2], arg_1.4: f32[1,4,4,2], arg_2.1: f32[1,4,4,2]) -> f32[1,1,2,2] {
   arg_0.6 = f32[1,1,2,2]{3,2,1,0} parameter(0)
   arg_1.4 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   arg_2.1 = f32[1,4,4,2]{3,2,1,0} parameter(2)
   convolution.63.clone = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_1.4, f32[1,4,4,2]{3,2,1,0} arg_2.1), window={size=4x4}, dim_labels=f01b_i01o->01bf
   constant.55.clone.1 = f32[] constant(0.1)
   broadcast.67.clone.1 = f32[1,1,2,2]{3,2,1,0} broadcast(f32[] constant.55.clone.1), dimensions={}
   multiply.68.clone = f32[1,1,2,2]{3,2,1,0} multiply(f32[1,1,2,2]{3,2,1,0} convolution.63.clone, f32[1,1,2,2]{3,2,1,0} broadcast.67.clone.1)
   ROOT subtract.69.clone = f32[1,1,2,2]{3,2,1,0} subtract(f32[1,1,2,2]{3,2,1,0} arg_0.6, f32[1,1,2,2]{3,2,1,0} multiply.68.clone)
 }

 _pop_op_scaled_inplace (arg_0.7: f32[2], arg_1.5: f32[2]) -> f32[2] {
   arg_0.7 = f32[2]{0} parameter(0)
   arg_1.5 = f32[2]{0} parameter(1)
   constant.55.clone.2 = f32[] constant(0.1)
   broadcast.56.clone = f32[2]{0} broadcast(f32[] constant.55.clone.2), dimensions={}
   multiply.87.clone = f32[2]{0} multiply(f32[2]{0} arg_1.5, f32[2]{0} broadcast.56.clone)
   ROOT subtract.88.clone = f32[2]{0} subtract(f32[2]{0} arg_0.7, f32[2]{0} multiply.87.clone)
 }

 _pop_op_scaled_inplace.1 (arg_0.8: f32[2], arg_1.6: f32[2]) -> f32[2] {
   arg_0.8 = f32[2]{0} parameter(0)
   arg_1.6 = f32[2]{0} parameter(1)
   constant.55.clone.3 = f32[] constant(0.1)
   broadcast.56.clone.1 = f32[2]{0} broadcast(f32[] constant.55.clone.3), dimensions={}
   multiply.83.clone = f32[2]{0} multiply(f32[2]{0} arg_1.6, f32[2]{0} broadcast.56.clone.1)
   ROOT subtract.84.clone = f32[2]{0} subtract(f32[2]{0} arg_0.8, f32[2]{0} multiply.83.clone)
 }

 _pop_op_scaled_inplace.2 (arg_0.9: f32[2], arg_1.7: f32[2]) -> f32[2] {
   arg_0.9 = f32[2]{0} parameter(0)
   arg_1.7 = f32[2]{0} parameter(1)
   constant.55.clone.4 = f32[] constant(0.1)
   broadcast.56.clone.2 = f32[2]{0} broadcast(f32[] constant.55.clone.4), dimensions={}
   multiply.61.clone = f32[2]{0} multiply(f32[2]{0} arg_1.7, f32[2]{0} broadcast.56.clone.2)
   ROOT subtract.62.clone = f32[2]{0} subtract(f32[2]{0} arg_0.9, f32[2]{0} multiply.61.clone)
 }

 _pop_op_scaled_inplace.3 (arg_0.10: f32[2], arg_1.8: f32[2]) -> f32[2] {
   arg_0.10 = f32[2]{0} parameter(0)
   arg_1.8 = f32[2]{0} parameter(1)
   constant.55.clone.5 = f32[] constant(0.1)
   broadcast.56.clone.3 = f32[2]{0} broadcast(f32[] constant.55.clone.5), dimensions={}
   multiply.57.clone = f32[2]{0} multiply(f32[2]{0} arg_1.8, f32[2]{0} broadcast.56.clone.3)
   ROOT subtract.58.clone = f32[2]{0} subtract(f32[2]{0} arg_0.10, f32[2]{0} multiply.57.clone)
 }

 _pop_op_wide_const () -> f32[1,4,4,2] {
   constant.9.clone = f32[] constant(1)
   ROOT broadcast.10.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.9.clone), dimensions={}
 }

 ENTRY top (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[2], arg3.4: f32[1,1,2,2], arg4.5: f32[2], arg5.6: f32[2], arg6.7: f32[1,1,2,2]) -> (f32[], f32[2], f32[2], f32[1,1,2,2], f32[2], f32[2], f32[1,1,2,2]) {
   arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0), metadata={op_name="XLA_Args"}
   arg6.7 = f32[1,1,2,2]{3,2,1,0} parameter(6), metadata={op_name="XLA_Args"}
   convolution.11 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg0.1, f32[1,1,2,2]{3,2,1,0} arg6.7), window={size=1x1}, dim_labels=b01f_01io->b01f
   arg5.6 = f32[2]{0} parameter(5), control-predecessors={convolution.11}, metadata={op_name="XLA_Args"}
   arg4.5 = f32[2]{0} parameter(4), control-predecessors={convolution.11}, metadata={op_name="XLA_Args"}
   batch-norm-training.13 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[1,4,4,2]{3,2,1,0} convolution.11, f32[2]{0} arg5.6, f32[2]{0} arg4.5), epsilon=0.001, feature_index=3
   get-tuple-element.14 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.13), index=0
   fusion.2 = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} get-tuple-element.14), kind=kCustom, calls=_pop_op_relu.1, backend_config="{}"
   arg3.4 = f32[1,1,2,2]{3,2,1,0} parameter(3), metadata={op_name="XLA_Args"}
   convolution.23 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} fusion.2, f32[1,1,2,2]{3,2,1,0} arg3.4), window={size=1x1}, dim_labels=b01f_01io->b01f
   arg2.3 = f32[2]{0} parameter(2), control-predecessors={convolution.23}, metadata={op_name="XLA_Args"}
   arg1.2 = f32[2]{0} parameter(1), control-predecessors={convolution.23}, metadata={op_name="XLA_Args"}
   batch-norm-training.25 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[1,4,4,2]{3,2,1,0} convolution.23, f32[2]{0} arg2.3, f32[2]{0} arg1.2), epsilon=0.001, feature_index=3
   get-tuple-element.26 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.25), index=0
   fusion.1 = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} get-tuple-element.26), kind=kCustom, calls=_pop_op_relu, backend_config="{}"
   constant.20 = f32[] constant(0)
   reduce.42 = f32[] reduce(f32[1,4,4,2]{3,2,1,0} fusion.1, f32[] constant.20), dimensions={0,1,2,3}, to_apply=Sum-reduction.38
   get-tuple-element.28 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.25), index=1
   get-tuple-element.29 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.25), index=2
   fusion.11 = f32[1,4,4,2]{3,2,1,0} fusion(), kind=kCustom, calls=_pop_op_wide_const, backend_config="{}"
   fusion.4 = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} fusion.1, f32[1,4,4,2]{3,2,1,0} fusion.11), kind=kCustom, calls=_pop_op_relugrad.1, backend_config="{}"
   batch-norm-grad.50 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[1,4,4,2]{3,2,1,0} convolution.23, f32[2]{0} arg2.3, f32[2]{0} get-tuple-element.28, f32[2]{0} get-tuple-element.29, f32[1,4,4,2]{3,2,1,0} fusion.4), epsilon=0.001, feature_index=3
   get-tuple-element.53 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.50), index=2
   fusion.10 = f32[2]{0} fusion(f32[2]{0} arg1.2, f32[2]{0} get-tuple-element.53), kind=kCustom, calls=_pop_op_scaled_inplace.3, backend_config="{}"
   get-tuple-element.52 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.50), index=1
   fusion.9 = f32[2]{0} fusion(f32[2]{0} arg2.3, f32[2]{0} get-tuple-element.52), kind=kCustom, calls=_pop_op_scaled_inplace.2, backend_config="{}"
   get-tuple-element.51 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.50), index=0
   fusion = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} get-tuple-element.51, f32[1,1,2,2]{3,2,1,0} arg3.4), kind=kCustom, calls=_pop_op_conv_with_reverse, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"1\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"1\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelInputFeatureDimension\":\"3\",\"kernelOutputFeatureDimension\":\"2\",\"kernelSpatialDimensions\":[\"0\",\"1\"],\"inputFeatureDimension\":\"3\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"1\",\"2\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
   fusion.6 = f32[1,1,2,2]{3,2,1,0} fusion(f32[1,1,2,2]{3,2,1,0} arg3.4, f32[1,4,4,2]{3,2,1,0} fusion.2, f32[1,4,4,2]{3,2,1,0} get-tuple-element.51), kind=kCustom, calls=_pop_op_conv_scaled_inplace.1, control-predecessors={fusion}, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelOutputFeatureDimension\":\"3\",\"kernelSpatialDimensions\":[\"1\",\"2\"],\"inputBatchDimension\":\"3\",\"outputBatchDimension\":\"2\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"0\",\"1\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
   get-tuple-element.16 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.13), index=1
   get-tuple-element.17 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training.13), index=2
   fusion.3 = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} fusion.2, f32[1,4,4,2]{3,2,1,0} fusion), kind=kCustom, calls=_pop_op_relugrad, backend_config="{}"
   batch-norm-grad.76 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[1,4,4,2]{3,2,1,0} convolution.11, f32[2]{0} arg5.6, f32[2]{0} get-tuple-element.16, f32[2]{0} get-tuple-element.17, f32[1,4,4,2]{3,2,1,0} fusion.3), epsilon=0.001, feature_index=3
   get-tuple-element.79 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.76), index=2
   fusion.8 = f32[2]{0} fusion(f32[2]{0} arg4.5, f32[2]{0} get-tuple-element.79), kind=kCustom, calls=_pop_op_scaled_inplace.1, backend_config="{}"
   get-tuple-element.78 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.76), index=1
   fusion.7 = f32[2]{0} fusion(f32[2]{0} arg5.6, f32[2]{0} get-tuple-element.78), kind=kCustom, calls=_pop_op_scaled_inplace, backend_config="{}"
   get-tuple-element.77 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad.76), index=0
   fusion.5 = f32[1,1,2,2]{3,2,1,0} fusion(f32[1,1,2,2]{3,2,1,0} arg6.7, f32[1,4,4,2]{3,2,1,0} arg0.1, f32[1,4,4,2]{3,2,1,0} get-tuple-element.77), kind=kCustom, calls=_pop_op_conv_scaled_inplace, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelOutputFeatureDimension\":\"3\",\"kernelSpatialDimensions\":[\"1\",\"2\"],\"inputBatchDimension\":\"3\",\"outputBatchDimension\":\"2\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"0\",\"1\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
   ROOT tuple.107 = (f32[], f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(f32[] reduce.42, f32[2]{0} fusion.10, f32[2]{0} fusion.9, f32[1,1,2,2]{3,2,1,0} fusion.6, f32[2]{0} fusion.8, f32[2]{0} fusion.7, f32[1,1,2,2]{3,2,1,0} fusion.5), metadata={op_name="XLA_Retvals"}
 }
  )";

TEST_F(NormInputRecomputationTest, RecomputeInput) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({1, 2, 3, 4, 5, 6});
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  HloInstruction* input1 = FindInstruction(module, "convolution.11");
  HloInstruction* bn1 = FindInstruction(module, "batch-norm-training.13");
  HloInstruction* bn_grad1 = FindInstruction(module, "batch-norm-grad.76");

  HloInstruction* input2 = FindInstruction(module, "convolution.23");
  HloInstruction* bn2 = FindInstruction(module, "batch-norm-training.25");
  HloInstruction* bn_grad2 = FindInstruction(module, "batch-norm-grad.50");

  ASSERT_EQ(input1->users().size(), 2);
  ASSERT_EQ(bn1->operand(0), input1);
  ASSERT_EQ(bn_grad1->operand(0), input1);

  ASSERT_EQ(input2->users().size(), 2);
  ASSERT_EQ(bn2->operand(0), input2);
  ASSERT_EQ(bn_grad2->operand(0), input2);

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  ASSERT_TRUE(RecomputeInstructions(true).Run(module).ValueOrDie());

  ASSERT_EQ(bn1->operand(0), input1);
  ASSERT_NE(bn1->operand(0), bn_grad1->operand(0));
  ASSERT_TRUE(bn1->operand(0)->Identical(*bn_grad1->operand(0)));

  ASSERT_EQ(bn2->operand(0), input2);
  ASSERT_NE(bn2->operand(0), bn_grad2->operand(0));
  ASSERT_TRUE(bn2->operand(0)->Identical(*bn_grad2->operand(0)));
}

TEST_F(NormInputRecomputationTest, RecomputeInputOff) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({1, 2, 3, 4, 5, 6});
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

  ASSERT_FALSE(RecomputeInstructions(false).Run(module).ValueOrDie());
}

const std::string hlo_string_relu = R"(
 HloModule top

 Sum-reduction.38 (x.39: f32[], y.40: f32[]) -> f32[] {
   y.40 = f32[] parameter(1)
   x.39 = f32[] parameter(0)
   ROOT add.41 = f32[] add(f32[] x.39, f32[] y.40)
 }

 _pop_op_conv_with_reverse (arg_0: f32[1,4,4,2], arg_1: f32[1,1,2,2]) -> f32[1,4,4,2] {
   arg_0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
   arg_1 = f32[1,1,2,2]{3,2,1,0} parameter(1)
   reverse.64.clone = f32[1,1,2,2]{3,2,1,0} reverse(f32[1,1,2,2]{3,2,1,0} arg_1), dimensions={0,1}
   ROOT convolution.65.clone = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_0, f32[1,1,2,2]{3,2,1,0} reverse.64.clone), window={size=1x1}, dim_labels=b01f_01oi->b01f
 }

 _pop_op_conv_scaled_inplace (arg_0.5: f32[1,1,2,2], arg_1.3: f32[1,4,4,2], arg_2: f32[1,4,4,2]) -> f32[1,1,2,2] {
   arg_0.5 = f32[1,1,2,2]{3,2,1,0} parameter(0)
   arg_1.3 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   arg_2 = f32[1,4,4,2]{3,2,1,0} parameter(2)
   convolution.89.clone = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_1.3, f32[1,4,4,2]{3,2,1,0} arg_2), window={size=4x4}, dim_labels=f01b_i01o->01bf
   constant.55.clone = f32[] constant(0.1)
   broadcast.67.clone = f32[1,1,2,2]{3,2,1,0} broadcast(f32[] constant.55.clone), dimensions={}
   multiply.92.clone = f32[1,1,2,2]{3,2,1,0} multiply(f32[1,1,2,2]{3,2,1,0} convolution.89.clone, f32[1,1,2,2]{3,2,1,0} broadcast.67.clone)
   ROOT subtract.93.clone = f32[1,1,2,2]{3,2,1,0} subtract(f32[1,1,2,2]{3,2,1,0} arg_0.5, f32[1,1,2,2]{3,2,1,0} multiply.92.clone)
 }

 _pop_op_conv_scaled_inplace.1 (arg_0.6: f32[1,1,2,2], arg_1.4: f32[1,4,4,2], arg_2.1: f32[1,4,4,2]) -> f32[1,1,2,2] {
   arg_0.6 = f32[1,1,2,2]{3,2,1,0} parameter(0)
   arg_1.4 = f32[1,4,4,2]{3,2,1,0} parameter(1)
   arg_2.1 = f32[1,4,4,2]{3,2,1,0} parameter(2)
   convolution.63.clone = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} arg_1.4, f32[1,4,4,2]{3,2,1,0} arg_2.1), window={size=4x4}, dim_labels=f01b_i01o->01bf
   constant.55.clone.1 = f32[] constant(0.1)
   broadcast.67.clone.1 = f32[1,1,2,2]{3,2,1,0} broadcast(f32[] constant.55.clone.1), dimensions={}
   multiply.68.clone = f32[1,1,2,2]{3,2,1,0} multiply(f32[1,1,2,2]{3,2,1,0} convolution.63.clone, f32[1,1,2,2]{3,2,1,0} broadcast.67.clone.1)
   ROOT subtract.69.clone = f32[1,1,2,2]{3,2,1,0} subtract(f32[1,1,2,2]{3,2,1,0} arg_0.6, f32[1,1,2,2]{3,2,1,0} multiply.68.clone)
 }

 _pop_op_scaled_inplace (arg_0.7: f32[2], arg_1.5: f32[2]) -> f32[2] {
   arg_0.7 = f32[2]{0} parameter(0)
   arg_1.5 = f32[2]{0} parameter(1)
   constant.55.clone.2 = f32[] constant(0.1)
   broadcast.56.clone = f32[2]{0} broadcast(f32[] constant.55.clone.2), dimensions={}
   multiply.87.clone = f32[2]{0} multiply(f32[2]{0} arg_1.5, f32[2]{0} broadcast.56.clone)
   ROOT subtract.88.clone = f32[2]{0} subtract(f32[2]{0} arg_0.7, f32[2]{0} multiply.87.clone)
 }

 _pop_op_scaled_inplace.1 (arg_0.8: f32[2], arg_1.6: f32[2]) -> f32[2] {
   arg_0.8 = f32[2]{0} parameter(0)
   arg_1.6 = f32[2]{0} parameter(1)
   constant.55.clone.3 = f32[] constant(0.1)
   broadcast.56.clone.1 = f32[2]{0} broadcast(f32[] constant.55.clone.3), dimensions={}
   multiply.83.clone = f32[2]{0} multiply(f32[2]{0} arg_1.6, f32[2]{0} broadcast.56.clone.1)
   ROOT subtract.84.clone = f32[2]{0} subtract(f32[2]{0} arg_0.8, f32[2]{0} multiply.83.clone)
 }

 _pop_op_scaled_inplace.2 (arg_0.9: f32[2], arg_1.7: f32[2]) -> f32[2] {
   arg_0.9 = f32[2]{0} parameter(0)
   arg_1.7 = f32[2]{0} parameter(1)
   constant.55.clone.4 = f32[] constant(0.1)
   broadcast.56.clone.2 = f32[2]{0} broadcast(f32[] constant.55.clone.4), dimensions={}
   multiply.61.clone = f32[2]{0} multiply(f32[2]{0} arg_1.7, f32[2]{0} broadcast.56.clone.2)
   ROOT subtract.62.clone = f32[2]{0} subtract(f32[2]{0} arg_0.9, f32[2]{0} multiply.61.clone)
 }

 _pop_op_scaled_inplace.3 (arg_0.10: f32[2], arg_1.8: f32[2]) -> f32[2] {
   arg_0.10 = f32[2]{0} parameter(0)
   arg_1.8 = f32[2]{0} parameter(1)
   constant.55.clone.5 = f32[] constant(0.1)
   broadcast.56.clone.3 = f32[2]{0} broadcast(f32[] constant.55.clone.5), dimensions={}
   multiply.57.clone = f32[2]{0} multiply(f32[2]{0} arg_1.8, f32[2]{0} broadcast.56.clone.3)
   ROOT subtract.58.clone = f32[2]{0} subtract(f32[2]{0} arg_0.10, f32[2]{0} multiply.57.clone)
 }

 _pop_op_wide_const () -> f32[1,4,4,2] {
   constant.9.clone = f32[] constant(1)
   ROOT broadcast.10.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] constant.9.clone), dimensions={}
 }

 ENTRY top (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[2], arg3.4: f32[1,1,2,2], arg4.5: f32[2], arg5.6: f32[2], arg6.7: f32[1,1,2,2]) -> (f32[], f32[2], f32[2], f32[1,1,2,2], f32[2], f32[2], f32[1,1,2,2]) {
 arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0), metadata={op_name="XLA_Args"}
 arg6.7 = f32[1,1,2,2]{3,2,1,0} parameter(6), metadata={op_name="XLA_Args"}
convolution.11 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %arg0.1, f32[1,1,2,2]{3,2,1,0} %arg6.7), window={size=1x1}, dim_labels=b01f_01io->b01f
arg5.6 = f32[2]{0} parameter(5), control-predecessors={%convolution.11}, metadata={op_name="XLA_Args"}
arg4.5 = f32[2]{0} parameter(4), control-predecessors={%convolution.11}, metadata={op_name="XLA_Args"}
batch-norm-training.13 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[1,4,4,2]{3,2,1,0} %convolution.11, f32[2]{0} %arg5.6, f32[2]{0} %arg4.5), epsilon=0.001, feature_index=3
get-tuple-element.14 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.13), index=0
relu.1 = f32[1,4,4,2]{3,2,1,0} custom-call(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.14), custom_call_target="Relu", backend_config="{}"
arg3.4 = f32[1,1,2,2]{3,2,1,0} parameter(3), metadata={op_name="XLA_Args"}
convolution.23 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %relu.1, f32[1,1,2,2]{3,2,1,0} %arg3.4), window={size=1x1}, dim_labels=b01f_01io->b01f
arg2.3 = f32[2]{0} parameter(2), control-predecessors={%convolution.23}, metadata={op_name="XLA_Args"}
arg1.2 = f32[2]{0} parameter(1), control-predecessors={%convolution.23}, metadata={op_name="XLA_Args"}
batch-norm-training.25 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[1,4,4,2]{3,2,1,0} %convolution.23, f32[2]{0} %arg2.3, f32[2]{0} %arg1.2), epsilon=0.001, feature_index=3
get-tuple-element.26 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.25), index=0
relu.2 = f32[1,4,4,2]{3,2,1,0} custom-call(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.26),  custom_call_target="Relu", backend_config="{}"
constant.20 = f32[] constant(0)
reduce.42 = f32[] reduce(f32[1,4,4,2]{3,2,1,0} %relu.2, f32[] %constant.20), dimensions={0,1,2,3}, to_apply=%Sum-reduction.38
get-tuple-element.28 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.25), index=1
get-tuple-element.29 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.25), index=2
fusion.11 = f32[1,4,4,2]{3,2,1,0} fusion(), kind=kCustom, calls=%_pop_op_wide_const, backend_config="{}"
relugrad.1 = f32[1,4,4,2]{3,2,1,0} custom-call(f32[1,4,4,2]{3,2,1,0} %relu.2, f32[1,4,4,2]{3,2,1,0} %fusion.11), custom_call_target="ReluGrad", backend_config="{}"
batch-norm-grad.50 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[1,4,4,2]{3,2,1,0} %convolution.23, f32[2]{0} %arg2.3, f32[2]{0} %get-tuple-element.28, f32[2]{0} %get-tuple-element.29, f32[1,4,4,2]{3,2,1,0} %relugrad.1), epsilon=0.001, feature_index=3
get-tuple-element.53 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.50), index=2
fusion.10 = f32[2]{0} fusion(f32[2]{0} %arg1.2, f32[2]{0} %get-tuple-element.53), kind=kCustom, calls=%_pop_op_scaled_inplace.3, backend_config="{}"
get-tuple-element.52 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.50), index=1
fusion.9 = f32[2]{0} fusion(f32[2]{0} %arg2.3, f32[2]{0} %get-tuple-element.52), kind=kCustom, calls=%_pop_op_scaled_inplace.2, backend_config="{}"
get-tuple-element.51 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.50), index=0
fusion = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.51, f32[1,1,2,2]{3,2,1,0} %arg3.4), kind=kCustom, calls=%_pop_op_conv_with_reverse, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"1\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"1\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelInputFeatureDimension\":\"3\",\"kernelOutputFeatureDimension\":\"2\",\"kernelSpatialDimensions\":[\"0\",\"1\"],\"inputFeatureDimension\":\"3\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"1\",\"2\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
fusion.6 = f32[1,1,2,2]{3,2,1,0} fusion(f32[1,1,2,2]{3,2,1,0} %arg3.4, f32[1,4,4,2]{3,2,1,0} %relu.1, f32[1,4,4,2]{3,2,1,0} %get-tuple-element.51), kind=kCustom, calls=%_pop_op_conv_scaled_inplace.1, control-predecessors={%fusion}, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelOutputFeatureDimension\":\"3\",\"kernelSpatialDimensions\":[\"1\",\"2\"],\"inputBatchDimension\":\"3\",\"outputBatchDimension\":\"2\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"0\",\"1\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
get-tuple-element.16 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.13), index=1
get-tuple-element.17 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-training.13), index=2
relugrad.2 = f32[1,4,4,2]{3,2,1,0} custom-call(f32[1,4,4,2]{3,2,1,0} %relu.1, f32[1,4,4,2]{3,2,1,0} %fusion), custom_call_target="ReluGrad", backend_config="{}"
batch-norm-grad.76 = (f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[1,4,4,2]{3,2,1,0} %convolution.11, f32[2]{0} %arg5.6, f32[2]{0} %get-tuple-element.16, f32[2]{0} %get-tuple-element.17, f32[1,4,4,2]{3,2,1,0} %relugrad.2), epsilon=0.001, feature_index=3
get-tuple-element.79 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.76), index=2
fusion.8 = f32[2]{0} fusion(f32[2]{0} %arg4.5, f32[2]{0} %get-tuple-element.79), kind=kCustom, calls=%_pop_op_scaled_inplace.1, backend_config="{}"
get-tuple-element.78 = f32[2]{0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.76), index=1
fusion.7 = f32[2]{0} fusion(f32[2]{0} %arg5.6, f32[2]{0} %get-tuple-element.78), kind=kCustom, calls=%_pop_op_scaled_inplace, backend_config="{}"
get-tuple-element.77 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) %batch-norm-grad.76), index=0
fusion.5 = f32[1,1,2,2]{3,2,1,0} fusion(f32[1,1,2,2]{3,2,1,0} %arg6.7, f32[1,4,4,2]{3,2,1,0} %arg0.1, f32[1,4,4,2]{3,2,1,0} %get-tuple-element.77), kind=kCustom, calls=%_pop_op_conv_scaled_inplace, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelOutputFeatureDimension\":\"3\",\"kernelSpatialDimensions\":[\"1\",\"2\"],\"inputBatchDimension\":\"3\",\"outputBatchDimension\":\"2\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"0\",\"1\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
   ROOT %tuple.107 = (f32[], f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[1,1,2,2]{3,2,1,0}) tuple(f32[] %reduce.42, f32[2]{0} %fusion.10, f32[2]{0} %fusion.9, f32[1,1,2,2]{3,2,1,0} %fusion.6, f32[2]{0} %fusion.8, f32[2]{0} %fusion.7, f32[1,1,2,2]{3,2,1,0} %fusion.5), metadata={op_name="XLA_Retvals"}}
    )";

TEST_F(NormInputRecomputationTest, RecomputeRelu) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({1, 2, 3, 4, 5, 6});
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string_relu, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);
  CustomOpReplacer replacer{};

  auto replacer_res = replacer.Run(module);

  EXPECT_TRUE(replacer_res.ok());
  EXPECT_TRUE(replacer_res.ValueOrDie());

  HloInstruction* input1 = FindInstruction(module, "convolution.11");
  HloInstruction* input2 = FindInstruction(module, "convolution.23");

  HloInstruction* bn1 = FindInstruction(module, "batch-norm-training.13");
  HloInstruction* bn2 = FindInstruction(module, "batch-norm-training.25");

  HloInstruction* bn_grad1 = FindInstruction(module, "batch-norm-grad.76");
  HloInstruction* bn_grad2 = FindInstruction(module, "batch-norm-grad.50");

  HloInstruction* relu1 = input2->mutable_operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::Relu)(relu1));

  HloInstruction* relu_grad2 = bn_grad1->mutable_operand(4);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReluGrad)(relu_grad2));

  ASSERT_EQ(input1->users().size(), 2);
  ASSERT_EQ(bn1->operand(0), input1);
  ASSERT_EQ(bn_grad1->operand(0), input1);

  ASSERT_EQ(input2->users().size(), 2);
  ASSERT_EQ(bn2->operand(0), input2);
  ASSERT_EQ(bn_grad2->operand(0), input2);

  ASSERT_EQ(relu_grad2->operand(0), relu1);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  ASSERT_TRUE(RecomputeInstructions(true).Run(module).ValueOrDie());

  ASSERT_EQ(bn1->operand(0), input1);
  ASSERT_NE(bn1->operand(0), bn_grad1->operand(0));
  ASSERT_TRUE(bn1->operand(0)->Identical(*bn_grad1->operand(0)));

  ASSERT_EQ(bn2->operand(0), input2);
  ASSERT_NE(bn2->operand(0), bn_grad2->operand(0));
  ASSERT_TRUE(bn2->operand(0)->Identical(*bn_grad2->operand(0)));

  // Check that the relu has been replaced
  ASSERT_NE(relu_grad2->operand(0), relu1);
}

TEST_F(NormInputRecomputationTest, RecomputeReluOff) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  config.set_resource_update_to_input_index({1, 2, 3, 4, 5, 6});
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string_relu, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  ASSERT_FALSE(RecomputeInstructions(false).Run(module).ValueOrDie());
}

TEST_F(NormInputRecomputationTest, RecomputeF16WithCast) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  std::string module_string = R"(
   HloModule top

   _pop_op_wide_const () -> f16[32,32,32,1] {
     singleconst = f16[] constant(1)
     ROOT broadcastconst = f16[32,32,32,1] broadcast(f16[] singleconst), dimensions={}
   }

   ENTRY top (arg0: f16[32,32,32,3], arg1: f16[1,1,3,1], arg2: f32[1], arg3: f16[1]) -> (f16[32,32,32,1]) {
     arg0 = f16[32,32,32,3] parameter(0)
     arg1 = f16[1,1,3,1] parameter(1)
     arg2 = f32[1] parameter(2)
     arg3 = f16[1] parameter(3)
     gamma16 = f16[1] convert(f32[1] %arg2), metadata={op_type="Cast" op_name="gamma_cast"}
     convolution = f16[32,32,32,1]{3,2,1,0} convolution(f16[32,32,32,3]{3,2,1,0} %arg0, f16[1,1,3,1]{3,2,1,0} %arg1), window={size=1x1}, dim_labels=b01f_01io->b01f
     batch-norm-training = (f16[32,32,32,1], f16[1], f16[1]) batch-norm-training(f16[32,32,32,1] %convolution, f16[1] %gamma16, f16[1] %arg3), epsilon=0.001, feature_index=3
     gte0 = f16[32,32,32,1] get-tuple-element((f16[32,32,32,1], f16[1], f16[1]) %batch-norm-training), index=0
     relu = f16[32,32,32,1] custom-call(f16[32,32,32,1] %gte0), custom_call_target="Relu", backend_config="{}"
     gte1 = f16[1] get-tuple-element((f16[32,32,32,1], f16[1], f16[1]) %batch-norm-training), index=1
     gte2 = f16[1] get-tuple-element((f16[32,32,32,1], f16[1], f16[1]) %batch-norm-training), index=2
     zeros = f16[32,32,32,1] fusion(), kind=kCustom, calls=%_pop_op_wide_const, backend_config="{}"
     relugrad = f16[32,32,32,1] custom-call(f16[32,32,32,1] relu, f16[32,32,32,1] %zeros), custom_call_target="ReluGrad", backend_config="{}"
     batch-norm-grad = (f16[32,32,32,1], f16[1], f16[1]) batch-norm-grad(f16[32,32,32,1] %convolution, f16[1] %gamma16, f16[1] %gte1, f16[1] %gte2, f16[32,32,32,1] %relugrad), epsilon=0.001, feature_index=3
     gte-grad0 = f16[32,32,32,1] get-tuple-element((f16[32,32,32,1], f16[1], f16[1]) %batch-norm-grad), index=0
   ROOT %tuple = (f16[32,32,32,1]) tuple(f16[32,32,32,1] %gte-grad0), metadata={op_name="XLA_Retvals"}}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(module_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ModuleFlatten flatten(annotations);
  ConvolutionClassifier classifier(annotations);

  EXPECT_TRUE(flatten.Run(module).ValueOrDie());
  auto res = classifier.Run(module);

  EXPECT_TRUE(res.ok());
  EXPECT_TRUE(res.ValueOrDie());

  ASSERT_TRUE(RecomputeInstructions(true).Run(module).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
