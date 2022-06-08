/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_PIPELINE_GROUPED_RECOMPUTATION_HLO_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_PIPELINE_GROUPED_RECOMPUTATION_HLO_H_

const char* pipeline_grouped_recomputation_hlo = R"(
HloModule cluster_1183973981639933248_f15n_0__.382

%Sum-reduction.14 (x.15: f32[], y.16: f32[]) -> f32[] {
  %x.15 = f32[] parameter(0)
  %y.16 = f32[] parameter(1)
  ROOT %add.17 = f32[] add(f32[] %x.15, f32[] %y.16)
}

%max_float_.18 (x.19: f32[], y.20: f32[]) -> f32[] {
  %x.19 = f32[] parameter(0)
  %y.20 = f32[] parameter(1)
  ROOT %maximum.21 = f32[] maximum(f32[] %x.19, f32[] %y.20)
}

%add_float_.22 (x.23: f32[], y.24: f32[]) -> f32[] {
  %x.23 = f32[] parameter(0)
  %y.24 = f32[] parameter(1)
  ROOT %add.25 = f32[] add(f32[] %x.23, f32[] %y.24)
}

%add_float_.26 (x.27: f32[], y.28: f32[]) -> f32[] {
  %x.27 = f32[] parameter(0)
  %y.28 = f32[] parameter(1)
  ROOT %add.29 = f32[] add(f32[] %x.27, f32[] %y.28)
}

%Mean-reduction.30 (x.31: f32[], y.32: f32[]) -> f32[] {
  %x.31 = f32[] parameter(0)
  %y.32 = f32[] parameter(1)
  ROOT %add.33 = f32[] add(f32[] %x.31, f32[] %y.32)
}

%gradients_MatMul_2_grad_Sum_1-reduction.34 (x.35: f32[], y.36: f32[]) -> f32[] {
  %x.35 = f32[] parameter(0)
  %y.36 = f32[] parameter(1)
  ROOT %add.37 = f32[] add(f32[] %x.35, f32[] %y.36)
}

%gradients_MatMul_1_grad_Sum_1-reduction.38 (x.39: f32[], y.40: f32[]) -> f32[] {
  %x.39 = f32[] parameter(0)
  %y.40 = f32[] parameter(1)
  ROOT %add.41 = f32[] add(f32[] %x.39, f32[] %y.40)
}

%gradients_MatMul_grad_Sum_1-reduction.42 (x.43: f32[], y.44: f32[]) -> f32[] {
  %x.43 = f32[] parameter(0)
  %y.44 = f32[] parameter(1)
  ROOT %add.45 = f32[] add(f32[] %x.43, f32[] %y.44)
}

%Momentum_WU_func_509_rearrange_0__.46 (arg0.47: f32[4,4], arg1.48: f32[4,4], arg2.49: f32[4,4], arg3.50: f32[4,4], arg4.51: f32[4,4], arg5.52: f32[4,4], arg6.53: f32[4,4], arg7.54: f32[4,4], arg8.55: f32[4,4]) -> (f32[4,4], f32[4,4], f32[4,4], f32[4,4], f32[4,4], /*index=5*/f32[4,4]) {
  %constant.80 = s32[] constant(24), metadata={op_type="GradientAccumulationCount" op_name="GradientAccumulationCount"}
  %custom-call.81 = () custom-call(s32[] %constant.80), custom_call_target="GradientAccumulationCount", metadata={op_type="GradientAccumulationCount" op_name="GradientAccumulationCount"}
  %arg3.50 = f32[4,4]{1,0} parameter(3), metadata={op_name="XLA_Args/ipu_sharded/s1/w"}
  %arg4.51 = f32[4,4]{1,0} parameter(4), metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s1/w/Momentum"}
  %constant.57 = f32[] constant(0.98), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %broadcast.58 = f32[4,4]{1,0} broadcast(f32[] %constant.57), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %multiply.59 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %arg4.51, f32[4,4]{1,0} %broadcast.58), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %arg0.47 = f32[4,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %add.60 = f32[4,4]{1,0} add(f32[4,4]{1,0} %multiply.59, f32[4,4]{1,0} %arg0.47), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %constant.56 = f32[] constant(0.01), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %broadcast.61 = f32[4,4]{1,0} broadcast(f32[] %constant.56), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %multiply.62 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %add.60, f32[4,4]{1,0} %broadcast.61), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %subtract.63 = f32[4,4]{1,0} subtract(f32[4,4]{1,0} %arg3.50, f32[4,4]{1,0} %multiply.62), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s1/w/ResourceApplyMomentum"}
  %tuple.82 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %subtract.63), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.83 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.82), index=0, metadata={op_name="XLA_Retvals"}
  %tuple.84 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.60), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.85 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.84), index=0, metadata={op_name="XLA_Retvals"}
  %arg5.52 = f32[4,4]{1,0} parameter(5), metadata={op_name="XLA_Args/ipu_sharded/s2/w"}
  %arg6.53 = f32[4,4]{1,0} parameter(6), metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s2/w/Momentum"}
  %constant.65 = f32[] constant(0.98), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %broadcast.66 = f32[4,4]{1,0} broadcast(f32[] %constant.65), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %multiply.67 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %arg6.53, f32[4,4]{1,0} %broadcast.66), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %arg1.48 = f32[4,4]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %add.68 = f32[4,4]{1,0} add(f32[4,4]{1,0} %multiply.67, f32[4,4]{1,0} %arg1.48), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %constant.64 = f32[] constant(0.01), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %broadcast.69 = f32[4,4]{1,0} broadcast(f32[] %constant.64), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %multiply.70 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %add.68, f32[4,4]{1,0} %broadcast.69), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %subtract.71 = f32[4,4]{1,0} subtract(f32[4,4]{1,0} %arg5.52, f32[4,4]{1,0} %multiply.70), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s2/w/ResourceApplyMomentum"}
  %tuple.86 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %subtract.71), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.87 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.86), index=0, metadata={op_name="XLA_Retvals"}
  %tuple.88 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.68), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.89 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.88), index=0, metadata={op_name="XLA_Retvals"}
  %arg7.54 = f32[4,4]{1,0} parameter(7), metadata={op_name="XLA_Args/ipu_sharded/s3/w"}
  %arg8.55 = f32[4,4]{1,0} parameter(8), metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s3/w/Momentum"}
  %constant.73 = f32[] constant(0.98), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %broadcast.74 = f32[4,4]{1,0} broadcast(f32[] %constant.73), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %multiply.75 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %arg8.55, f32[4,4]{1,0} %broadcast.74), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %arg2.49 = f32[4,4]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %add.76 = f32[4,4]{1,0} add(f32[4,4]{1,0} %multiply.75, f32[4,4]{1,0} %arg2.49), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %constant.72 = f32[] constant(0.01), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %broadcast.77 = f32[4,4]{1,0} broadcast(f32[] %constant.72), dimensions={}, metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %multiply.78 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %add.76, f32[4,4]{1,0} %broadcast.77), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %subtract.79 = f32[4,4]{1,0} subtract(f32[4,4]{1,0} %arg7.54, f32[4,4]{1,0} %multiply.78), metadata={op_type="ResourceApplyMomentum" op_name="Momentum/update_ipu_sharded/s3/w/ResourceApplyMomentum"}
  %tuple.90 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %subtract.79), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.91 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.90), index=0, metadata={op_name="XLA_Retvals"}
  %tuple.92 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.76), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.93 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.92), index=0, metadata={op_name="XLA_Retvals"}
  ROOT %tuple.94 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.83, f32[4,4]{1,0} %get-tuple-element.85, f32[4,4]{1,0} %get-tuple-element.87, f32[4,4]{1,0} %get-tuple-element.89, f32[4,4]{1,0} %get-tuple-element.91, /*index=5*/f32[4,4]{1,0} %get-tuple-element.93), metadata={op_name="XLA_Retvals"}
}

%_functionalize_body_0_const_0__.95 (arg_tuple.96: (s32[], s32[], s32[], f32[4,4], f32[4,4], /*index=5*/f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> (s32[], s32[], s32[], f32[4,4], f32[4,4], /*index=5*/f32[4,4], f32[4,4], f32[4,4], f32[4,4]) {
  %arg_tuple.96 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0), metadata={op_name="XLA_Args"}
  %get-tuple-element.98 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=1
  %constant.112 = s32[] constant(1), metadata={op_type="AddV2" op_name="add"}
  %constant.115 = s32[] constant(24), metadata={op_type="FloorMod" op_name="mod"}
  %constant.127 = f32[] constant(1), metadata={op_type="AddV2" op_name="add_1"}
  %constant.130 = f32[] constant(1), metadata={op_type="AddV2" op_name="add_2"}
  %constant.134 = f32[] constant(1), metadata={op_type="RealDiv" op_name="truediv_1"}
  %after-all.107 = token[] after-all(), metadata={op_type="PopDatastreamInfeedDequeue" op_name="PopDatastreamInfeedDequeue"}
  %infeed.108 = ((f32[2,4,4]{2,1,0}, s32[2]{0}), token[]) infeed(token[] %after-all.107), infeed_config="\022\0011\"\002\001\003(\003", metadata={op_type="PopDatastreamInfeedDequeue" op_name="PopDatastreamInfeedDequeue"}
  %get-tuple-element.109 = (f32[2,4,4]{2,1,0}, s32[2]{0}) get-tuple-element(((f32[2,4,4]{2,1,0}, s32[2]{0}), token[]) %infeed.108), index=0, metadata={op_type="PopDatastreamInfeedDequeue" op_name="PopDatastreamInfeedDequeue"}
  %get-tuple-element.111 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}) %get-tuple-element.109), index=1, metadata={op_type="PopDatastreamInfeedDequeue" op_name="PopDatastreamInfeedDequeue"}
  %broadcast.174 = s32[2,4]{1,0} broadcast(s32[2]{0} %get-tuple-element.111), dimensions={0}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %iota.173 = s32[2,4]{1,0} iota(), iota_dimension=1, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %compare.175 = pred[2,4]{1,0} compare(s32[2,4]{1,0} %broadcast.174, s32[2,4]{1,0} %iota.173), direction=EQ, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.170 = f32[] constant(1), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.172 = f32[2,4]{1,0} broadcast(f32[] %constant.170), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.169 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.171 = f32[2,4]{1,0} broadcast(f32[] %constant.169), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %select.176 = f32[2,4]{1,0} select(pred[2,4]{1,0} %compare.175, f32[2,4]{1,0} %broadcast.172, f32[2,4]{1,0} %broadcast.171), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.184 = s32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.185 = s32[2]{0} broadcast(s32[] %constant.184), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %compare.186 = pred[2]{0} compare(s32[2]{0} %broadcast.185, s32[2]{0} %get-tuple-element.111), direction=LE, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.181 = s32[] constant(4), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.182 = s32[2]{0} broadcast(s32[] %constant.181), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %compare.183 = pred[2]{0} compare(s32[2]{0} %get-tuple-element.111, s32[2]{0} %broadcast.182), direction=LT, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %and.187 = pred[2]{0} and(pred[2]{0} %compare.186, pred[2]{0} %compare.183), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.179 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.180 = f32[2]{0} broadcast(f32[] %constant.179), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.177 = f32[] constant(nan), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.178 = f32[2]{0} broadcast(f32[] %constant.177), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %select.188 = f32[2]{0} select(pred[2]{0} %and.187, f32[2]{0} %broadcast.180, f32[2]{0} %broadcast.178), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.189 = f32[2,4]{1,0} broadcast(f32[2]{0} %select.188), dimensions={0}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %add.190 = f32[2,4]{1,0} add(f32[2,4]{1,0} %select.176, f32[2,4]{1,0} %broadcast.189), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %negate.209 = f32[2,4]{1,0} negate(f32[2,4]{1,0} %add.190), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.205 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.206 = f32[2,4]{1,0} broadcast(f32[] %constant.205), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %compare.207 = pred[2,4]{1,0} compare(f32[2,4]{1,0} %add.190, f32[2,4]{1,0} %broadcast.206), direction=EQ, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.203 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.204 = f32[2,4]{1,0} broadcast(f32[] %constant.203), dimensions={}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %get-tuple-element.110 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}) %get-tuple-element.109), index=0, metadata={op_type="PopDatastreamInfeedDequeue" op_name="PopDatastreamInfeedDequeue"}
  %get-tuple-element.100 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=3
  %dot.138 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.110, f32[4,4]{1,0} %get-tuple-element.100), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={maximal device=0}, metadata={op_type="BatchMatMulV2" op_name="MatMul"}
  %transpose.139 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.138), dimensions={0,1,2}, sharding={maximal device=0}, metadata={op_type="BatchMatMulV2" op_name="MatMul"}
  %constant.140 = s32[] constant(10), sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}
  %broadcast.141 = s32[2]{0} broadcast(s32[] %constant.140), dimensions={}, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}
  %custom-call.142 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %transpose.139, s32[2]{0} %broadcast.141), custom_call_target="Dropout", sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.143 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.142), index=0, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}
  %get-tuple-element.101 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=4
  %dot.147 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.143, f32[4,4]{1,0} %get-tuple-element.101), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="MatMul_1"}
  %transpose.148 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.147), dimensions={0,1,2}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="MatMul_1"}
  %constant.149 = s32[] constant(10), sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}
  %broadcast.150 = s32[2]{0} broadcast(s32[] %constant.149), dimensions={}, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}
  %custom-call.151 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %transpose.148, s32[2]{0} %broadcast.150), custom_call_target="Dropout", sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.152 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.151), index=0, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}
  %get-tuple-element.102 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=5
  %dot.156 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.152, f32[4,4]{1,0} %get-tuple-element.102), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="MatMul_2"}
  %transpose.157 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.156), dimensions={0,1,2}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="MatMul_2"}
  %constant.158 = s32[] constant(10), sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}
  %broadcast.159 = s32[2]{0} broadcast(s32[] %constant.158), dimensions={}, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}
  %custom-call.160 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %transpose.157, s32[2]{0} %broadcast.159), custom_call_target="Dropout", sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.161 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.160), index=0, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}
  %convert.164 = f32[2,4,4]{2,1,0} convert(f32[2,4,4]{2,1,0} %get-tuple-element.161), sharding={maximal device=2}, metadata={op_type="Sum" op_name="Sum"}
  %constant.165 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="Sum" op_name="Sum"}
  %convert.166 = f32[] convert(f32[] %constant.165), sharding={maximal device=2}, metadata={op_type="Sum" op_name="Sum"}
  %reduce.167 = f32[2,4]{1,0} reduce(f32[2,4,4]{2,1,0} %convert.164, f32[] %convert.166), dimensions={1}, to_apply=%Sum-reduction.14, sharding={maximal device=2}, metadata={op_type="Sum" op_name="Sum"}
  %convert.168 = f32[2,4]{1,0} convert(f32[2,4]{1,0} %reduce.167), sharding={maximal device=2}, metadata={op_type="Sum" op_name="Sum"}
  %constant.191 = f32[] constant(-inf), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %reduce.192 = f32[2]{0} reduce(f32[2,4]{1,0} %convert.168, f32[] %constant.191), dimensions={1}, to_apply=%max_float_.18, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.193 = f32[2,4]{1,0} broadcast(f32[2]{0} %reduce.192), dimensions={0}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %subtract.194 = f32[2,4]{1,0} subtract(f32[2,4]{1,0} %convert.168, f32[2,4]{1,0} %broadcast.193), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %exponential.195 = f32[2,4]{1,0} exponential(f32[2,4]{1,0} %subtract.194), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %convert.196 = f32[2,4]{1,0} convert(f32[2,4]{1,0} %exponential.195), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.197 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %reduce.198 = f32[2]{0} reduce(f32[2,4]{1,0} %convert.196, f32[] %constant.197), dimensions={1}, to_apply=%add_float_.22, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %convert.199 = f32[2]{0} convert(f32[2]{0} %reduce.198), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %log.200 = f32[2]{0} log(f32[2]{0} %convert.199), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %broadcast.201 = f32[2,4]{1,0} broadcast(f32[2]{0} %log.200), dimensions={0}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %subtract.202 = f32[2,4]{1,0} subtract(f32[2,4]{1,0} %subtract.194, f32[2,4]{1,0} %broadcast.201), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %select.208 = f32[2,4]{1,0} select(pred[2,4]{1,0} %compare.207, f32[2,4]{1,0} %broadcast.204, f32[2,4]{1,0} %subtract.202), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %multiply.210 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %negate.209, f32[2,4]{1,0} %select.208), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %convert.212 = f32[2,4]{1,0} convert(f32[2,4]{1,0} %multiply.210), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %constant.211 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %reduce.213 = f32[2]{0} reduce(f32[2,4]{1,0} %convert.212, f32[] %constant.211), dimensions={1}, to_apply=%add_float_.26, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %convert.214 = f32[2]{0} convert(f32[2]{0} %reduce.213), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %convert.218 = f32[2]{0} convert(f32[2]{0} %convert.214), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %constant.219 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %convert.220 = f32[] convert(f32[] %constant.219), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %reduce.221 = f32[] reduce(f32[2]{0} %convert.218, f32[] %convert.220), dimensions={0}, to_apply=%Mean-reduction.30, sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %constant.222 = s32[] constant(2), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %convert.223 = f32[] convert(s32[] %constant.222), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %divide.224 = f32[] divide(f32[] %reduce.221, f32[] %convert.223), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %convert.225 = f32[] convert(f32[] %divide.224), sharding={maximal device=2}, metadata={op_type="Mean" op_name="Mean"}
  %constant.228 = f32[] constant(0.5), sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %broadcast.229 = f32[2,1]{1,0} broadcast(f32[] %constant.228), dimensions={}, sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %constant.235 = s32[3]{0} constant({2, 1, 4}), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/Sum_grad/Reshape"}
  %constant.237 = s32[3]{0} constant({1, 4, 1}), sharding={maximal device=2}, metadata={op_type="Tile" op_name="gradients/Sum_grad/Tile"}
  %constant.230 = f32[] constant(0.5), sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %broadcast.231 = f32[2,1]{1,0} broadcast(f32[] %constant.230), dimensions={}, sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %reshape.232 = f32[2]{0} reshape(f32[2,1]{1,0} %broadcast.231), sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %broadcast.233 = f32[2,4]{1,0} broadcast(f32[2]{0} %reshape.232), dimensions={0}, sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %broadcast.215 = f32[2,4]{1,0} broadcast(f32[2]{0} %convert.199), dimensions={0}, sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %divide.216 = f32[2,4]{1,0} divide(f32[2,4]{1,0} %exponential.195, f32[2,4]{1,0} %broadcast.215), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %subtract.217 = f32[2,4]{1,0} subtract(f32[2,4]{1,0} %divide.216, f32[2,4]{1,0} %add.190), sharding={maximal device=2}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"}
  %multiply.234 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast.233, f32[2,4]{1,0} %subtract.217), sharding={maximal device=2}, metadata={op_type="Mul" op_name="gradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul"}
  %reshape.236 = f32[2,1,4]{2,1,0} reshape(f32[2,4]{1,0} %multiply.234), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/Sum_grad/Reshape"}
  %broadcast.238 = f32[2,1,4]{2,1,0} broadcast(f32[2,1,4]{2,1,0} %reshape.236), dimensions={0,1,2}, sharding={maximal device=2}, metadata={op_type="Tile" op_name="gradients/Sum_grad/Tile"}
  %reshape.239 = f32[2,4]{1,0} reshape(f32[2,1,4]{2,1,0} %broadcast.238), sharding={maximal device=2}, metadata={op_type="Tile" op_name="gradients/Sum_grad/Tile"}
  %broadcast.240 = f32[2,4,4]{2,1,0} broadcast(f32[2,4]{1,0} %reshape.239), dimensions={0,2}, sharding={maximal device=2}, metadata={op_type="Tile" op_name="gradients/Sum_grad/Tile"}
  %get-tuple-element.162 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.160), index=1, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}
  %get-tuple-element.163 = opaque[] get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.160), index=2, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_2"}
  %custom-call.241 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %broadcast.240, s32[2]{0} %get-tuple-element.162, opaque[] %get-tuple-element.163), custom_call_target="Dropout", sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_2_grad/IpuDropoutWithSeedAndReference"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.243 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.241), index=1, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_2_grad/IpuDropoutWithSeedAndReference"}
  %constant.251 = s32[] constant(4), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/MatMul_2_grad/Reshape_1"}
  %broadcast.252 = s32[2]{0} broadcast(s32[] %constant.251), dimensions={}, sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/MatMul_2_grad/Reshape_1"}
  %constant.260 = s32[3]{0} constant({2, 4, 4}), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/MatMul_2_grad/Reshape"}
  %get-tuple-element.242 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.241), index=0, sharding={maximal device=2}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_2_grad/IpuDropoutWithSeedAndReference"}
  %dot.258 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.242, f32[4,4]{1,0} %get-tuple-element.102), lhs_contracting_dims={2}, rhs_contracting_dims={1}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_2_grad/MatMul"}
  %transpose.259 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.258), dimensions={0,1,2}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_2_grad/MatMul"}
  %reshape.261 = f32[2,4,4]{2,1,0} reshape(f32[2,4,4]{2,1,0} %transpose.259), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/MatMul_2_grad/Reshape"}
  %get-tuple-element.153 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.151), index=1, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}
  %get-tuple-element.154 = opaque[] get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.151), index=2, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed_1"}
  %custom-call.262 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %reshape.261, s32[2]{0} %get-tuple-element.153, opaque[] %get-tuple-element.154), custom_call_target="Dropout", sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_1_grad/IpuDropoutWithSeedAndReference"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.264 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.262), index=1, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_1_grad/IpuDropoutWithSeedAndReference"}
  %constant.267 = s32[3]{0} constant({2, 4, 4}), sharding={maximal device=1}, metadata={op_type="Reshape" op_name="gradients/MatMul_1_grad/Reshape"}
  %constant.276 = s32[] constant(4), sharding={maximal device=1}, metadata={op_type="Reshape" op_name="gradients/MatMul_1_grad/Reshape_1"}
  %broadcast.277 = s32[2]{0} broadcast(s32[] %constant.276), dimensions={}, sharding={maximal device=1}, metadata={op_type="Reshape" op_name="gradients/MatMul_1_grad/Reshape_1"}
  %get-tuple-element.263 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.262), index=0, sharding={maximal device=1}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_1_grad/IpuDropoutWithSeedAndReference"}
  %dot.265 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.263, f32[4,4]{1,0} %get-tuple-element.101), lhs_contracting_dims={2}, rhs_contracting_dims={1}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_1_grad/MatMul"}
  %transpose.266 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.265), dimensions={0,1,2}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_1_grad/MatMul"}
  %reshape.268 = f32[2,4,4]{2,1,0} reshape(f32[2,4,4]{2,1,0} %transpose.266), sharding={maximal device=1}, metadata={op_type="Reshape" op_name="gradients/MatMul_1_grad/Reshape"}
  %get-tuple-element.144 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.142), index=1, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}
  %get-tuple-element.145 = opaque[] get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.142), index=2, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeed" op_name="IpuDropoutWithSeed"}
  %custom-call.279 = (f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) custom-call(f32[2,4,4]{2,1,0} %reshape.268, s32[2]{0} %get-tuple-element.144, opaque[] %get-tuple-element.145), custom_call_target="Dropout", sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_grad/IpuDropoutWithSeedAndReference"}, backend_config="{\n\t\"rate\" : 0.5,\n\t\"scale\" : 2.0\n}"
  %get-tuple-element.281 = s32[2]{0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.279), index=1, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_grad/IpuDropoutWithSeedAndReference"}
  %constant.289 = s32[] constant(4), sharding={maximal device=0}, metadata={op_type="Reshape" op_name="gradients/MatMul_grad/Reshape_1"}
  %broadcast.290 = s32[2]{0} broadcast(s32[] %constant.289), dimensions={}, sharding={maximal device=0}, metadata={op_type="Reshape" op_name="gradients/MatMul_grad/Reshape_1"}
  %get-tuple-element.97 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=0
  %constant.113 = s32[] constant(1), metadata={op_type="AddV2" op_name="add"}
  %add.114 = s32[] add(s32[] %get-tuple-element.97, s32[] %constant.113), metadata={op_type="AddV2" op_name="add"}
  %constant.307 = s32[] constant(0)
  %get-tuple-element.99 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=2
  %custom-call.137 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %get-tuple-element.100), custom_call_target="GradientAccumulatorCreate", metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate"}, backend_config="null"
  %get-tuple-element.280 = f32[2,4,4]{2,1,0} get-tuple-element((f32[2,4,4]{2,1,0}, s32[2]{0}, opaque[]) %custom-call.279), index=0, sharding={maximal device=0}, metadata={op_type="IpuDropoutWithSeedAndReference" op_name="gradients/IpuDropoutWithSeed_grad/IpuDropoutWithSeedAndReference"}
  %dot.282 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.110, f32[2,4,4]{2,1,0} %get-tuple-element.280), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, sharding={maximal device=0}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_grad/MatMul_1"}
  %transpose.283 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.282), dimensions={0,1,2}, sharding={maximal device=0}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_grad/MatMul_1"}
  %convert.284 = f32[2,4,4]{2,1,0} convert(f32[2,4,4]{2,1,0} %transpose.283), sharding={maximal device=0}, metadata={op_type="Sum" op_name="gradients/MatMul_grad/Sum_1"}
  %constant.285 = f32[] constant(0), sharding={maximal device=0}, metadata={op_type="Sum" op_name="gradients/MatMul_grad/Sum_1"}
  %convert.286 = f32[] convert(f32[] %constant.285), sharding={maximal device=0}, metadata={op_type="Sum" op_name="gradients/MatMul_grad/Sum_1"}
  %reduce.287 = f32[4,4]{1,0} reduce(f32[2,4,4]{2,1,0} %convert.284, f32[] %convert.286), dimensions={0}, to_apply=%gradients_MatMul_grad_Sum_1-reduction.42, sharding={maximal device=0}, metadata={op_type="Sum" op_name="gradients/MatMul_grad/Sum_1"}
  %convert.288 = f32[4,4]{1,0} convert(f32[4,4]{1,0} %reduce.287), sharding={maximal device=0}, metadata={op_type="Sum" op_name="gradients/MatMul_grad/Sum_1"}
  %reshape.291 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %convert.288), sharding={maximal device=0}, metadata={op_type="Reshape" op_name="gradients/MatMul_grad/Reshape_1"}
  %constant.135 = f32[] constant(1), metadata={op_type="RealDiv" op_name="truediv_1"}
  %custom-call.106 = s32[] custom-call(), custom_call_target="ExecutionCounter", metadata={op_type="ExecutionCounter" op_name="ExecutionCounter"}, backend_config="{\n\t\"lower_into_pipeline_stage\" : false\n}"
  %constant.116 = s32[] constant(24), metadata={op_type="FloorMod" op_name="mod"}
  %remainder.118 = s32[] remainder(s32[] %custom-call.106, s32[] %constant.116), metadata={op_type="FloorMod" op_name="mod"}
  %constant.117 = s32[] constant(0), metadata={op_type="FloorMod" op_name="mod"}
  %compare.121 = pred[] compare(s32[] %remainder.118, s32[] %constant.117), direction=LT, metadata={op_type="FloorMod" op_name="mod"}
  %compare.120 = pred[] compare(s32[] %constant.116, s32[] %constant.117), direction=LT, metadata={op_type="FloorMod" op_name="mod"}
  %compare.122 = pred[] compare(pred[] %compare.121, pred[] %compare.120), direction=NE, metadata={op_type="FloorMod" op_name="mod"}
  %compare.119 = pred[] compare(s32[] %remainder.118, s32[] %constant.117), direction=NE, metadata={op_type="FloorMod" op_name="mod"}
  %and.123 = pred[] and(pred[] %compare.122, pred[] %compare.119), metadata={op_type="FloorMod" op_name="mod"}
  %add.124 = s32[] add(s32[] %remainder.118, s32[] %constant.116), metadata={op_type="FloorMod" op_name="mod"}
  %select.125 = s32[] select(pred[] %and.123, s32[] %add.124, s32[] %remainder.118), metadata={op_type="FloorMod" op_name="mod"}
  %convert.126 = f32[] convert(s32[] %select.125), metadata={op_type="Cast" op_name="Cast"}
  %constant.131 = f32[] constant(1), metadata={op_type="AddV2" op_name="add_2"}
  %add.132 = f32[] add(f32[] %convert.126, f32[] %constant.131), metadata={op_type="AddV2" op_name="add_2"}
  %divide.136 = f32[] divide(f32[] %constant.135, f32[] %add.132), metadata={op_type="RealDiv" op_name="truediv_1"}
  %broadcast.292 = f32[4,4]{1,0} broadcast(f32[] %divide.136), dimensions={}, metadata={op_type="Mul" op_name="mul"}
  %multiply.293 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %reshape.291, f32[4,4]{1,0} %broadcast.292), metadata={op_type="Mul" op_name="mul"}
  %constant.128 = f32[] constant(1), metadata={op_type="AddV2" op_name="add_1"}
  %add.129 = f32[] add(f32[] %convert.126, f32[] %constant.128), metadata={op_type="AddV2" op_name="add_1"}
  %divide.133 = f32[] divide(f32[] %convert.126, f32[] %add.129), metadata={op_type="RealDiv" op_name="truediv"}
  %custom-call.294 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.137, f32[4,4]{1,0} %multiply.293, f32[] %divide.133), custom_call_target="GradientAccumulatorAddWithScale", metadata={op_type="GradientAccumulatorAddWithScale" op_name="GradientAccumulatorAddWithScale"}, backend_config="null"
  %custom-call.295 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.294), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink"}, backend_config="null"
  %custom-call.146 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %get-tuple-element.101), custom_call_target="GradientAccumulatorCreate", metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate_1"}, backend_config="null"
  %dot.269 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.143, f32[2,4,4]{2,1,0} %get-tuple-element.263), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %transpose.270 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.269), dimensions={0,1,2}, sharding={maximal device=1}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %convert.271 = f32[2,4,4]{2,1,0} convert(f32[2,4,4]{2,1,0} %transpose.270), sharding={maximal device=1}, metadata={op_type="Sum" op_name="gradients/MatMul_1_grad/Sum_1"}
  %constant.272 = f32[] constant(0), sharding={maximal device=1}, metadata={op_type="Sum" op_name="gradients/MatMul_1_grad/Sum_1"}
  %convert.273 = f32[] convert(f32[] %constant.272), sharding={maximal device=1}, metadata={op_type="Sum" op_name="gradients/MatMul_1_grad/Sum_1"}
  %reduce.274 = f32[4,4]{1,0} reduce(f32[2,4,4]{2,1,0} %convert.271, f32[] %convert.273), dimensions={0}, to_apply=%gradients_MatMul_1_grad_Sum_1-reduction.38, sharding={maximal device=1}, metadata={op_type="Sum" op_name="gradients/MatMul_1_grad/Sum_1"}
  %convert.275 = f32[4,4]{1,0} convert(f32[4,4]{1,0} %reduce.274), sharding={maximal device=1}, metadata={op_type="Sum" op_name="gradients/MatMul_1_grad/Sum_1"}
  %reshape.278 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %convert.275), sharding={maximal device=1}, metadata={op_type="Reshape" op_name="gradients/MatMul_1_grad/Reshape_1"}
  %broadcast.296 = f32[4,4]{1,0} broadcast(f32[] %divide.136), dimensions={}, metadata={op_type="Mul" op_name="mul_1"}
  %multiply.297 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %reshape.278, f32[4,4]{1,0} %broadcast.296), metadata={op_type="Mul" op_name="mul_1"}
  %custom-call.298 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.146, f32[4,4]{1,0} %multiply.297, f32[] %divide.133), custom_call_target="GradientAccumulatorAddWithScale", metadata={op_type="GradientAccumulatorAddWithScale" op_name="GradientAccumulatorAddWithScale_1"}, backend_config="null"
  %custom-call.299 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.298), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink_1"}, backend_config="null"
  %custom-call.155 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %get-tuple-element.102), custom_call_target="GradientAccumulatorCreate", metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate_2"}, backend_config="null"
  %dot.244 = f32[2,4,4]{2,1,0} dot(f32[2,4,4]{2,1,0} %get-tuple-element.152, f32[2,4,4]{2,1,0} %get-tuple-element.242), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_2_grad/MatMul_1"}
  %transpose.245 = f32[2,4,4]{2,1,0} transpose(f32[2,4,4]{2,1,0} %dot.244), dimensions={0,1,2}, sharding={maximal device=2}, metadata={op_type="BatchMatMulV2" op_name="gradients/MatMul_2_grad/MatMul_1"}
  %convert.246 = f32[2,4,4]{2,1,0} convert(f32[2,4,4]{2,1,0} %transpose.245), sharding={maximal device=2}, metadata={op_type="Sum" op_name="gradients/MatMul_2_grad/Sum_1"}
  %constant.247 = f32[] constant(0), sharding={maximal device=2}, metadata={op_type="Sum" op_name="gradients/MatMul_2_grad/Sum_1"}
  %convert.248 = f32[] convert(f32[] %constant.247), sharding={maximal device=2}, metadata={op_type="Sum" op_name="gradients/MatMul_2_grad/Sum_1"}
  %reduce.249 = f32[4,4]{1,0} reduce(f32[2,4,4]{2,1,0} %convert.246, f32[] %convert.248), dimensions={0}, to_apply=%gradients_MatMul_2_grad_Sum_1-reduction.34, sharding={maximal device=2}, metadata={op_type="Sum" op_name="gradients/MatMul_2_grad/Sum_1"}
  %convert.250 = f32[4,4]{1,0} convert(f32[4,4]{1,0} %reduce.249), sharding={maximal device=2}, metadata={op_type="Sum" op_name="gradients/MatMul_2_grad/Sum_1"}
  %reshape.253 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %convert.250), sharding={maximal device=2}, metadata={op_type="Reshape" op_name="gradients/MatMul_2_grad/Reshape_1"}
  %broadcast.254 = f32[4,4]{1,0} broadcast(f32[] %divide.136), dimensions={}, metadata={op_type="Mul" op_name="mul_2"}
  %multiply.255 = f32[4,4]{1,0} multiply(f32[4,4]{1,0} %reshape.253, f32[4,4]{1,0} %broadcast.254), metadata={op_type="Mul" op_name="mul_2"}
  %custom-call.256 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.155, f32[4,4]{1,0} %multiply.255, f32[] %divide.133), custom_call_target="GradientAccumulatorAddWithScale", metadata={op_type="GradientAccumulatorAddWithScale" op_name="GradientAccumulatorAddWithScale_2"}, backend_config="null"
  %custom-call.257 = f32[4,4]{1,0} custom-call(f32[4,4]{1,0} %custom-call.256), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink_2"}, backend_config="null"
  %get-tuple-element.103 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=6
  %get-tuple-element.104 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=7
  %get-tuple-element.105 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.96), index=8
  %call.300 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) call(f32[4,4]{1,0} %custom-call.295, f32[4,4]{1,0} %custom-call.299, f32[4,4]{1,0} %custom-call.257, f32[4,4]{1,0} %get-tuple-element.100, f32[4,4]{1,0} %get-tuple-element.103, /*index=5*/f32[4,4]{1,0} %get-tuple-element.101, f32[4,4]{1,0} %get-tuple-element.104, f32[4,4]{1,0} %get-tuple-element.102, f32[4,4]{1,0} %get-tuple-element.105), to_apply=%Momentum_WU_func_509_rearrange_0__.46, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate",OFFLOAD_WEIGHT_UPDATE_VARIABLES="THREESTATE_UNDEFINED",PARTITION_OFFLOADED_WEIGHT_UPDATE_VARIABLES="THREESTATE_OFF"}, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %get-tuple-element.301 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=0, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.308 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.301), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.309 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.308), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.303 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=2, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.310 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.303), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.311 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.310), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.305 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=4, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.312 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.305), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.313 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.312), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.302 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=1, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.314 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.302), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.315 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.314), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.304 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=3, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.316 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.304), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.317 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.316), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.306 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) %call.300), index=5, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}
  %tuple.318 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.306), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.319 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.318), index=0, metadata={op_name="XLA_Retvals"}
  ROOT %tuple.320 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(s32[] %add.114, s32[] %constant.307, s32[] %get-tuple-element.99, f32[4,4]{1,0} %get-tuple-element.309, f32[4,4]{1,0} %get-tuple-element.311, /*index=5*/f32[4,4]{1,0} %get-tuple-element.313, f32[4,4]{1,0} %get-tuple-element.315, f32[4,4]{1,0} %get-tuple-element.317, f32[4,4]{1,0} %get-tuple-element.319), metadata={op_name="XLA_Retvals"}
}

%_functionalize_cond_0_const_0__.321 (arg_tuple.322: (s32[], s32[], s32[], f32[4,4], f32[4,4], /*index=5*/f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> (pred[]) {
  %arg_tuple.322 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0), metadata={op_name="XLA_Args"}
  %get-tuple-element.324 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=1
  %get-tuple-element.325 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=2
  %get-tuple-element.326 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=3
  %get-tuple-element.327 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=4
  %get-tuple-element.328 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=5
  %get-tuple-element.329 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=6
  %get-tuple-element.330 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=7
  %get-tuple-element.331 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=8
  %constant.332 = s32[] constant(24), metadata={op_type="Less" op_name="Less"}
  %constant.335 = pred[] constant(true), metadata={op_type="LogicalAnd" op_name="LogicalAnd"}
  %get-tuple-element.323 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.322), index=0
  %constant.333 = s32[] constant(24), metadata={op_type="Less" op_name="Less"}
  %compare.334 = pred[] compare(s32[] %get-tuple-element.323, s32[] %constant.333), direction=LT, metadata={op_type="Less" op_name="Less"}
  %constant.336 = pred[] constant(true), metadata={op_type="LogicalAnd" op_name="LogicalAnd"}
  %and.337 = pred[] and(pred[] %compare.334, pred[] %constant.336), metadata={op_type="LogicalAnd" op_name="LogicalAnd"}
  ROOT %tuple.338 = (pred[]) tuple(pred[] %and.337), metadata={op_name="XLA_Retvals"}
}

%cond_wrapper.339 (inputs.340: (s32[], s32[], s32[], f32[4,4], f32[4,4], /*index=5*/f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> pred[] {
  %inputs.340 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %call.341 = (pred[]) call((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %inputs.340), to_apply=%_functionalize_cond_0_const_0__.321
  ROOT %get-tuple-element.342 = pred[] get-tuple-element((pred[]) %call.341), index=0
}

ENTRY %cluster_1183973981639933248_f15n_0__.382 (arg0.1: f32[4,4], arg1.2: f32[4,4], arg2.3: f32[4,4], arg3.4: f32[4,4], arg4.5: f32[4,4], arg5.6: f32[4,4]) -> (f32[4,4], f32[4,4], f32[4,4], f32[4,4], f32[4,4], /*index=5*/f32[4,4]) {
  %constant.7 = s32[] constant(0), metadata={op_type="While" op_name="LoopCond"}
  %constant.8 = s32[] constant(0), metadata={op_type="While" op_name="LoopCond"}
  %constant.9 = s32[] constant(24), metadata={op_type="While" op_name="LoopCond"}
  %constant.10 = s32[] constant(0), metadata={op_type="While" op_name="LoopCond"}
  %constant.11 = s32[] constant(0), metadata={op_type="While" op_name="LoopCond"}
  %constant.12 = s32[] constant(24), metadata={op_type="While" op_name="LoopCond"}
  %arg3.4 = f32[4,4]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/s1/w"}
  %arg4.5 = f32[4,4]{1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/s2/w"}
  %arg5.6 = f32[4,4]{1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/s3/w"}
  %arg0.1 = f32[4,4]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s1/w/Momentum"}
  %arg1.2 = f32[4,4]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s2/w/Momentum"}
  %arg2.3 = f32[4,4]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args/ipu_sharded/ipu_sharded/s3/w/Momentum"}
  %tuple.13 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(s32[] %constant.10, s32[] %constant.11, s32[] %constant.12, f32[4,4]{1,0} %arg3.4, f32[4,4]{1,0} %arg4.5, /*index=5*/f32[4,4]{1,0} %arg5.6, f32[4,4]{1,0} %arg0.1, f32[4,4]{1,0} %arg1.2, f32[4,4]{1,0} %arg2.3), metadata={op_type="While" op_name="LoopCond"}
  %while.343 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.13), condition=%cond_wrapper.339, body=%_functionalize_body_0_const_0__.95, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.344 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=0, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.345 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=1, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.346 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=2, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.347 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=3, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.348 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=4, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.349 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=5, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.350 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=6, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.351 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=7, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.352 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.343), index=8, metadata={op_type="While" op_name="LoopCond"}
  %tuple.353 = (s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(s32[] %get-tuple-element.344, s32[] %get-tuple-element.345, s32[] %get-tuple-element.346, f32[4,4]{1,0} %get-tuple-element.347, f32[4,4]{1,0} %get-tuple-element.348, /*index=5*/f32[4,4]{1,0} %get-tuple-element.349, f32[4,4]{1,0} %get-tuple-element.350, f32[4,4]{1,0} %get-tuple-element.351, f32[4,4]{1,0} %get-tuple-element.352), metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.354 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=0, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.355 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=1, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.356 = s32[] get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=2, metadata={op_type="While" op_name="LoopCond"}
  %get-tuple-element.360 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=6, metadata={op_type="While" op_name="LoopCond"}
  %reshape.363 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.360), metadata={op_name="XLA_Retvals"}
  %tuple.364 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.363), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.365 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.364), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.361 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=7, metadata={op_type="While" op_name="LoopCond"}
  %reshape.366 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.361), metadata={op_name="XLA_Retvals"}
  %tuple.367 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.366), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.368 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.367), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.362 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=8, metadata={op_type="While" op_name="LoopCond"}
  %reshape.369 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.362), metadata={op_name="XLA_Retvals"}
  %tuple.370 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.369), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.371 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.370), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.357 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=3, metadata={op_type="While" op_name="LoopCond"}
  %reshape.372 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.357), metadata={op_name="XLA_Retvals"}
  %tuple.373 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.372), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.374 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.373), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.358 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=4, metadata={op_type="While" op_name="LoopCond"}
  %reshape.375 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.358), metadata={op_name="XLA_Retvals"}
  %tuple.376 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.375), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.377 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.376), index=0, metadata={op_name="XLA_Retvals"}
  %get-tuple-element.359 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], s32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.353), index=5, metadata={op_type="While" op_name="LoopCond"}
  %reshape.378 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %get-tuple-element.359), metadata={op_name="XLA_Retvals"}
  %tuple.379 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %reshape.378), metadata={op_name="XLA_Retvals"}
  %get-tuple-element.380 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %tuple.379), index=0, metadata={op_name="XLA_Retvals"}
  ROOT %tuple.381 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.365, f32[4,4]{1,0} %get-tuple-element.368, f32[4,4]{1,0} %get-tuple-element.371, f32[4,4]{1,0} %get-tuple-element.374, f32[4,4]{1,0} %get-tuple-element.377, /*index=5*/f32[4,4]{1,0} %get-tuple-element.380), metadata={op_name="XLA_Retvals"}
}
)";

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_PIPELINE_GROUPED_RECOMPUTATION_HLO_H_
