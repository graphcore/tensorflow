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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_SERIALIZED_MATMUL_HLO_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_SERIALIZED_MATMUL_HLO_H_

const char* serialized_matmul_hlo = R"(
HloModule serial_cluster_14460244887885090275__.91

%serial_Sum-reduction.35 (x.36: f32[], y.37: f32[]) -> f32[] {
  %x.36 = f32[] parameter(0)
  %y.37 = f32[] parameter(1)
  ROOT %add.38 = f32[] add(f32[] %x.36, f32[] %y.37)
}

%serial_gradients_serial_IdentityN_grad_Sum-reduction.57 (x.58: f32[], y.59: f32[]) -> f32[] {
  %x.58 = f32[] parameter(0)
  %y.59 = f32[] parameter(1)
  ROOT %add.60 = f32[] add(f32[] %x.58, f32[] %y.59)
}

%serial_gradients_serial_IdentityN_grad_Sum_1-reduction.77 (x.78: f32[], y.79: f32[]) -> f32[] {
  %x.78 = f32[] parameter(0)
  %y.79 = f32[] parameter(1)
  ROOT %add.80 = f32[] add(f32[] %x.78, f32[] %y.79)
}

ENTRY %serial_cluster_14460244887885090275__.91 (arg0.1: f32[4,4], arg1.2: f32[3,44,4]) -> (f32[4,4], f32[3,44,4], f32[]) {
  %constant.7 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_2"}
  %constant.13 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_3"}
  %constant.15 = s32[] constant(-2), metadata={op_type="ConcatV2" op_name="serial/gradients/serial/IdentityN_grad/concat"}
  %constant.17 = s32[] constant(0), metadata={op_type="Slice" op_name="serial/Slice"}
  %broadcast.18 = s32[3]{0} broadcast(s32[] %constant.17), dimensions={}, metadata={op_type="Slice" op_name="serial/Slice"}
  %constant.19 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/Slice"}
  %constant.21 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/Slice"}
  %constant.24 = s32[3]{0} constant({0, 22, 0}), metadata={op_type="Slice" op_name="serial/Slice_1"}
  %constant.25 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/Slice_1"}
  %constant.27 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/Slice_1"}
  %constant.30 = s32[] constant(-1), metadata={op_type="ConcatV2" op_name="serial/concat"}
  %constant.41 = f32[] constant(0), metadata={op_type="ZerosLike" op_name="serial/gradients/zeros_like"}
  %broadcast.42 = f32[4,4]{1,0} broadcast(f32[] %constant.41), dimensions={}, metadata={op_type="ZerosLike" op_name="serial/gradients/zeros_like"}
  %constant.43 = f32[] constant(0), metadata={op_type="ZerosLike" op_name="serial/gradients/zeros_like_1"}
  %broadcast.44 = f32[3,44,4]{2,1,0} broadcast(f32[] %constant.43), dimensions={}, metadata={op_type="ZerosLike" op_name="serial/gradients/zeros_like_1"}
  %constant.45 = s32[] constant(0), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_1"}
  %broadcast.46 = s32[3]{0} broadcast(s32[] %constant.45), dimensions={}, metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_1"}
  %constant.47 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_1"}
  %constant.49 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_1"}
  %constant.63 = s32[] constant(4), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape"}
  %broadcast.64 = s32[2]{0} broadcast(s32[] %constant.63), dimensions={}, metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape"}
  %constant.66 = s32[3]{0} constant({0, 22, 0}), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_3"}
  %constant.67 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_3"}
  %constant.69 = s32[3]{0} constant({3, 22, 4}), metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_3"}
  %constant.83 = s32[] constant(4), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_1"}
  %broadcast.84 = s32[2]{0} broadcast(s32[] %constant.83), dimensions={}, metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_1"}
  %constant.50 = f32[] constant(1), frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns"}
  %broadcast.51 = f32[3,4,22]{2,1,0} broadcast(f32[] %constant.50), dimensions={}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns"}
  %arg1.2 = f32[3,44,4]{2,1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/serial/b"}
  %slice.48 = f32[3,22,4]{2,1,0} slice(f32[3,44,4]{2,1,0} %arg1.2), slice={[0:3], [0:22], [0:4]}, metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_1"}
  %dot.52 = f32[3,4,4]{2,1,0} dot(f32[3,4,22]{2,1,0} %broadcast.51, f32[3,22,4]{2,1,0} %slice.48), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns"}
  %transpose.53 = f32[3,4,4]{2,1,0} transpose(f32[3,4,4]{2,1,0} %dot.52), dimensions={0,1,2}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns"}
  %convert.54 = f32[3,4,4]{2,1,0} convert(f32[3,4,4]{2,1,0} %transpose.53), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum"}
  %constant.55 = f32[] constant(0), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum"}
  %convert.56 = f32[] convert(f32[] %constant.55), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum"}
  %reduce.61 = f32[4,4]{1,0} reduce(f32[3,4,4]{2,1,0} %convert.54, f32[] %convert.56), dimensions={0}, to_apply=%serial_gradients_serial_IdentityN_grad_Sum-reduction.57, metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum"}
  %convert.62 = f32[4,4]{1,0} convert(f32[4,4]{1,0} %reduce.61), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum"}
  %reshape.65 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %convert.62), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape"}
  %constant.70 = f32[] constant(1), frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns_1"}
  %broadcast.71 = f32[3,4,22]{2,1,0} broadcast(f32[] %constant.70), dimensions={}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns_1"}
  %slice.68 = f32[3,22,4]{2,1,0} slice(f32[3,44,4]{2,1,0} %arg1.2), slice={[0:3], [22:44], [0:4]}, metadata={op_type="Slice" op_name="serial/gradients/serial/IdentityN_grad/Slice_3"}
  %dot.72 = f32[3,4,4]{2,1,0} dot(f32[3,4,22]{2,1,0} %broadcast.71, f32[3,22,4]{2,1,0} %slice.68), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns_1"}
  %transpose.73 = f32[3,4,4]{2,1,0} transpose(f32[3,4,4]{2,1,0} %dot.72), dimensions={0,1,2}, frontend_attributes={ML_TYPE="TRAINING_BWD"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradASplitARowsBColumns_1"}
  %convert.74 = f32[3,4,4]{2,1,0} convert(f32[3,4,4]{2,1,0} %transpose.73), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum_1"}
  %constant.75 = f32[] constant(0), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum_1"}
  %convert.76 = f32[] convert(f32[] %constant.75), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum_1"}
  %reduce.81 = f32[4,4]{1,0} reduce(f32[3,4,4]{2,1,0} %convert.74, f32[] %convert.76), dimensions={0}, to_apply=%serial_gradients_serial_IdentityN_grad_Sum_1-reduction.77, metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum_1"}
  %convert.82 = f32[4,4]{1,0} convert(f32[4,4]{1,0} %reduce.81), metadata={op_type="Sum" op_name="serial/gradients/serial/IdentityN_grad/Sum_1"}
  %reshape.85 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %convert.82), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_1"}
  %add.86 = f32[4,4]{1,0} add(f32[4,4]{1,0} %reshape.65, f32[4,4]{1,0} %reshape.85), metadata={op_type="AddV2" op_name="serial/gradients/serial/IdentityN_grad/add"}
  %reshape.87 = f32[4,4]{1,0} reshape(f32[4,4]{1,0} %add.86), metadata={op_name="XLA_Retvals"}
  %constant.3 = f32[] constant(1), frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns"}
  %broadcast.4 = f32[3,4,22]{2,1,0} broadcast(f32[] %constant.3), dimensions={}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns"}
  %arg0.1 = f32[4,4]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/serial/a"}
  %dot.5 = f32[3,22,4]{2,1,0} dot(f32[3,4,22]{2,1,0} %broadcast.4, f32[4,4]{1,0} %arg0.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns"}
  %transpose.6 = f32[3,22,4]{2,1,0} transpose(f32[3,22,4]{2,1,0} %dot.5), dimensions={0,1,2}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns"}
  %reshape.8 = f32[3,22,4]{2,1,0} reshape(f32[3,22,4]{2,1,0} %transpose.6), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_2"}
  %constant.9 = f32[] constant(1), frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns_1"}
  %broadcast.10 = f32[3,4,22]{2,1,0} broadcast(f32[] %constant.9), dimensions={}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns_1"}
  %dot.11 = f32[3,22,4]{2,1,0} dot(f32[3,4,22]{2,1,0} %broadcast.10, f32[4,4]{1,0} %arg0.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns_1"}
  %transpose.12 = f32[3,22,4]{2,1,0} transpose(f32[3,22,4]{2,1,0} %dot.11), dimensions={0,1,2}, frontend_attributes={ML_TYPE="TRAINING_WU"}, metadata={op_type="BatchMatMulV2" op_name="serial/gradients/serial/IdentityN_grad/serialized_matmulGradBSplitAColumns_1"}
  %reshape.14 = f32[3,22,4]{2,1,0} reshape(f32[3,22,4]{2,1,0} %transpose.12), metadata={op_type="Reshape" op_name="serial/gradients/serial/IdentityN_grad/Reshape_3"}
  %concatenate.16 = f32[3,44,4]{2,1,0} concatenate(f32[3,22,4]{2,1,0} %reshape.8, f32[3,22,4]{2,1,0} %reshape.14), dimensions={1}, metadata={op_type="ConcatV2" op_name="serial/gradients/serial/IdentityN_grad/concat"}
  %reshape.88 = f32[3,44,4]{2,1,0} reshape(f32[3,44,4]{2,1,0} %concatenate.16), metadata={op_name="XLA_Retvals"}
  %slice.20 = f32[3,22,4]{2,1,0} slice(f32[3,44,4]{2,1,0} %arg1.2), slice={[0:3], [0:22], [0:4]}, metadata={op_type="Slice" op_name="serial/Slice"}
  %dot.22 = f32[4,3,22]{2,1,0} dot(f32[4,4]{1,0} %arg0.1, f32[3,22,4]{2,1,0} %slice.20), lhs_contracting_dims={1}, rhs_contracting_dims={2}, metadata={op_type="BatchMatMulV2" op_name="serial/serialized_matmulSplitBRows"}
  %transpose.23 = f32[3,4,22]{2,0,1} transpose(f32[4,3,22]{2,1,0} %dot.22), dimensions={1,0,2}, metadata={op_type="BatchMatMulV2" op_name="serial/serialized_matmulSplitBRows"}
  %slice.26 = f32[3,22,4]{2,1,0} slice(f32[3,44,4]{2,1,0} %arg1.2), slice={[0:3], [22:44], [0:4]}, metadata={op_type="Slice" op_name="serial/Slice_1"}
  %dot.28 = f32[4,3,22]{2,1,0} dot(f32[4,4]{1,0} %arg0.1, f32[3,22,4]{2,1,0} %slice.26), lhs_contracting_dims={1}, rhs_contracting_dims={2}, metadata={op_type="BatchMatMulV2" op_name="serial/serialized_matmulSplitBRows_1"}
  %transpose.29 = f32[3,4,22]{2,0,1} transpose(f32[4,3,22]{2,1,0} %dot.28), dimensions={1,0,2}, metadata={op_type="BatchMatMulV2" op_name="serial/serialized_matmulSplitBRows_1"}
  %concatenate.31 = f32[3,4,44]{2,1,0} concatenate(f32[3,4,22]{2,0,1} %transpose.23, f32[3,4,22]{2,0,1} %transpose.29), dimensions={2}, metadata={op_type="ConcatV2" op_name="serial/concat"}
  %convert.32 = f32[3,4,44]{2,1,0} convert(f32[3,4,44]{2,1,0} %concatenate.31), metadata={op_type="Sum" op_name="serial/Sum"}
  %constant.33 = f32[] constant(0), metadata={op_type="Sum" op_name="serial/Sum"}
  %convert.34 = f32[] convert(f32[] %constant.33), metadata={op_type="Sum" op_name="serial/Sum"}
  %reduce.39 = f32[] reduce(f32[3,4,44]{2,1,0} %convert.32, f32[] %convert.34), dimensions={0,1,2}, to_apply=%serial_Sum-reduction.35, metadata={op_type="Sum" op_name="serial/Sum"}
  %convert.40 = f32[] convert(f32[] %reduce.39), metadata={op_type="Sum" op_name="serial/Sum"}
  %reshape.89 = f32[] reshape(f32[] %convert.40), metadata={op_name="XLA_Retvals"}
  ROOT %tuple.90 = (f32[4,4]{1,0}, f32[3,44,4]{2,1,0}, f32[]) tuple(f32[4,4]{1,0} %reshape.87, f32[3,44,4]{2,1,0} %reshape.88, f32[] %reshape.89), metadata={op_name="XLA_Retvals"}
}
)";

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_HLO_SAMPLES_SERIALIZED_MATMUL_HLO_H_
