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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("IpuStatefulGradientAccumulate")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("num_mini_batches: int")
    .Attr("verify_usage: bool = true")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IpuStatefulGradientAccumulateWithMomentum")
    .Input("accum: resource")
    .Input("grad: dtype")
    .Input("momentum: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("num_mini_batches: int")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("GradientAccumulatorCreateFromShape")
    .Output("output: output_type")
    .Attr("output_type: {float16, float32}")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("GradientAccumulatorCreate")
    .Input("variable: input_type")
    .Output("output: output_type")
    .Attr("input_type: {float16, float32}")
    .Attr("output_type: {float16, float32}")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("GradientAccumulatorAdd")
    .Input("accumulator: accumulator_type")
    .Input("gradients: gradient_type")
    .Output("accumulated: accumulator_type")
    .Attr("accumulator_type: {float16, float32}")
    .Attr("gradient_type: {float16, float32}")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("GradientAccumulatorSink")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32}")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("GradientAccumulationCount")
    .Input("value: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
