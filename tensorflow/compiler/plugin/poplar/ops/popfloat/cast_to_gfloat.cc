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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
REGISTER_OP("CalcGfloatParams")
    .Attr("T: {int32}")
    .Output("output: T")
    .Input("shape: T")
    .Attr("mantissa: int = 10")
    .Attr("exponent: int = 5")
    .Attr("bias: int = 15")
    .Attr("en_denorm: bool = true")
    .Attr("en_inf: bool = true")
    .Attr("calc_type: {float16, float32}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Set output shape
      shape_inference::ShapeHandle gf_params_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &gf_params_shape));
      c->set_output(0, gf_params_shape);
      return Status::OK();
    })
    .Doc(R"doc(CalcGfloatParams Implementation.)doc");

REGISTER_OP("CastNativeToGfloat")
    .Attr("in_type: {float16, float32}")
    .Attr("out_type: {float16, float32, int8, int16}")
    .Input("input: in_type")
    .Input("params: int32")
    .Output("output: out_type")
    .Attr("en_nanoo: bool = true")
    .Attr("round_mode: string = 'RN'")
    .Attr("sr_density: string = 'Invalid'")
    .Attr("sr_bits: int = 24")
    .Attr("sr_norm_offset: float = 0.0")
    .Attr("sr_norm_scale: float = 1.0")
    .Attr("sr_norm_min: float = -0.5")
    .Attr("sr_norm_max: float = 0.5")
    .Attr("sr_prob: float = 1.0")
    .Attr("gfloat_format: string = 'Invalid'")
    .Attr("calc_type: {float16, float32}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(CastNativeToGfloat implementation.)doc");

REGISTER_OP("CastGfloatToNative")
    .Attr("in_type: {int16, int8}")
    .Attr("out_type: {float16, float32}")
    .Input("input: in_type")
    .Input("params: int32")
    .Output("output: out_type")
    .Attr("gfloat_format: string = 'Invalid'")
    .Attr("calc_type: {float16, float32}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(CastGfloatToNative implementation.)doc");

}  // namespace tensorflow
