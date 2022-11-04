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

#include <numeric>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("IpuF8Matmul")
    .Input("lhs: lhs_type")
    .Input("lhs_meta: uint8")
    .Input("rhs: rhs_type")
    .Input("rhs_meta: uint8")
    .Output("output: float16")
    .Output("output_meta: uint8")
    .Attr("lhs_type: {uint8, float16, float32}")
    .Attr("rhs_type: {uint8, float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto lhs_shape = c->input(0);
      auto rhs_shape = c->input(2);
      auto groups = c->Dim(lhs_shape, 0);
      auto M = c->Dim(lhs_shape, 1);
      auto N = c->Dim(rhs_shape, 2);
      c->set_output(0, c->MakeShape({groups, M, N}));
      c->set_output(1, {});
      return Status::OK();
    })
    .Doc(R"doc(Matmul which handles an input being fp8)doc");

REGISTER_OP("IpuF8Conv2D")
    .Input("inputs: T")
    .Input("filters: U")
    .Input("inputs_meta: uint8")
    .Input("filters_meta: uint8")
    .Output("output: float16")
    .Output("output_meta: uint8")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("T: {uint8, float16, float32}")
    .Attr("U: {uint8, float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Ideally, we'd have the meta tensors adjacent to their respective
      // input tensors. However, for this to work we'd have to rewrite (or
      // at least copy/paste/change) Conv2DShapeWithExplicitPadding, which
      // is rather substantial and depends on input positioning.
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShapeWithExplicitPadding(c));
      c->set_output(1, {});
      return Status::OK();
    })
    .Doc(R"doc(Conv2D which handles an input being fp8)doc");

REGISTER_OP("IpuF8Conv3D")
    .Input("inputs: T")
    .Input("filters: U")
    .Input("inputs_meta: uint8")
    .Input("filters_meta: uint8")
    .Output("output: float16")
    .Output("output_meta: uint8")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .Attr("T: {uint8, float16, float32}")
    .Attr("U: {uint8, float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Ideally, we'd have the meta tensors adjacent to their respective
      // input tensors. However, for this to work we'd have to rewrite (or
      // at least copy/paste/change) Conv3DShape, which is rather
      // substantial and depends on input positioning.
      TF_RETURN_IF_ERROR(shape_inference::Conv3DShape(c));
      c->set_output(1, {});
      return Status::OK();
    })
    .Doc(R"doc(Conv3D which handles an input being fp8)doc");

}  // namespace tensorflow
