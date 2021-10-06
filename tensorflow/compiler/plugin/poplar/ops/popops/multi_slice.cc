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

REGISTER_OP("IpuMultiSlice")
    .Input("input: dtype")
    .Input("indices: int32")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("indices_are_sorted: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // outshape = list(ids.shape) + [N]
      shape_inference::ShapeHandle output, N, in_shape, indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &in_shape));
      TF_RETURN_IF_ERROR(c->Subshape(in_shape, 1, 2, &N));
      TF_RETURN_IF_ERROR(c->Concatenate(c->input(1), N, &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation multiSlice for embeddings.
)doc");

REGISTER_OP("IpuMultiUpdate")
    .Input("input: dtype")
    .Input("indices: int32")
    .Input("updates: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("indices_are_sorted: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IpuMultiUpdateAdd")
    .Input("input: dtype")
    .Input("indices: int32")
    .Input("updates: dtype")
    .Input("scale: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("indices_are_sorted: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
