/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("IpuHistogram")
    .Input("input: dtype")
    .Input("levels: dtype")
    .Output("output: float32")
    .Attr("dtype: {float16, float32}")
    .Attr("absolute_of_input: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto levels_shape = c->input(1);
      auto num_levels = c->Dim(levels_shape, 0);

      decltype(num_levels) num_out;
      c->Add(num_levels, 1, &num_out);

      decltype(levels_shape) out_shape;
      c->ReplaceDim(levels_shape, 0, num_out, &out_shape);
      c->set_output(0, out_shape);
      return Status::OK();
    })
    .Doc(R"doc(Internal implementation of a Histogram.)doc");

REGISTER_OP("IpuHistogramUpdate")
    .Input("hist: float32")
    .Input("input: dtype")
    .Input("levels: dtype")
    .Output("output: float32")
    .Attr("dtype: {float16, float32}")
    .Attr("absolute_of_input: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Internal implementation of a Histogram update.)doc");

}  // namespace tensorflow
