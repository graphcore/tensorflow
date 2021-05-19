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
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

REGISTER_OP("IpuDropout")
    .Input("input: dtype")
    .Output("output: dtype")
    .Output("seed: int32")
    .Output("reference: variant")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      c->set_output(0, in_shape);
      c->set_output(1, c->MakeShape({2}));
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IpuDropoutNoRef")
    .Input("input: dtype")
    .Output("output: dtype")
    .Output("seed: int32")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      c->set_output(0, in_shape);
      c->set_output(1, c->MakeShape({2}));
      return Status::OK();
    });

REGISTER_OP("IpuDropoutWithSeed")
    .Input("input: dtype")
    .Input("seed: int32")
    .Output("output: dtype")
    .Output("output_seed: int32")
    .Output("reference: variant")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto seed_shape = c->input(1);
      c->set_output(0, in_shape);
      c->set_output(1, seed_shape);
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("IpuDropoutWithSeedNoRef")
    .Input("input: dtype")
    .Input("seed: int32")
    .Output("output: dtype")
    .Output("output_seed: int32")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      c->set_output(0, in_shape);
      c->set_output(1, c->MakeShape({2}));
      return Status::OK();
    });

REGISTER_OP("IpuDropoutWithSeedAndReference")
    .Input("input: dtype")
    .Input("seed: int32")
    .Input("reference: variant")
    .Output("output: dtype")
    .Output("output_seed: int32")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto seed_shape = c->input(1);
      c->set_output(0, in_shape);
      c->set_output(1, seed_shape);
      return Status::OK();
    });
}  // namespace tensorflow
