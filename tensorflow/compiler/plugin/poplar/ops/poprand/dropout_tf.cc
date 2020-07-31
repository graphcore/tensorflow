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
    .Input("seed: int32")
    .Output("output: dtype")
    .Output("seed_used: int32")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("rate: float")
    .Attr("scale: float")
    .Attr("is_using_user_seed: bool")
    .Attr("modify_seed: bool")
    .Attr("noise_shape: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      c->set_output(0, in_shape);
      c->set_output(1, c->MakeShape({2}));
      return Status::OK();
    });
}  // namespace tensorflow
