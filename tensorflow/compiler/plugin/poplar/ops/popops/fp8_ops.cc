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
    .Input("lhs: uint8")
    .Input("lhs_meta: uint8")
    .Input("rhs: uint8")
    .Input("rhs_meta: uint8")
    .Output("output: float16")
    .Output("output_meta: uint8")
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

}  // namespace tensorflow
