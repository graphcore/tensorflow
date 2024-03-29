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

REGISTER_OP("IpuConvertFromF8")
    .Input("input: uint8")
    .Input("input_metadata: uint8")
    .Output("output: float16")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Converts input from any type to F8 representation (U8 tensor + U8 metadata).
)doc");

REGISTER_OP("IpuConvertToF8")
    .Input("input: float16")
    .Input("input_metadata: uint8")
    .Output("output: uint8")
    .Output("output_metadata: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Converts from F8 representation (U8 tensor + U8 metadata) to F8.
)doc");

}  // namespace tensorflow
