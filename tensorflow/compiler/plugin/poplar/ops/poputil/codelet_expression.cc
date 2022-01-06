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

namespace tensorflow {

REGISTER_OP("CodeletExpressionOp")
    .Input("input: input_types")
    .Output("output: dtype")
    .Attr("input_types: list(type) >= 1")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("source: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->num_inputs() == 1) {
        return shape_inference::UnchangedShape(c);
      } else {
        return shape_inference::BroadcastBinaryOpShapeFn(c);
      }
    })
    .Doc(R"doc(
        Adds a user operation which compiles the elementwise codelet which is
        supplies in the 'source' property.
    )doc");

}  // namespace tensorflow
