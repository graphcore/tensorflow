/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/ops/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

REGISTER_OP("Function")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("unique_sharding: bool")
    .Attr("keep_input_layouts: bool = True")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes on the IPU.
)doc");

REGISTER_OP("MultiConv")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("option_flags: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes on the IPU.
)doc");
}  // namespace tensorflow
