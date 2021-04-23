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

REGISTER_OP("NormaliseImage")
    .Input("image: Tin")
    .Input("channel_offsets: dtype")
    .Input("channel_scales: dtype")
    .Output("out: dtype")
    .Attr("dtype: {float16, float32}")
    .Attr("Tin: {float16, float32, uint8}")
    .Attr("scale: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Out image has 4 channels.
      auto image_shape = c->input(0);
      shape_inference::ShapeHandle out_shape;
      c->ReplaceDim(image_shape, -1, c->MakeDim(4), &out_shape);
      c->set_output(0, out_shape);
      return Status::OK();
    });

}  // namespace tensorflow
