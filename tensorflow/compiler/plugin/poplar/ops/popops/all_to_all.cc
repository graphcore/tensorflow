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

REGISTER_OP("IpuAllGather")
    .Input("inputs: dtype")
    .Output("output: dtype")
    .Attr("dtype: list({float16, float32, int32}) >= 1")
    .Attr("replication_factor: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;

      int32 replication_factor = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("replication_factor", &replication_factor));

      for (int64_t i = 0; i != c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->Concatenate(c->MakeShape({replication_factor}),
                                          c->input(i), &output));

        c->set_output(i, output);
      }
      return Status::OK();
    });

REGISTER_OP("IpuAllToAll")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("split_dimension: int")
    .Attr("concat_dimension: int")
    .Attr("number_of_replicas: int")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
