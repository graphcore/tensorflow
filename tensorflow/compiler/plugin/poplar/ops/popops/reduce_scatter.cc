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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

REGISTER_OP("IpuReduceScatter")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, int32}")
    .Attr("op: string")
    .Attr("replication_factor: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 replication_factor;
      TF_RETURN_IF_ERROR(c->GetAttr("replication_factor", &replication_factor));

      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));

      shape_inference::DimensionHandle d = c->Dim(input_shape, 0);
      if (c->ValueKnown(d)) {
        const int64 input_length = c->Value(d);
        const int64 output_length =
            MathUtil::CeilOfRatio<int64>(input_length, replication_factor);

        c->set_output(0, c->MakeShape({output_length}));
        return Status::OK();
      } else {
        return errors::InvalidArgument("Unknown input tensor shape.");
      }
    });

}  // namespace tensorflow
