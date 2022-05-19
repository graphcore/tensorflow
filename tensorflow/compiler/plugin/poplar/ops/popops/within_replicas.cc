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

#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

REGISTER_OP("IpuAllGatherWithinReplica")
    .Input("inputs: T")
    .Output("output: T")
    .Attr("T: list(type) >= 2")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::DimensionHandle inner = c->MakeDim(0ll);

      // Setup a tuple output of num_inputs length with each element of shape
      // concat(inputs). Each input is expected to be a 1d tensor.
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        ::tensorflow::shape_inference::ShapeHandle temp_input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &temp_input));
        TF_RETURN_IF_ERROR(c->Add(c->Dim(temp_input, 0), inner, &inner));
      }

      const auto output_shape = c->MakeShape({inner});
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        c->set_output(i, output_shape);
      }

      return Status::OK();
    });

REGISTER_OP("IpuReduceScatterWithinReplica")
    .Input("inputs: input_types")
    .Output("output: input_types")
    .Attr("input_types: list(type) >= 2")
    .Attr("collective_op: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));

      shape_inference::DimensionHandle d = c->Dim(input_shape, 0);
      const auto input_length = c->Value(d);

      const auto shard_count = c->num_inputs();
      const int64_t output_length =
          MathUtil::CeilOfRatio<int64_t>(input_length, shard_count);

      for (size_t i = 0; i < shard_count; ++i) {
        ::tensorflow::shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &input_shape));

        c->set_output(i, c->MakeShape({output_length}));
      }

      return Status::OK();
    });

REGISTER_OP("IpuAllReduceWithinReplica")
    .Input("inputs: T")
    .Output("output: T")
    .Attr("T: list(type) >= 2")
    .Attr("collective_op: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Copy input shape -> output shape tuple with rank checks.
      const auto shard_count = c->num_inputs();
      for (size_t i = 0; i < shard_count; ++i) {
        ::tensorflow::shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &input_shape));

        shape_inference::DimensionHandle d = c->Dim(input_shape, 0);
        const auto output_length = c->Value(d);
        c->set_output(i, c->MakeShape({output_length}));
      }

      return Status::OK();
    });

}  // namespace tensorflow
