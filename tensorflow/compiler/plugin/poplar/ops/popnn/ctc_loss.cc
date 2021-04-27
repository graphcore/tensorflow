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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

REGISTER_OP("PopnnCTCLossWithLogits")
    .Input("data: in_dtype")
    .Input("labels: int32")
    .Input("data_lengths: int32")
    .Input("label_lengths: int32")
    .Output("loss: out_dtype")
    .Output("grad: out_dtype")
    .Attr("blank_index: int")
    .Attr("in_dtype: {float16, float32}")
    .Attr("out_dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto data_shape = c->input(0);
      auto batch_size = c->Dim(data_shape, 1);
      c->set_output(0, c->MakeShape({batch_size}));  // Loss
      c->set_output(1, data_shape);                  // Grad
      return Status::OK();
    });

REGISTER_OP("PopnnCTCLossWithLogProbs")
    .Input("data: in_dtype")
    .Input("labels: int32")
    .Input("data_lengths: int32")
    .Input("label_lengths: int32")
    .Output("loss: out_dtype")
    .Output("grad: out_dtype")
    .Attr("blank_index: int")
    .Attr("in_dtype: {float16, float32}")
    .Attr("out_dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto data_shape = c->input(0);
      auto batch_size = c->Dim(data_shape, 1);
      c->set_output(0, c->MakeShape({batch_size}));  // Loss
      c->set_output(1, data_shape);                  // Grad
      return Status::OK();
    });

REGISTER_OP("PopnnCTCBeamSearchWithLogits")
    .Input("data: in_dtype")
    .Input("data_lengths: int32")
    .Output("label_probabilities: in_dtype")
    .Output("label_lengths: int32")
    .Output("decoded_labels: int32")
    .Attr("blank_index: int")
    .Attr("beam_width: int")
    .Attr("in_dtype: {float16, float32}")
    .Attr("top_paths: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto data_shape = c->input(0);
      auto batch_size = c->Dim(data_shape, 1);
      auto max_time = c->Dim(data_shape, 0);
      int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));
      c->set_output(0, c->MakeShape({batch_size, top_paths}));
      c->set_output(1, c->MakeShape({batch_size, top_paths}));
      c->set_output(2, c->MakeShape({batch_size, top_paths, max_time}));
      return Status::OK();
    });

REGISTER_OP("PopnnCTCBeamSearchWithLogProbs")
    .Input("data: in_dtype")
    .Input("data_lengths: int32")
    .Output("label_probabilities: in_dtype")
    .Output("label_lengths: int32")
    .Output("decoded_labels: int32")
    .Attr("blank_index: int")
    .Attr("beam_width: int")
    .Attr("in_dtype: {float16, float32}")
    .Attr("top_paths: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto data_shape = c->input(0);
      auto batch_size = c->Dim(data_shape, 1);
      auto max_time = c->Dim(data_shape, 0);
      int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));
      c->set_output(0, c->MakeShape({batch_size, top_paths}));
      c->set_output(1, c->MakeShape({batch_size, top_paths}));
      c->set_output(2, c->MakeShape({batch_size, top_paths, max_time}));
      return Status::OK();
    });

}  // namespace tensorflow
