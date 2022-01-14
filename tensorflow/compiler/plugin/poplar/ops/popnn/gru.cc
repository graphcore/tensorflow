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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

REGISTER_OP("PopnnGRULayer")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Output("output: dtype")
    .Output("output_state: dtype")
    .Output("intermediates: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("partials_dtype: {float16, float32} = DT_FLOAT")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .Attr("options_bwd: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      shape_inference::DimensionOrConstant doc_num_channels(num_channels);

      auto inputs = c->input(0);
      auto time_steps = c->Dim(inputs, 0);
      auto batch_size = c->Dim(inputs, 1);

      c->set_output(0,
                    c->MakeShape({time_steps, batch_size, doc_num_channels}));
      c->set_output(1, c->MakeShape({batch_size, doc_num_channels}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGRULayer.
)doc");

REGISTER_OP("PopnnGRULayerBackprop")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("output: dtype")
    .Input("output_state: dtype")
    .Input("intermediates: dtype")
    .Input("output_backprop: dtype")
    .Input("output_state_backprop: dtype")
    .Output("inputs_backprop: dtype")
    .Output("initial_state_backprop: dtype")
    .Output("kernel_backprop: dtype")
    .Output("biases_backprop: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("partials_dtype: {float16, float32}")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto in_state_shape = c->input(1);
      auto kernel_shape = c->input(2);
      auto biases_shape = c->input(3);
      c->set_output(0, in_shape);
      c->set_output(1, in_state_shape);
      c->set_output(2, kernel_shape);
      c->set_output(3, biases_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGRULayerBackprop.
)doc");

REGISTER_OP("PopnnDynamicGRULayer")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("seq_len: seq_dtype")
    .Output("output: dtype")
    .Output("output_state: dtype")
    .Output("intermediates: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("seq_dtype: {int32}")
    .Attr("partials_dtype: {float16, float32} = DT_FLOAT")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .Attr("options_bwd: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      shape_inference::DimensionOrConstant doc_num_channels(num_channels);

      auto inputs = c->input(0);
      auto time_steps = c->Dim(inputs, 0);
      auto batch_size = c->Dim(inputs, 1);

      c->set_output(0,
                    c->MakeShape({time_steps, batch_size, doc_num_channels}));
      c->set_output(1, c->MakeShape({batch_size, doc_num_channels}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnDynamicGRULayer.
)doc");

REGISTER_OP("PopnnDynamicGRULayerBackprop")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("seq_len: seq_dtype")
    .Input("output: dtype")
    .Input("output_state: dtype")
    .Input("intermediates: dtype")
    .Input("output_backprop: dtype")
    .Input("output_state_backprop: dtype")
    .Output("inputs_backprop: dtype")
    .Output("initial_state_backprop: dtype")
    .Output("kernel_backprop: dtype")
    .Output("biases_backprop: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("seq_dtype: {int32}")
    .Attr("partials_dtype: {float16, float32}")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto in_state_shape = c->input(1);
      auto kernel_shape = c->input(2);
      auto biases_shape = c->input(3);
      c->set_output(0, in_shape);
      c->set_output(1, in_state_shape);
      c->set_output(2, kernel_shape);
      c->set_output(3, biases_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnDynamicGRULayerBackprop.
)doc");

REGISTER_OP("PopnnAUGRULayer")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("seq_len: seq_dtype")
    .Input("att_score: dtype")
    .Output("output: dtype")
    .Output("output_state: dtype")
    .Output("intermediates: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("seq_dtype: {int32}")
    .Attr("partials_dtype: {float16, float32} = DT_FLOAT")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .Attr("options_bwd: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      shape_inference::DimensionOrConstant doc_num_channels(num_channels);

      auto inputs = c->input(0);
      auto time_steps = c->Dim(inputs, 0);
      auto batch_size = c->Dim(inputs, 1);

      c->set_output(0,
                    c->MakeShape({time_steps, batch_size, doc_num_channels}));
      c->set_output(1, c->MakeShape({batch_size, doc_num_channels}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnAUGRULayer.
)doc");

REGISTER_OP("PopnnAUGRULayerBackprop")
    .Input("inputs: dtype")
    .Input("initial_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("seq_len: seq_dtype")
    .Input("att_score: dtype")
    .Input("output: dtype")
    .Input("output_state: dtype")
    .Input("intermediates: dtype")
    .Input("output_backprop: dtype")
    .Input("output_state_backprop: dtype")
    .Output("inputs_backprop: dtype")
    .Output("initial_state_backprop: dtype")
    .Output("kernel_backprop: dtype")
    .Output("biases_backprop: dtype")
    .Output("att_backprop: dtype")
    .Attr("activation: string")
    .Attr("recurrent_activation: string")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("seq_dtype: {int32}")
    .Attr("partials_dtype: {float16, float32}")
    .Attr("reset_after: bool = false")
    // TODO(T53098): Remove `available_memory_proportion_fwd` &
    // `available_memory_proportion_bwd`.
    .Attr("available_memory_proportion_fwd: float = -1.0")
    .Attr("available_memory_proportion_bwd: float = -1.0")
    .Attr("options: string = '{}'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto in_state_shape = c->input(1);
      auto kernel_shape = c->input(2);
      auto biases_shape = c->input(3);
      auto att_shape = c->input(5);
      c->set_output(0, in_shape);
      c->set_output(1, in_state_shape);
      c->set_output(2, kernel_shape);
      c->set_output(3, biases_shape);
      c->set_output(4, att_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnAUGRULayerBackprop.
)doc");

}  // namespace tensorflow
