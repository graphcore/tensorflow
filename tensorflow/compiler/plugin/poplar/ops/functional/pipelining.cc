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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {
Status SetOutputShape(shape_inference::InferenceContext* c) {
  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  if (output_shapes.size() != static_cast<size_t>(c->num_outputs())) {
    return errors::InvalidArgument(
        "`output_shapes` must be the same length as num outputs (",
        output_shapes.size(), " vs. ", c->num_outputs());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shape_inference::ShapeHandle out;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromPartialTensorShape(output_shapes[i], &out));
    c->set_output(i, out);
  }
  return Status::OK();
}
}  // namespace

REGISTER_OP("Pipeline")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("pipeline_depth: int >= 1")
    .Attr("repeat_count: int >= 1")
    .Attr("schedule: int")
    .Attr("offload_weight_update_variables: bool")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("pipeline_poplar_config: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return SetOutputShape(c);
    })
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which is made up of serially calling pipeline stages.
)doc");

REGISTER_OP("PipelineStage")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("stage_id: int >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return SetOutputShape(c);
    })
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes the pipeline stage on the
  IPU.
)doc");

REGISTER_OP("PipelineStageBackward")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("stage_id: int >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return SetOutputShape(c);
    })
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes the gradient pipeline
  stage on the IPU.
)doc");

REGISTER_OP("PipelineResourceUpdate")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return SetOutputShape(c);
    })
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes the gradient pipeline
  stage on the IPU.
)doc");
}  // namespace tensorflow
