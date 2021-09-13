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

#include "tensorflow/compiler/plugin/poplar/ops/common_shape_fns.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

REGISTER_OP("Pipeline")
    .Input("inputs: Tin")
    .Input("gradient_accumulation_count: int32")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("batch_serialization_iterations: int >= 1")
    .Attr("repeat_count: int >= 1")
    .Attr("schedule: int")
    .Attr("recomputation_mode: string")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("pipeline_poplar_config: string")
    .Attr("offload_activations: string")
    .Attr("offload_gradient_accumulation_buffers: string")
    .Attr("replicated_weight_sharding: string")
    .Attr("offload_weights: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
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
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
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
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes the gradient pipeline
  stage on the IPU.
)doc");

REGISTER_OP("ResourceUpdate")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("to_apply: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("offload_weight_update_variables: string")
    .Attr("replicated_optimizer_state_sharding: string")
    .Attr("num_batches_to_accumulate: int >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
inputs: A list of input tensors.
output: A list of tensors returned by computing to_apply on a device.
to_apply: A function which takes 'inputs' and computes the gradient pipeline
  stage on the IPU.
)doc");

REGISTER_OP("RecomputationCheckpoint")
    .Input("inputs: dtype")
    .Output("output: dtype")
    .Attr("dtype: type")
    .SetShapeFn(shape_inference::UnchangedShape);
}  // namespace tensorflow
