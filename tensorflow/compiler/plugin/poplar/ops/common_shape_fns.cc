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

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace shape_inference {
namespace poplarplugin {
Status ShapeFromOutputShapeAttribute(InferenceContext* c) {
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

Status UnchangedTupleShape(InferenceContext* c) {
  for (int64_t i = 0; i != c->num_outputs(); ++i) {
    c->set_output(i, c->input(i));
    auto* handle_data = c->input_handle_shapes_and_types(i);
    if (handle_data != nullptr) {
      c->set_output_handle_shapes_and_types(i, *handle_data);
    }
  }
  return Status::OK();
}
}  // namespace poplarplugin
}  // namespace shape_inference
}  // namespace tensorflow
