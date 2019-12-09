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

namespace tensorflow {

// The _XlaRecvAtHost and _XlaSendFromHost ops are registered in
// tensorflow/core/ops/tpu_host_compute_ops.cc.

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("XlaHostCompute")
    .Input("inputs: Tinputs")
    .Output("outputs: Toutputs")
    .Attr("Tinputs: list(type) >= 0")
    .Attr("Toutputs: list(type) >= 0")
    .Attr("ancestors: list(string) >= 0")
    .Attr("key: string")
    .Attr("shape_inference_graph: func")
    .Attr("shapes: list(shape) >= 0")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (size_t i = 0; i < shapes.size(); ++i) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &out));
        c->set_output(i, out);
      }
      return Status::OK();
    });

// Note: This is currently not implemented, only registered
// in order to report a good error message when requested by
// enclosing an outside compilation scope in control flow.
REGISTER_OP("XlaSendToHost")
    .Input("input: Tinput")
    .Attr("Tinput: type")
    .Attr("key: string")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
