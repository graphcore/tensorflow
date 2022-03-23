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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("DatasetExtractor")
    .Input("input_dataset: variant")
    .Attr("print_stats: bool")
    .Attr("num_elements: int")
    .Attr("filename: string")
    .Attr("feed_name: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("VariablesExporter")
    .Input("variables: var_types")
    .Attr("print_stats: bool")
    .Attr("filename: string")
    .Attr("names: list(string)")
    .Attr("metadata_file: string")
    .Attr("var_types: list(type) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("ResourceToHandleName")
    .Input("variables: N * resource")
    .Output("outputs: N * string")
    .Attr("N: int >= 0")
    .SetIsStateful();

REGISTER_OP("VariablesImporter")
    .Output("outputs: output_types")
    .Attr("print_stats: bool")
    .Attr("is_input: bool")
    .Attr("filenames: list(string)")
    .Attr("names: list(string)")
    .Attr("strict: bool")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute);

}  // namespace tensorflow
