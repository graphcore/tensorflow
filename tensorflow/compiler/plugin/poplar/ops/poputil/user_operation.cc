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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("IpuUserOp")
    .Input("input: input_types")
    .Output("output: output_types")
    .Attr("input_types: list(type) >= 0")
    .Attr("output_types: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("library_path: string")
    .Attr("op_name: string")
    .Attr("gp_path: string")
    .Attr("separate_gradients: bool")
    .Attr("gradient_size: int")
    .Attr("inputs_with_gradients: list(int) >= 0")
    .Attr("partial_derivative_index: int")
    .Attr("attributes: string = ''")
    .Attr("gradient_attributes: string = ''")
    // We don't know what the user is going to do.
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
        Adds a prebuilt user operation to the tensorflow graph. 
        input: The variadic input to the user op.
        output_shapes: The shape of each tuple element output
        output_types: The type of each tuple element output
        library_path: The path to the shared library containing
            the operation.
        op_name: The name of the prefix in the shared object for the
            building/metadata functions.
        gp_path (optional): Path to the gp file if provided.
        separate_gradients: When true, generating the partial derivatives will
            create one grad op per input.  When false, one grad op will be
            created that should generate all partial derivatives.
        gradient_size: indicates number of gradients, 0 means forward op
        inputs_with_gradients: List of input indices indicating which input
            needs gradient.
        partial_derivative_index: the list of inputs for which the op should
            produce gradients w.r.t. the outputs.
        attributes: The string of user defined attributes passed to the
            operation.
        gradient_attributes: The string of user defined attributes passed to the
            gradient of the current operation.
    )doc");

REGISTER_OP("IpuUserReadWriteOp")
    .Input("input: input_types")
    .Output("output: output_types")
    .Attr("input_types: list(type) >= 0")
    .Attr("output_types: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .Attr("library_path: string")
    .Attr("op_name: string")
    .Attr("separate_gradients: bool")
    .Attr("gradient_size: int")
    .Attr("inputs_with_gradients: list(int) >= 0")
    .Attr("partial_derivative_index: int")
    .Attr("attributes: string = ''")
    .Attr("gradient_attributes: string = ''")
    // We don't know what the user is going to do.
    .SetIsStateful()
    .SetShapeFn(shape_inference::poplarplugin::ShapeFromOutputShapeAttribute)
    .Doc(R"doc(
        Adds a prebuilt user operation to the tensorflow graph. 
        input: The variadic input to the user op.
        output_shapes: The shape of each tuple element output
        output_types: The type of each tuple element output
        library_path: The path to the shared library containing
            the operation.
        op_name: The name of the prefix in the shared object for the
            building/metadata functions.
        separate_gradients: When true, generating the partial derivatives will
            create one grad op per input.  When false, one grad op will be
            created that should generate all partial derivatives.
        gradient_size: indicates number of gradients, 0 means forward op
        inputs_with_gradients: List of input indices indicating which input
            needs gradient.
        partial_derivative_index: the list of inputs for which the op should
            produce gradients w.r.t. the outputs.
        attributes: The string of user defined attributes passed to the
            operation.
        gradient_attributes: The string of user defined attributes passed to the
            gradient of the current operation.
    )doc");

}  // namespace tensorflow
