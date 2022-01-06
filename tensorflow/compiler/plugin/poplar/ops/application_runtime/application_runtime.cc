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

namespace tensorflow {

REGISTER_OP("ApplicationRuntime")
    .Input("inputs: input_types")
    .Output("anchor: int32")
    .Attr("input_types: list(type) >= 0")
    .Attr("filename: string")
    .Attr("engine_name: string")
    .Attr("timeout_us: int = 5000");

REGISTER_OP("ApplicationCall")
    .Input("infeeds: infeed_types")
    .Input("anchor: int32")
    .Attr("infeed_types: list(type) >= 0")
    .Output("outfeeds: outfeed_types")
    .Attr("outfeed_types: list(type) >= 0")
    .Attr("engine_name: string");

}  // namespace tensorflow
