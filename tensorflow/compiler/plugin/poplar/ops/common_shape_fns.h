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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_OPS_COMMON_SHAPE_FNS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_OPS_COMMON_SHAPE_FNS_H_

#include <array>

#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace shape_inference {
namespace poplarplugin {
// Expects the context to have `output_shapes` attribute which is used to assign
// the output shapes.
Status ShapeFromOutputShapeAttribute(InferenceContext* c);

// Like UnchangedShape, but handles tuples.
Status UnchangedTupleShape(InferenceContext* c);
}  // namespace poplarplugin
}  // namespace shape_inference
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_OPS_COMMON_SHAPE_FNS_H_
