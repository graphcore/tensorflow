/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

bool Is1DSliceInFirstDimension(const HloInstruction* slice) {
  auto* broadcast = slice->operand(0);
  return ShapeUtil::DeleteDimension(0, broadcast->shape()) ==
             ShapeUtil::DeleteDimension(0, slice->shape()) &&
         ShapeUtil::GetDimension(slice->shape(), 0) == 1;
}

}  // namespace poplarplugin
}  // namespace xla
