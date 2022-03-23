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

#include "tensorflow/compiler/plugin/poplar/driver/tools/popef_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace poplarplugin {

Status PopEFExceptionToTensorflowStatus(const std::string& origin,
                                        const std::exception& e) {
  const std::string prefix = "[PopEF]" + origin + ": %s";
  return InternalError(prefix.c_str(), e.what());
}

void ToPopEFShape(const xla::Shape& shape, std::vector<int64_t>& popef_shape) {
  for (size_t i = 0; i < shape.dimensions_size(); i++) {
    popef_shape.push_back(shape.dimensions(i));
  }
}

StatusOr<popef::DataType> ToPopEFDataType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PRED:
      return popef::DataType::BOOL;
    case xla::F32:
      return popef::DataType::F32;
    case xla::F16:
      return popef::DataType::F16;
    case xla::S32:
      return popef::DataType::S32;
    case xla::U32:
      return popef::DataType::U32;
    case xla::S8:
      return popef::DataType::S8;
    case xla::U8:
      return popef::DataType::U8;
    case xla::S16:
      return popef::DataType::S16;
    case xla::U16:
      return popef::DataType::U16;
    case xla::F64:
      return popef::DataType::F64;
    case xla::S64:
      return popef::DataType::S64;
    case xla::U64:
      return popef::DataType::U64;
  }
  return tensorflow::errors::Internal(
      "[PopEF][ToMetadata]: ", "Unsupported PrimitiveType ",
      std::to_string(type));
}

}  // namespace poplarplugin
}  // namespace xla
