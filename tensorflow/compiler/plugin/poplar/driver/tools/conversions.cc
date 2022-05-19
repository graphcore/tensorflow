/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/conversions.h"

namespace xla {
namespace poplarplugin {

std::vector<char> ConvInt64ToInt32(const void* src, int64_t ssize,
                                   int64_t dsize) {
  int64_t count = ssize / sizeof(int64_t);
  if (count == 0) {
    count = dsize / sizeof(int32);
  }
  std::vector<char> result(count * sizeof(int32));
  const int64_t* src64 = reinterpret_cast<const int64_t*>(src);
  int32* dst32 = reinterpret_cast<int32*>(result.data());
  for (int64_t i = 0; i < count; i++) {
    *dst32++ = *src64++;
  }
  return result;
}

std::vector<char> ConvInt32ToInt64(const void* src, int64_t ssize,
                                   int64_t dsize) {
  int64_t count = ssize / sizeof(int32);
  if (count == 0) {
    count = dsize / sizeof(int64_t);
  }
  std::vector<char> result(count * sizeof(int64_t));
  const int32* src32 = reinterpret_cast<const int32*>(src);
  int64_t* dst64 = reinterpret_cast<int64_t*>(result.data());
  for (int64_t i = 0; i < count; i++) {
    *dst64++ = *src32++;
  }
  return result;
}

ConversionFn GetInputConversionFunction(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case xla::S64:
    case xla::U64:
      return ConvInt64ToInt32;
    default:
      return nullptr;
  }
}

ConversionFn GetOutputConversionFunction(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case xla::S64:
    case xla::U64:
      return ConvInt32ToInt64;
    default:
      return nullptr;
  }
}

std::size_t HostSizeToDeviceSize(std::size_t size, PrimitiveType type) {
  CHECK_NE(type, PRIMITIVE_TYPE_INVALID);
  switch (type) {
    case xla::S64:
    case xla::U64:
      CHECK_EQ(size % 2, 0);
      return size / 2;
    default:
      return size;
  }
}

}  // namespace poplarplugin
}  // namespace xla
