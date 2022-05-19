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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONVERSIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CONVERSIONS_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"

namespace xla {
namespace poplarplugin {

// NOTE: There is an assumption in poplar_executor.cc that the device
//       representation of the data is less than or equal to the host side.
//       During the copy from device to host, the device format data is first
//       copied into the host buffer, then converted through an intermediate
//       buffer.

// NOTE: for these convertors, either the source size or the dest size will be
//       non-zero.  The convertor needs to work out the number of items to
//       transfer by considering both.
std::vector<char> ConvInt64ToInt32(const void* src, int64_t ssize,
                                   int64_t dsize);
std::vector<char> ConvInt32ToInt64(const void* src, int64_t ssize,
                                   int64_t dsize);

ConversionFn GetInputConversionFunction(const xla::Shape&);
ConversionFn GetOutputConversionFunction(const xla::Shape&);

std::size_t HostSizeToDeviceSize(std::size_t size, PrimitiveType type);

}  // namespace poplarplugin
}  // namespace xla

#endif
