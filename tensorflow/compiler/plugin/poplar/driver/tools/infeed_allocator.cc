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

#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"

#include "tensorflow/core/platform/mem.h"

namespace xla {
namespace poplarplugin {
std::string InfeedAllocator::Name() { return "infeed-allocator"; }

void* InfeedAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  const size_t min_alignment = 64;
  alignment = alignment < min_alignment ? min_alignment : alignment;
  return tensorflow::port::AlignedMalloc(num_bytes, min_alignment);
}

void InfeedAllocator::DeallocateRaw(void* ptr) {
  tensorflow::port::AlignedFree(ptr);
}

}  // namespace poplarplugin
}  // namespace xla
