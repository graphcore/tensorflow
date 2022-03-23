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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ALLOCATOR_H_

#include <string>

#include "tensorflow/core/framework/allocator.h"

namespace xla {
namespace poplarplugin {
class InfeedAllocator : public tensorflow::Allocator {
 public:
  // Returns a string identifying this allocator
  std::string Name() override;

  // Returns an uninitialized block of memory that is "num_bytes" bytes
  // in size.  The returned pointer is guaranteed to be aligned to a
  // multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  // Deallocate a block of memory pointer to by "ptr"
  // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
  void DeallocateRaw(void* ptr) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  //  TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ALLOCATOR_H_
