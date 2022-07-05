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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_MEMORY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_MEMORY_H_

#include <memory>
#include <utility>

#include <popit/popit.hpp>

namespace se = stream_executor;

namespace xla {
namespace poplarplugin {

struct PopItDeallocator {
  void operator()(popitMem_t* ptr) { popitFree(ptr); }
};

using PopItBufferType = std::shared_ptr<popitMem_t>;

// We provide an extra wrapper to device memory so that we can
// reference count buffers. This is because tensorflow can deallocate
// the parent buffer while sub buffers are still live.
// To avoid this reference count parents in this wrapper.
// We are also passing back a pointer to heap allocated popitsubbuffer
// in the DeviceMemoryBase class. As this isn't RAII managed risk of
// spilling if upstream TF isn't careful. Maybe we should try upstream
// a virtual destroy method to DeviceMemoryBase to avoid this.
struct PopItSubBuffer {
  PopItBufferType parent_;
  int64_t offset_;
  int64_t size_;

  PopItSubBuffer CreateSubBuffer(int64_t offset, int64_t size) const {
    return PopItSubBuffer(parent_, offset_ + offset, size);
  }

  PopItSubBuffer(PopItBufferType parent, int64_t offset, int64_t size)
      : parent_(std::move(parent)), offset_(offset), size_(size) {}

  PopItSubBuffer(popitMem_t* mem, int64_t size)
      : PopItSubBuffer(PopItBufferType(mem, PopItDeallocator()), 0, size) {}
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_MEMORY_H_
