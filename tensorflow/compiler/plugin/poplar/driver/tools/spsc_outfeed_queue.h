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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_OUTFEED_QUEUE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_OUTFEED_QUEUE_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_queue.h"

#include "tensorflow/core/platform/mem.h"

namespace xla {
namespace poplarplugin {

/**
 * SPSCQueue specialization for the outfeeds in order to reduce the time spent
 * where the device is blocked due to an outfeed callback.
 *
 * This queue pre-allocates all the memory so that the the outfeed callback
 * only needs to perform a memory copy before returning.
 *
 * We also keep track of the threads which are blocked and currently waiting to
 * write.
 *
 * \tparam Capacity The capacity of the queue.
 */
template <std::size_t Capacity>
class SPSCOutfeedQueue : SPSCQueue<void*, Capacity> {
 public:
  /**
   * Construct the SPSCOutfeedQueue.
   *
   * \param element_size The size of each buffer in the queue.
   */
  explicit SPSCOutfeedQueue(std::size_t element_size)
      : SPSCQueue<void*, Capacity>(nullptr, [](void*& ptr) {
          if (ptr) {
            tensorflow::port::AlignedFree(ptr);
            ptr = nullptr;
          }
        }) {
    for (std::size_t i = 0; i != Capacity; ++i) {
      buffer_[i] = tensorflow::port::AlignedMalloc(element_size, 64);
    }
  }

  /**
   * Access the back elemenet of the queue, but it will block until the queue is
   * not full. This is only safe to call on the same thread which pushes to the
   * queue.
   *
   * \param item The element to push.
   */
  inline void*& BlockBack() {
    while (SPSCQueue<void*, Capacity>::IsFull()) {
      _mm_pause();
    }

    return buffer_[push_count_ % Capacity];
  }

  /**
   * Called when writing to the back of the queue is finished.
   *
   */
  inline void FinishedBack() {
    SPSCQueue<void*, Capacity>::AdvanceWritePosition();
  }

  /**
   * Access the front elemenet of the queue, except it will block until the
   * queue is not full. This is only safe to call on the same thread which pops
   * from the queue.
   *
   * \param item The element to push.
   */
  inline void*& BlockFront() {
    while (SPSCQueue<void*, Capacity>::IsEmpty()) {
      _mm_pause();
    }

    return buffer_[pop_count_ % Capacity];
  }

  inline void* Front() { return buffer_[pop_count_ % Capacity]; }

  /**
   * Called when reading from the front of the queue is finished.
   *
   */
  inline void FinishedFront() {
    SPSCQueue<void*, Capacity>::AdvanceReadPosition();
  }

  /**
   * Test whether the queue has any items waiting to be read, including any
   * threads which are currently blocked as the queue is full/are currently
   * writing to the data.
   *
   * \return True if the queue has items waiting, otherwise false.
   */
  inline bool HasItemsWaiting() const {
    return !SPSCQueue<void*, Capacity>::IsEmpty();
  }

  using SPSCQueue<void*, Capacity>::buffer_;
  using SPSCQueue<void*, Capacity>::push_count_;
  using SPSCQueue<void*, Capacity>::pop_count_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_OUTFEED_QUEUE_H_
