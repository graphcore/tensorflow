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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_

#include <immintrin.h>
#include <atomic>
#include <bitset>
#include <cassert>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "tensorflow/core/platform/default/logging.h"

namespace xla {
namespace poplarplugin {

namespace {
constexpr bool is_powerof2(std::size_t v) { return v && ((v & (v - 1)) == 0); }
}  // namespace

/**
 * Statically bounded single-producer/single-consumer lock-free queue.
 *
 * This can be used for buffered unidirectional communication between two
 * threads. https://en.wikipedia.org/wiki/Producer–consumer_problem
 *
 * Particular attention has be paid to keeping the Pop as cheap as possible.
 * This is achieved through the `post_apply` function, which can be used for
 * resource management purely on the "Push" thread.
 *
 * \tparam T The element type to store in the queue.
 * \tparam Capacity The capacity of the queue.
 */
template <typename T, std::size_t Capacity>
class SPSCQueue {
  static_assert(is_powerof2(Capacity),
                "SPSCQueue requires a power of 2 capacity");
  static_assert(Capacity > 8, "SPSCQueue requires a capacity greater than 8");

 public:
  /**
   * Construct the SPSCQueue.
   *
   * \param init The initial value to fill the queue.
   * \param post_apply The function that is called on each element after it has
   *        been popped.
   *
   * \note post_apply must be resistant to multiple applications on the same
   *       element.
   */
  explicit SPSCQueue(T init, std::function<void(T&)> post_apply)
      : push_count_(0), pop_count_(0), post_apply_(post_apply) {
    CHECK(post_apply);
    std::fill(buffer_.begin(), buffer_.end(), init);
  }

  virtual ~SPSCQueue() {
    for (auto& elem : buffer_) {
      post_apply_(elem);
    }
  }

  /**
   * Advance the write position of the queue.
   * This is only safe to call on the same thread which pushes to the queue.
   *
   */
  inline void AdvanceWritePosition() {
    const std::size_t push_count =
        std::atomic_load_explicit(&push_count_, std::memory_order_relaxed);
    std::atomic_store_explicit(&push_count_, push_count + 1,
                               std::memory_order_release);
  }

  /**
   * Push an element into the queue.
   *
   * \param item The element to push.
   *
   * \note This function won't block, but assumes there is space
   *       (i.e. IsFull() == false) and it does not advance the write position.
   */
  inline void Push(const T& item) {
    CHECK(!IsFull(0));

    const std::size_t push_count =
        std::atomic_load_explicit(&push_count_, std::memory_order_relaxed) %
        Capacity;

    post_apply_(buffer_[push_count]);
    buffer_[push_count] = item;
  }

  /**
   * Similar to push, except it will block until a slot is available.
   *
   * \param item The element to push.
   */
  inline void BlockPush(const T& item) {
    while (IsFull()) {
      _mm_pause();
    }

    Push(item);
  }

  /**
   * Similar to push, except it will return whether the operation was
   * successful.
   *
   * \param item The element to push.
   *
   * \return true if the element was successfully pushed, otherwise false.
   */
  inline bool TryPush(const T& item) {
    if (IsFull()) {
      return false;
    }

    Push(item);
    return true;
  }

  /**
   * Advance the read position of the queue.
   * This is only safe to call on the same thread which pops from the queue.
   *
   */
  inline void AdvanceReadPosition() {
    const std::size_t pop_count =
        std::atomic_load_explicit(&pop_count_, std::memory_order_relaxed);
    std::atomic_store_explicit(&pop_count_, pop_count + 1,
                               std::memory_order_release);
  }

  /**
   * Pop an element from the queue.
   *
   * \param item The element to pop into.
   *
   * \note This function won't block, but assumes there is at least a single
   * element (i.e. IsEmpty() == false) and it does not advance the read
   * position.
   */
  inline void Pop(T& item, std::size_t look_ahead = 0) {
    CHECK(!IsEmpty());
    CHECK(HasLookAhead(look_ahead));

    item = buffer_[(pop_count_ + look_ahead) % Capacity];
  }

  /**
   * Similar to Pop, but will block until a slot is occupied.
   *
   * \param item The element to pop into.
   */
  inline void BlockPop(T& item, std::size_t look_ahead = 0) {
    while (!HasLookAhead(look_ahead)) {
      _mm_pause();
    }

    Pop(item, look_ahead);
  }

  /**
   * Similar to Pop, except it will return whether the operation was
   * successful
   *
   * \param item The element to pop into.
   *
   * \return true if the element was successfully poped, otherwise false.
   */
  inline bool TryPop(T& item, std::size_t look_ahead = 0) {
    if (!HasLookAhead(look_ahead)) {
      return false;
    }

    Pop(item, look_ahead);
    return true;
  }

  /**
   * Test whether the queue is full.
   *
   * \return True if the queue is full, otherwise false.
   */
  inline bool IsFull(std::size_t backoff = 8) const {
    const std::size_t push_count =
        std::atomic_load_explicit(&push_count_, std::memory_order_consume);
    const std::size_t pop_count =
        std::atomic_load_explicit(&pop_count_, std::memory_order_consume);

    return __builtin_expect((push_count - pop_count) >= Capacity - backoff, 1);
  }

  /**
   * Test whether the queue is empty.
   *
   * \return True if the queue is empty, otherwise false.
   */
  inline bool IsEmpty() const {
    const std::size_t push_count =
        std::atomic_load_explicit(&push_count_, std::memory_order_consume);
    const std::size_t pop_count =
        std::atomic_load_explicit(&pop_count_, std::memory_order_consume);

    return __builtin_expect(push_count == pop_count, 0);
  }

  /**
   * Test whether the queue has at least `look_ahead` elements.
   *
   * \return True if the queue is has at least `look_ahead` elements, otherwise
   * false.
   */
  inline bool HasLookAhead(std::size_t look_ahead) const {
    CHECK(look_ahead < Capacity);

    const std::size_t push_count =
        std::atomic_load_explicit(&push_count_, std::memory_order_consume);
    const std::size_t pop_count =
        std::atomic_load_explicit(&pop_count_, std::memory_order_consume);

    return __builtin_expect((push_count - pop_count) > look_ahead, 1);
  }

 protected:
  std::array<T, Capacity> buffer_;

  alignas(64) std::atomic<std::size_t> push_count_;
  alignas(64) std::atomic<std::size_t> pop_count_;

  std::function<void(T&)> post_apply_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_
