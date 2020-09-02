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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ITERATOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ITERATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_queue.h"

#include "tensorflow/compiler/xla/shape.h"

#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
class CancellationManager;
class DeviceMgr;
class FunctionLibraryDefinition;
class ProcessFunctionLibraryRuntime;
class TensorBuffer;
namespace data {
class DatasetBase;
class IteratorBase;
class IteratorContext;
}  // namespace data
namespace thread {
class ThreadPool;
}  // namespace thread
}  // namespace tensorflow

namespace xla {
namespace poplarplugin {
class InfeedAllocator;

// This is a wrapper around SPSCQueue with additional support
// for using a sentinel value to signal the end of the queue.
// Member functions are inline in the header to allow for
// inlining the wrapped calls directly at the call site.
class InfeedQueue {
 public:
  InfeedQueue();

  using T = tensorflow::TensorBuffer*;

  // Functions delegating directly to the underlying queue.
  void AdvanceReadPosition() { queue_.AdvanceReadPosition(); }
  void AdvanceWritePosition() { queue_.AdvanceWritePosition(); }
  bool IsFull() const { return queue_.IsFull(); }
  bool IsEmpty() const { return queue_.IsEmpty(); }

  // Pushing with sanity checking against the sentinel.
  void Push(const T& item) {
    CHECK(item != kEndOfQueueSentinel);
    queue_.Push(item);
  }
  void BlockPush(const T& item) {
    CHECK(item != kEndOfQueueSentinel);
    queue_.BlockPush(item);
  }
  bool TryPush(const T& item) {
    CHECK(item != kEndOfQueueSentinel);
    return queue_.TryPush(item);
  }

  // Pushing the sentinel.
  void SignalEndOfQueue() {
    queue_.BlockPush(kEndOfQueueSentinel);
    queue_.AdvanceWritePosition();
  }

  // Non-blocking pop with sentinel checking. Returns false if no items
  // are available or the end is reached.
  bool TryPop(T& item) {
    if (!queue_.TryPop(item)) {
      return false;
    }
    if (item == kEndOfQueueSentinel) {
      return false;
    }
    return true;
  }

  // Blocking pop with sentinel checking. Returns false if and only if the end
  // is reached.
  bool BlockPop(T& item) {
    queue_.BlockPop(item);
    return item != kEndOfQueueSentinel;
  }

 private:
  SPSCQueue<T, 2048> queue_;
  static constexpr T kEndOfQueueSentinel{nullptr};
  TF_DISALLOW_COPY_AND_ASSIGN(InfeedQueue);
};

class InfeedIterator {
 public:
  InfeedIterator(tensorflow::FunctionLibraryRuntime* flr,
                 tensorflow::data::IteratorContext::Params params,
                 tensorflow::data::DatasetBase* dataset,
                 tensorflow::CancellationManager* cancellation_manager,
                 InfeedAllocator* infeed_allocator_, int64 replication_factor,
                 const std::vector<xla::Shape>& shapes,
                 const std::string& feed_id);

  Status GetNext(std::vector<tensorflow::Tensor>* outputs,
                 bool* end_of_sequence);

  const std::vector<Shape>& GetShapes() const;

  std::vector<std::vector<InfeedQueue*>>& GetInfeedQueues();

 private:
  const int64 replication_factor_;
  std::vector<Shape> shapes_;

  // Not owned.
  // Cancellation manager from the poplar executor.
  tensorflow::CancellationManager* cancellation_manager_;
  // Allocator that should be used for allocating buffers for infeeds.
  InfeedAllocator* infeed_allocator_;

  // Owned
  // Note that member order is important.
  // Resource manager used by the iterators.
  tensorflow::ResourceMgr resource_mgr_;
  // The device manager which contains the device for this iterator.
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr_;
  // FLIB Definition and PFLR which use the above device manager.
  std::unique_ptr<tensorflow::FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr_;
  // An unbounded thread pool used by this iterator.
  std::unique_ptr<tensorflow::data::UnboundedThreadPool> unbounded_thread_pool_;
  // A thread pool used by this iterator.
  std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool_;
  // Per iterator specific function cache.
  std::unique_ptr<tensorflow::data::FunctionHandleCache> function_handle_cache_;
  // Context used for the iterator.
  std::unique_ptr<tensorflow::data::IteratorContext> iterator_ctx_;
  // The device iterator used for this queue.
  std::unique_ptr<tensorflow::data::IteratorBase> iterator_;
  // Storage of infeed queues.
  using InfeedQueueStorage = std::unique_ptr<InfeedQueue, void (*)(void*)>;
  std::vector<std::vector<InfeedQueueStorage>> infeed_queues_;
  // Used by the accessor.
  std::vector<std::vector<InfeedQueue*>> infeed_queues_ptrs_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  //  TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_INFEED_ITERATOR_H_
