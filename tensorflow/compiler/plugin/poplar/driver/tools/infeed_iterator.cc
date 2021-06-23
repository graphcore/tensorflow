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

#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_queue.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace xla {
namespace poplarplugin {
namespace {
const char kAnonymousCancellationManagerResource[] =
    "AnonymousCancellationManagerResource";

// Used to generate unique names for anonymous cancellation managers.
static std::atomic<int64> current_id_;

class CancellationManagerResource : public tensorflow::ResourceBase {
 public:
  CancellationManagerResource()
      : mu_(std::make_shared<tensorflow::mutex>()),
        cancelled_notification_(std::make_shared<absl::Notification>()) {}

  tensorflow::CancellationManager* cancellation_manager() {
    return &cancellation_manager_;
  }

  std::shared_ptr<tensorflow::mutex>& mutex() { return mu_; }

  std::shared_ptr<absl::Notification>& cancelled_notification() {
    return cancelled_notification_;
  }

  std::string DebugString() const override {
    return "Cancellation manager for InfeedIterator";
  }

 private:
  // We use a std::shared_ptr here because the either the dataset device or the
  // infeed device might be deleted first.
  // Lock makes sure that only one of the devices is deregistering from the
  // other one at a time.
  std::shared_ptr<tensorflow::mutex> mu_;
  // Notification indicates that one of the device has shut down already and the
  // other device doesn't need to do anything.
  std::shared_ptr<absl::Notification> cancelled_notification_;

  tensorflow::CancellationManager cancellation_manager_;
};

Status RegisterCancellationCallback(
    tensorflow::CancellationManager* cancellation_manager,
    std::function<void()> register_fn, std::function<void()>* deregister_fn) {
  tensorflow::CancellationToken token =
      cancellation_manager->get_cancellation_token();
  if (!cancellation_manager->RegisterCallback(token, std::move(register_fn))) {
    return Cancelled("Operation was cancelled");
  }
  *deregister_fn = [cancellation_manager, token]() {
    cancellation_manager->DeregisterCallback(token);
  };
  return Status::OK();
}
}  // namespace

/* static */ constexpr InfeedQueue::T InfeedQueue::kEndOfQueueSentinel;
InfeedQueue::InfeedQueue()
    : queue_(nullptr, [](tensorflow::TensorBuffer*& buffer) {
        if (buffer) {
          buffer->Unref();
          buffer = nullptr;
        }
      }) {}

InfeedIterator::InfeedIterator(tensorflow::FunctionLibraryRuntime* flr,
                               tensorflow::data::IteratorContext::Params params,
                               tensorflow::data::DatasetBase* dataset,
                               InfeedAllocator* infeed_allocator,
                               const std::vector<xla::Shape>& shapes,
                               const std::string& feed_id)
    : replication_factor_(0),
      shapes_(shapes),
      infeed_allocator_(infeed_allocator),
      buffer_position_(-1) {
  // Respect the user request for the number of threads.
  const int num_threads = PoplarXlaFlags::Get().max_infeed_threads > 0
                              ? PoplarXlaFlags::Get().max_infeed_threads
                              : tensorflow::port::MaxParallelism();

  // Wrap the given flr->device() in order to have access to any resources in
  // the resource manager of the device, which might be captured by ops in the
  // dataset pipeline. The device must outlive our usage here.
  device_mgr_ = absl::make_unique<tensorflow::DeviceMgr>(
      tensorflow::RenamedDevice::NewRenamedDevice(
          flr->device()->name(), flr->device(), /*owns_underlying=*/false,
          /*isolate_session_state=*/false));

  tensorflow::Device* device = device_mgr_->ListDevices()[0];

  // Create new `FunctionLibraryDefinition` and
  // `ProcessFunctionLibraryRuntime` with the new device manager so that we can
  // control the lifetime of the iterator.
  flib_def_ = absl::make_unique<tensorflow::FunctionLibraryDefinition>(
      *flr->GetFunctionLibraryDefinition());
  pflr_ = absl::make_unique<tensorflow::ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), tensorflow::Env::Default(), TF_GRAPH_DEF_VERSION,
      flib_def_.get(), tensorflow::OptimizerOptions{}, nullptr /* parent */);

  // Set up the thread pools.
  unbounded_thread_pool_ =
      absl::make_unique<tensorflow::data::UnboundedThreadPool>(
          tensorflow::Env::Default(), feed_id + "/unbounded_thread_pool");

  thread_pool_ = absl::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      feed_id + "/thread_pool", num_threads);
  tensorflow::thread::ThreadPool* thread_pool_ptr = thread_pool_.get();

  // Get the new FLR from the newly created PFLR and set up the cache.
  tensorflow::FunctionLibraryRuntime* new_flr = pflr_->GetFLR(device->name());
  function_handle_cache_ =
      absl::make_unique<tensorflow::data::FunctionHandleCache>(new_flr);

  // Given the previous params, create new params.
  tensorflow::data::IteratorContext::Params base_params(params);
  base_params.allocator_getter = [this](tensorflow::AllocatorAttributes) {
    return infeed_allocator_;
  };
  base_params.cancellation_manager = &cancellation_manager_;
  base_params.env = tensorflow::Env::Default();
  base_params.flr = new_flr;
  base_params.function_handle_cache = function_handle_cache_.get();
  base_params.resource_mgr = &resource_mgr_;
  base_params.runner = [thread_pool_ptr](std::function<void()> c) {
    thread_pool_ptr->Schedule(std::move(c));
  };
  base_params.runner_threadpool_size = num_threads;
  base_params.thread_factory = unbounded_thread_pool_->get_thread_factory();
  base_params.thread_pool = unbounded_thread_pool_.get();

  // Insert a resource into the device to notify when the device is about to be
  // destroyed.
  // The resource manager will take ownership of this pointer.
  CancellationManagerResource* resource = new CancellationManagerResource();
  const std::string unique_name = absl::StrCat(
      kAnonymousCancellationManagerResource, current_id_.fetch_add(1));
  Status s = device->resource_manager()->Create<CancellationManagerResource>(
      kAnonymousCancellationManagerResource, unique_name, resource);
  if (!s.ok()) {
    LOG(FATAL) << s.ToString();
  }

  // Create local copies of the shared_ptrs from the resource for cooridinating
  // shut down.
  mu_ = resource->mutex();
  cancelled_notification_ = resource->cancelled_notification();

  // Connect the cancellation managers so that when the device is being
  // destroyed, the dataset stops running.
  s = RegisterCancellationCallback(resource->cancellation_manager(),
                                   [this, resource]() {
                                     auto mu = resource->mutex();
                                     auto notification =
                                         resource->cancelled_notification();
                                     tensorflow::mutex_lock lk(*mu);
                                     // Check whether other device has finished,
                                     // if not, cancel execution before this
                                     // device is destroyed.
                                     if (!notification->HasBeenNotified()) {
                                       cancellation_manager_.StartCancel();
                                     }
                                     notification->Notify();
                                   },
                                   &deregister_cancellation_manager_parent_fn_);

  if (!s.ok()) {
    LOG(FATAL) << s.ToString();
  }

  // Create the context for the iterator.
  iterator_ctx_ = absl::make_unique<tensorflow::IteratorContext>(base_params);
  // Create the iterator.
  s = dataset->MakeIterator(iterator_ctx_.get(), feed_id, &iterator_);
  if (!s.ok()) {
    LOG(FATAL) << s.ToString();
  }
}

InfeedIterator::~InfeedIterator() {
  tensorflow::mutex_lock lk(*mu_);
  cancellation_manager_.StartCancel();
  // If the other device hasn't been destroyed yet, remove the callback
  // connection.
  if (!cancelled_notification_->HasBeenNotified()) {
    deregister_cancellation_manager_parent_fn_();
  }
  cancelled_notification_->Notify();
}

Status InfeedIterator::GetNext(std::vector<tensorflow::Tensor>* outputs,
                               bool* end_of_sequence) {
  if (cancellation_manager_.IsCancelled()) {
    *end_of_sequence = true;
  } else {
    // Here we preload N (buffer_size) elements every N calls, yielding one of
    // those values every time.
    auto buffer_size_ = buffer_.size();
    if (buffer_position_ == buffer_size_) {
      // We need to load elements.
      if (!iterator_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      *end_of_sequence = false;
      uint32 buffer_idx = 0;
      for (uint32 i = 0; i < buffer_size_ && !*end_of_sequence; ++i) {
        TF_RETURN_IF_ERROR(iterator_->GetNext(iterator_ctx_.get(), &buffer_[i],
                                              end_of_sequence));
        if (!*end_of_sequence) {
          buffer_idx++;
        } else {
          iterator_.reset();
        }
      }

      // We could not load enough data hence we are dropping it.
      if (buffer_idx < buffer_size_) {
        *end_of_sequence = true;
        return Status::OK();
      }
      buffer_position_ = 0;
    }
    // Set the output.
    *outputs = std::move(buffer_[buffer_position_]);
    // We can move the buffer position.
    ++buffer_position_;
    *end_of_sequence = false;
  }

  return Status::OK();
}

const std::vector<Shape>& InfeedIterator::GetShapes() const { return shapes_; }

std::vector<std::vector<InfeedQueue*>>& InfeedIterator::GetInfeedQueues() {
  return infeed_queues_ptrs_;
}

void InfeedIterator::SignalAllQueuesToEnd() {
  for (auto& queues : infeed_queues_ptrs_) {
    for (auto& queue : queues) {
      queue->SignalEndOfQueue();
    }
  }
}

bool InfeedIterator::HasReplicationFactor() const {
  return replication_factor_ > 0;
}

int64 InfeedIterator::ReplicationFactor() const { return replication_factor_; }

void InfeedIterator::SetReplicationFactor(int64 replication_factor) {
  CHECK_GT(replication_factor, 0);

  replication_factor_ = replication_factor;
  buffer_position_ = replication_factor;
  buffer_.resize(replication_factor);

  infeed_queues_ptrs_.resize(replication_factor);
  infeed_queues_.resize(replication_factor);

  // Create the queues.
  for (int64 replica_id = 0; replica_id < replication_factor; replica_id++) {
    infeed_queues_ptrs_[replica_id].clear();
    infeed_queues_[replica_id].clear();

    for (uint64 i = 0; i < shapes_.size(); i++) {
      void* buffer = tensorflow::port::AlignedMalloc(sizeof(InfeedQueue), 64);
      infeed_queues_[replica_id].emplace_back(
          new (buffer) InfeedQueue(), [](InfeedQueue* ptr) {
            ptr->~InfeedQueue();
            tensorflow::port::AlignedFree(ptr);
          });
      infeed_queues_ptrs_[replica_id].emplace_back(
          infeed_queues_[replica_id].back().get());
    }
  }
}

}  // namespace poplarplugin
}  // namespace xla
