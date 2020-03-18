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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/spsc_queue.h"

#include "tensorflow/compiler/xla/shape.h"

#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/kernels/data/unbounded_thread_pool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace {
std::string GetTaskName(const tensorflow::Device* device) {
  using Utils = tensorflow::DeviceNameUtils;

  CHECK(device != nullptr);
  const std::string& full_name = device->name();

  Utils::ParsedName parsed_name;
  CHECK(Utils::ParseFullName(full_name, &parsed_name))
      << "Failed to parse: " << full_name;

  std::string task_name;
  CHECK(Utils::GetTaskName(parsed_name, &task_name))
      << "Failed to get task name from: " << full_name;

  return task_name;
}
}  // namespace

namespace xla {
namespace poplarplugin {

/* static */ constexpr InfeedQueue::T InfeedQueue::kEndOfQueueSentinel;
InfeedQueue::InfeedQueue()
    : queue_(nullptr, [](tensorflow::TensorBuffer*& buffer) {
        if (buffer) {
          buffer->Unref();
          buffer = nullptr;
        }
      }) {}

InfeedIterator::InfeedIterator(
    tensorflow::FunctionLibraryRuntime* flr,
    tensorflow::data::IteratorContext::Params params,
    tensorflow::data::DatasetBase* dataset,
    tensorflow::CancellationManager* cancellation_manager,
    InfeedAllocator* infeed_allocator, int64 replication_factor,
    const std::vector<xla::Shape>& shapes, const std::string& feed_id)
    : replication_factor_(replication_factor),
      shapes_(shapes),
      cancellation_manager_(cancellation_manager),
      infeed_allocator_(infeed_allocator),
      infeed_queues_(replication_factor),
      infeed_queues_ptrs_(replication_factor) {
  // Respect the user request for the number of threads.
  const int num_threads = PoplarXlaFlags::Get().max_infeed_threads > 0
                              ? PoplarXlaFlags::Get().max_infeed_threads
                              : tensorflow::port::MaxParallelism();

  // Create the CPU device the base iterator runs on.
  // First set up the options.
  tensorflow::SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  // Get the task name (might vary in distributed contexts).
  const std::string task_name = GetTaskName(flr->device());
  // Create the device manager.
  device_mgr_ = absl::make_unique<tensorflow::StaticDeviceMgr>(
      tensorflow::DeviceFactory::NewDevice("CPU", options, task_name));

  tensorflow::Device* device = device_mgr_->ListDevices()[0];

  // Create new `FunctionLibraryDefinition` and
  // `ProcessFunctionLibraryRuntime` with the new device manager so that we can
  // control the lifetime of the iterator.
  flib_def_ = absl::make_unique<tensorflow::FunctionLibraryDefinition>(
      *flr->GetFunctionLibraryDefinition());
  pflr_ = absl::make_unique<tensorflow::ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), tensorflow::Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def_.get(), tensorflow::OptimizerOptions{});

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
  base_params.cancellation_manager = cancellation_manager_;
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

  // Create the context for the iterator.
  iterator_ctx_ = absl::make_unique<tensorflow::IteratorContext>(base_params);
  // Create the iterator.
  Status s = dataset->MakeIterator(iterator_ctx_.get(), feed_id, &iterator_);
  if (!s.ok()) {
    LOG(FATAL) << s.ToString();
  }

  // Create the queues.
  for (int64 replica_id = 0; replica_id < replication_factor; replica_id++) {
    for (uint64 i = 0; i < shapes.size(); i++) {
      void* ptr = tensorflow::port::AlignedMalloc(sizeof(InfeedQueue), 64);
      infeed_queues_[replica_id].emplace_back(new (ptr) InfeedQueue());
      infeed_queues_ptrs_[replica_id].emplace_back(
          infeed_queues_[replica_id].back().get());
    }
  }
}

Status InfeedIterator::GetNext(std::vector<tensorflow::Tensor>* outputs,
                               bool* end_of_sequence) {
  TF_RETURN_IF_ERROR(
      iterator_->GetNext(iterator_ctx_.get(), outputs, end_of_sequence));
  return Status::OK();
}

const std::vector<Shape>& InfeedIterator::GetShapes() const { return shapes_; }

std::vector<std::vector<InfeedQueue*>>& InfeedIterator::GetInfeedQueues() {
  return infeed_queues_ptrs_;
}

}  // namespace poplarplugin
}  // namespace xla
