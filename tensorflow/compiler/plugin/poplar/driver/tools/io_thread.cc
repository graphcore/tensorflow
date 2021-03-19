/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/io_thread.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace poplarplugin {

IOThread::IOThread(const std::string& name, IOFunction fn,
                   const tensorflow::ThreadOptions& options)
    : cancelled_(false) {
  thread_.reset(tensorflow::Env::Default()->StartThread(
      options, name, [options, name, f = std::move(fn), this]() {
        // StartThread currently ignores `options`.
        // Explicitly set the NUMA node if one was requested.
        if (options.numa_node != tensorflow::port::kNUMANoAffinity) {
          tensorflow::port::NUMASetThreadNodeAffinity(options.numa_node);
        }
        auto status = f(cancelled_);
        if (!status.ok()) {
          LOG(INFO) << "Thread " << name
                    << " has finished with status: " << status.ToString();
        }
        cancelled_ = true;
      }));
}

IOThread::~IOThread() {
  // Cancel the thread.
  cancelled_ = true;
  // The destructor of `thread_` will now block until the thread has joined.
}

}  // namespace poplarplugin
}  // namespace xla
