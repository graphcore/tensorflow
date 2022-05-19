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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_PLATFORM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_PLATFORM_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace se = stream_executor;

namespace tensorflow {
class IpuTraceEvent;
}

namespace poplar {
class Engine;
}

namespace xla {
namespace poplarplugin {

class PopItPlatform : public se::Platform {
 public:
  PopItPlatform();
  ~PopItPlatform() override;

  Platform::Id id() const override;

  int VisibleDeviceCount() const override;

  const std::string& Name() const override;

  StatusOr<se::StreamExecutor*> ExecutorForDevice(int ordinal) override;

  StatusOr<se::StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const se::PluginConfig& config) override;

  StatusOr<se::StreamExecutor*> GetExecutor(
      const se::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

  void RegisterTraceListener(std::unique_ptr<se::TraceListener>) override;

  void UnregisterTraceListener(se::TraceListener* listener) override;

  StatusOr<std::unique_ptr<se::DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;

 private:
  // This platform's name.
  std::string name_;

  // Cache of created StreamExecutors.
  se::ExecutorCache executor_cache_;

  SE_DISALLOW_COPY_AND_ASSIGN(PopItPlatform);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_PLATFORM_H_
