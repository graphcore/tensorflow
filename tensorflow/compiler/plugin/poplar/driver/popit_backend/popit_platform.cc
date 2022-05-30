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
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform.h"

#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

PopItPlatform::PopItPlatform() : name_("PopIt") {
  CheckPoplarPackageHash();
  VLOG(1) << "Poplar version: " << poplar::versionString()
          << " Poplar package: " << poplar::packageHash();
}

PopItPlatform::~PopItPlatform() {}

se::Platform::Id PopItPlatform::id() const { return kPopItPlatformId; }

int PopItPlatform::VisibleDeviceCount() const {
  int num_devices = PoplarExecutor::GetDeviceManager()
                        .getDevices(poplar::TargetType::IPU, 1)
                        .size();

  if (num_devices == 0) {
    // Allow for 2 virtual devices
    num_devices = 2;
  }

  return num_devices;
}

const std::string& PopItPlatform::Name() const { return name_; }

StatusOr<se::StreamExecutor*> PopItPlatform::ExecutorForDevice(int ordinal) {
  return ExecutorForDeviceWithPluginConfig(ordinal, se::PluginConfig());
}

StatusOr<se::StreamExecutor*> PopItPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const se::PluginConfig& plugin_config) {
  se::StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

StatusOr<se::StreamExecutor*> PopItPlatform::GetExecutor(
    const se::StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<se::StreamExecutor>>
PopItPlatform::GetUncachedExecutor(const se::StreamExecutorConfig& config) {
  auto executor = absl::make_unique<se::StreamExecutor>(
      this, absl::make_unique<PopItExecutor>(), config.ordinal);
  TF_RETURN_IF_ERROR(executor->Init(config.device_options));

  return executor;
}

void PopItPlatform::RegisterTraceListener(
    std::unique_ptr<se::TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register Poplar trace listener";
}

void PopItPlatform::UnregisterTraceListener(se::TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister Poplar trace listener";
}

StatusOr<std::unique_ptr<se::DeviceDescription>>
PopItPlatform::DescriptionForDevice(int ordinal) const {
  se::internal::DeviceDescriptionBuilder builder;
  builder.set_name("PopIt");
  builder.set_platform_version("");

  return builder.Build();
}

static void InitializePopItPlatform() {
  SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(
      absl::make_unique<PopItPlatform>()));
}

}  // namespace poplarplugin
}  // namespace xla

REGISTER_MODULE_INITIALIZER(popit_platform,
                            xla::poplarplugin::InitializePopItPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);

REGISTER_MODULE_INITIALIZER_SEQUENCE(popit_platform, multi_platform_manager);
