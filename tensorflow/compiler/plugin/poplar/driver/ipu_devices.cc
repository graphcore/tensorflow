/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/ipu_devices.h"

#include <string>
#include <thread>

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

namespace tensorflow {
namespace {

std::once_flag clearing_cache_message_printed;
void PrintClearingCacheMessage(const std::string& device_name) {
  LOG(INFO) << "Clearing XLA compilation cache of device '" << device_name
            << "' while executable caching is disabled. To enable the "
               "executable cache"
            << " and potentially save recompilation time set the "
               "'executable_cache_path'"
            << " option, e.g "
               "'TF_POPLAR_FLAGS=--executable_cache_path=/path/to/storage'"
            << ". This is logged at most once.";
}

Status DeleteXlaCompilationCache(XlaDevice* device) {
  ResourceMgr* resource_manager = device->resource_manager();
  CHECK(resource_manager != nullptr);

  XlaCompilationCache* cache = nullptr;
  Status lookup_status = resource_manager->Lookup(
      resource_manager->default_container(), "xla_cache", &cache);
  const bool has_cache = lookup_status.ok() && cache;

  if (has_cache) {
    // Remove the extra reference acquired during the lookup.
    cache->Unref();
    // Fragile but the arguments to Delete correspond to the values used when
    // constructing the cache in CompileToLocalExecutable (kernels/xla_ops.cc).
    // It'll be lazily recreated if the device is used for compilation again.
    TF_RETURN_IF_ERROR(resource_manager->Delete<XlaCompilationCache>(
        resource_manager->default_container(), "xla_cache"));

    const bool executable_cache_disabled =
        xla::poplarplugin::PoplarXlaFlags::Get().executable_cache_path.empty();
    if (executable_cache_disabled) {
      std::call_once(clearing_cache_message_printed, PrintClearingCacheMessage,
                     device->name());
    }
  }

  return Status::OK();
}

bool IsIpuDevice(XlaDevice* device) {
  const auto& attributes = device->attributes();
  return attributes.device_type() == DEVICE_XLA_IPU;
}

}  // namespace

IPUDevices& IPUDevices::GetActiveDevices() {
  static IPUDevices active_devices;
  return active_devices;
}

void IPUDevices::Add(XlaDevice* device) {
  CHECK(device != nullptr);
  CHECK(IsIpuDevice(device));

  std::lock_guard<std::mutex> lock(devices_mutex_);
  devices_.insert(device);
}

void IPUDevices::Remove(XlaDevice* device) {
  CHECK(device != nullptr);
  CHECK(IsIpuDevice(device));

  std::lock_guard<std::mutex> lock(devices_mutex_);
  devices_.erase(device);
}

Status IPUDevices::ClearXlaCompilationCache() {
  std::lock_guard<std::mutex> lock(devices_mutex_);

  for (auto* device : devices_) {
    TF_RETURN_IF_ERROR(DeleteXlaCompilationCache(device));
  }

  return Status::OK();
}

}  // namespace tensorflow
