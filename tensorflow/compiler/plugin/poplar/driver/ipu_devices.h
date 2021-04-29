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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_IPU_DEVICES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_IPU_DEVICES_H_

#include <mutex>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class XlaDevice;

// Simple class for manipulating the active IPU devices.
// Operates on XlaDevice objects as adding an equivalent
// function on the IpuDevice introduces a circular dependency
// between the kernels and the poplar lib modules.
class IPUDevices {
 public:
  static IPUDevices& GetActiveDevices();

  void Add(XlaDevice* device);
  void Remove(XlaDevice* device);

  Status ClearXlaCompilationCache();

 private:
  IPUDevices() = default;
  IPUDevices(const IPUDevices&) = delete;
  IPUDevices& operator=(const IPUDevices&) = delete;

  std::unordered_set<XlaDevice*> devices_;
  std::mutex devices_mutex_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_IPU_DEVICES_H_
