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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_CONFIG_IPU_CONFIG_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_CONFIG_IPU_CONFIG_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace ipu {

using IpuOptions = xla::poplarplugin::IpuOptions;

// Utility type for configuring IPUs with the
// required default options.
struct IpuConfig {
  IpuConfig();

  IpuOptions options;
};

// Apply the given configuration to the IPUs.
// This can only be done once per-process.
Status ConfigureIpuSystem(const IpuConfig& config);

// Get the configuration of the IPU system. Will be empty
// if the system has yet to be configured, otherwise an
// IpuConfig object will be returned for each configured
// device.
Status GetIpuConfig(std::vector<IpuConfig>& configs);

}  // namespace ipu
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_CONFIG_IPU_CONFIG_H_
