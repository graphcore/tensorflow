/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_XLA_IPU_COMMON_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_XLA_IPU_COMMON_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

const char* const DEVICE_XLA_IPU = "IPU";
const char* const DEVICE_IPU_XLA_JIT = "XLA_IPU_JIT";
const char* const PLATFORM_NAME = "Poplar";

const char* const DEVICE_XLA_POPIT = "POPIT";
const char* const DEVICE_POPIT_XLA_JIT = "XLA_POPIT_JIT";
const char* const POPIT_PLATFORM_NAME = "PopIt";

std::vector<DataType> GetIPUSupportedTypes();

bool OpFilter(KernelDef* kdef);

class XlaGraphcoreDeviceFactory : public DeviceFactory {
  const char* device_xla_;
  const char* device_xla_jit_;
  const char* platform_name_;

 public:
  XlaGraphcoreDeviceFactory(const char* device_xla, const char* device_xla_jit,
                            const char* platform_name)
      : device_xla_(device_xla),
        device_xla_jit_(device_xla_jit),
        platform_name_(platform_name) {}
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;

  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back(absl::StrCat("/physical_device:", device_xla_, ":0"));
    return Status::OK();
  }

  virtual std::unique_ptr<Device> CreateFromOptions(
      const SessionOptions& options,
      const XlaDevice::Options& devopts) const = 0;
};

}  // namespace tensorflow

#define REGISTER_IPU_XLA_DEVICES(device, factory)                            \
  REGISTER_LOCAL_DEVICE_FACTORY(device, factory);                            \
  REGISTER_XLA_LAUNCH_KERNEL(device, XlaLocalLaunchOp,                       \
                             GetIPUSupportedTypes());                        \
  REGISTER_XLA_COMPILE_KERNEL(device, XlaCompileOp, GetIPUSupportedTypes()); \
  REGISTER_XLA_RUN_KERNEL(device, XlaRunOp, GetIPUSupportedTypes());         \
  REGISTER_XLA_DEVICE_KERNELS(device, GetIPUSupportedTypes());

#endif
