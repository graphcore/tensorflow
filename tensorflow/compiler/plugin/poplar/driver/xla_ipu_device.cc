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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/kernels/no_op.h"

namespace xp = ::xla::poplarplugin;

namespace tensorflow {

class IpuDevice : public XlaDevice {
 public:
  IpuDevice(const SessionOptions& options, const XlaDevice::Options& devopts)
      : XlaDevice(options, devopts) {
    UseGpuDeviceInfo();
  }

  virtual ~IpuDevice() {}
};

class XlaIpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;

  virtual Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_IPU, ":0"));
    return Status::OK();
  }
};

Status XlaIpuDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_IPU, DEVICE_IPU_XLA_JIT);
  (void)registrations;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_IPU_XLA_JIT;
  registration.autoclustering_policy =
      XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = true;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_IPU, registration);

  auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
  if (!platform.ok()) {
    return platform.status();
  }

  auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  XlaDevice::Options devopts;
  devopts.platform = platform.ValueOrDie();
  devopts.device_name_prefix = name_prefix;
  devopts.compilation_device_name = DEVICE_IPU_XLA_JIT;
  devopts.device_name = DEVICE_XLA_IPU;

  int num_devices = p->VisibleDeviceCount();

  for (int ordinal = 0; ordinal < num_devices; ordinal++) {
    devopts.device_ordinal = ordinal;

    std::unique_ptr<Device> dev(new IpuDevice(options, devopts));
    devices->push_back(std::move(dev));
  }

  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaLocalLaunchOp,
                           GetIPUSupportedTypes());
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_IPU, XlaCompileOp,
                            GetIPUSupportedTypes());
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_IPU, XlaRunOp, GetIPUSupportedTypes());

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, GetIPUSupportedTypes());

}  // namespace tensorflow
