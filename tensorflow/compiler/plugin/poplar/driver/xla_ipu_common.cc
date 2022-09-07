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

#include <vector>

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace xla {

namespace poplarplugin {}

}  // namespace xla

namespace xp = ::xla::poplarplugin;

namespace tensorflow {

std::vector<DataType> GetIPUSupportedTypes() {
  // Lambda which will get all the supported types given the flags.
  auto get_types = [] {
    std::vector<DataType> supported = {DT_INT32, DT_INT64, DT_FLOAT, DT_HALF,
                                       DT_BOOL};
    if (getenv("TF_POPLAR_GFLOAT") != nullptr) {
      supported.push_back(DT_INT8);
      supported.push_back(DT_INT16);
    }
    return supported;
  };

  static std::vector<DataType> supported_types = get_types();
  return supported_types;
}

bool OpFilter(KernelDef* kdef) {
  if (kdef->op() == "Angle") return false;
  if (kdef->op() == "Complex") return false;
  if (kdef->op() == "ComplexAbs") return false;
  if (kdef->op() == "Conj") return false;
  if (kdef->op() == "FFT") return false;
  if (kdef->op() == "FFT2D") return false;
  if (kdef->op() == "FFT3D") return false;
  if (kdef->op() == "IFFT") return false;
  if (kdef->op() == "IFFT2D") return false;
  if (kdef->op() == "IFFT3D") return false;
  if (kdef->op() == "Imag") return false;
  if (kdef->op() == "MaxPoolGradGrad") return false;
  if (kdef->op() == "MaxPool3DGradGrad") return false;
  if (kdef->op() == "NonMaxSuppressionV4") return false;
  if (kdef->op() == "Real") return false;

  if (kdef->op() == "Assert") {
    AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
  }

  if (kdef->op() == "IpuF8Matmul") {
    AddDtypeToKernelDefConstraint("lhs_type", DT_UINT8, kdef);
    AddDtypeToKernelDefConstraint("rhs_type", DT_UINT8, kdef);
  }

  if (kdef->op() == "Const") {
    AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
    AddDtypeToKernelDefConstraint("dtype", DT_UINT8, kdef);
    AddDtypeToKernelDefConstraint("dtype", DT_INT8, kdef);
  }
  if (kdef->op() == "Function" || kdef->op() == "Pipeline" ||
      kdef->op() == "PipelineStage" || kdef->op() == "PipelineStageBackward" ||
      kdef->op() == "ResourceUpdate" || kdef->op() == "MultiConv") {
    AddDtypeToKernelDefConstraint("Tin", DT_RESOURCE, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_RESOURCE, kdef);
    AddDtypeToKernelDefConstraint("Tin", DT_VARIANT, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_VARIANT, kdef);
    // Add support for (u)int8.
    AddDtypeToKernelDefConstraint("Tin", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("Tin", DT_UINT8, kdef);
    AddDtypeToKernelDefConstraint("Tout", DT_UINT8, kdef);
  }

  // Support (u)int8 for data feeding to the IPU.
  if (kdef->op() == "Cast") {
    AddDtypeToKernelDefConstraint("SrcT", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("SrcT", DT_UINT8, kdef);
    AddDtypeToKernelDefConstraint("DstT", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("DstT", DT_UINT8, kdef);
  }

  if (kdef->op() == "Identity" || kdef->op() == "ExpandDims" ||
      kdef->op() == "Reshape" || kdef->op() == "BroadcastTo") {
    AddDtypeToKernelDefConstraint("T", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("T", DT_UINT8, kdef);
  }

  if (kdef->op() == "_Arg" || kdef->op() == "_Retval") {
    AddDtypeToKernelDefConstraint("T", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("T", DT_UINT8, kdef);
  }

  if (kdef->op() == "PopDatastreamInfeedDequeue" ||
      kdef->op() == "PopDatastreamOutfeedEnqueue") {
    AddDtypeToKernelDefConstraint("output_types", DT_INT8, kdef);
    AddDtypeToKernelDefConstraint("output_types", DT_UINT8, kdef);
  }

  if (kdef->op() == "NormaliseImage") {
    AddDtypeToKernelDefConstraint("Tin", DT_UINT8, kdef);
  }

  return true;
}

Status XlaGraphcoreDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(device_xla_, device_xla_jit_);
  (void)registrations;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = device_xla_jit_;
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
  XlaOpRegistry::RegisterCompilationDevice(device_xla_, registration);

  auto platform = se::MultiPlatformManager::PlatformWithName(platform_name_);
  if (!platform.ok()) {
    return platform.status();
  }

  auto* p = platform.ValueOrDie();

  XlaDevice::Options devopts;
  devopts.platform = platform.ValueOrDie();
  devopts.device_name_prefix = name_prefix;
  devopts.compilation_device_name = device_xla_jit_;
  devopts.device_name = device_xla_;
  devopts.supports_may_alias_resource_update = false;

  int num_devices = p->VisibleDeviceCount();

  for (int ordinal = 0; ordinal < num_devices; ordinal++) {
    devopts.device_ordinal = ordinal;
    devices->push_back(CreateFromOptions(options, devopts));
  }

  return Status::OK();
}

}  // namespace tensorflow
