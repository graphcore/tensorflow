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

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/ipu_devices.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace xp = ::xla::poplarplugin;

namespace tensorflow {

class IpuSummaryOp : public OpKernel {
 public:
  explicit IpuSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~IpuSummaryOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

    std::list<tensorflow::IpuTraceEvent> out;
    OP_REQUIRES_OK(ctx, p->GetCompilerEvents(out));

    int num = out.size();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("out", TensorShape({num}), &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    unsigned i = 0;
    for (auto& e : out) {
      std::string str;
      e.SerializeToString(&str);

      output_flat(i) = str;
      i++;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuSummaryOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuEventTrace").Device(DEVICE_CPU), IpuSummaryOp);

class IpuConfigureHardwareOp : public OpKernel {
 public:
  explicit IpuConfigureHardwareOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_));
  }
  ~IpuConfigureHardwareOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

    xla::poplarplugin::IpuOptions options;
    options.ParseFromString(config_);

    OP_REQUIRES_OK(ctx, p->ConfigurePoplarDevices(options));
  }

 private:
  std::string config_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuConfigureHardwareOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuConfigureHardware").Device(DEVICE_CPU),
                        IpuConfigureHardwareOp);

class IpuClearAllXlaCompilationCaches : public OpKernel {
 public:
  explicit IpuClearAllXlaCompilationCaches(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto& active_devices = IPUDevices::GetActiveDevices();
    OP_REQUIRES_OK(ctx, active_devices.ClearXlaCompilationCache());
  }
};
REGISTER_KERNEL_BUILDER(
    Name("IpuClearAllXlaCompilationCaches").Device(DEVICE_CPU),
    IpuClearAllXlaCompilationCaches);

class IpuResetDevicesOp : public OpKernel {
 public:
  explicit IpuResetDevicesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* poplar_platform =
        static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());
    OP_REQUIRES_OK(ctx, poplar_platform->ResetExecutors());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuResetDevicesOp);
};
REGISTER_KERNEL_BUILDER(Name("IpuResetDevices").Device(DEVICE_CPU),
                        IpuResetDevicesOp);

class IpuResetSeedOp : public OpKernel {
 public:
  explicit IpuResetSeedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device", &dev_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("identical_replicas", &identical_replicas_));
  }
  ~IpuResetSeedOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

    DeviceNameUtils::ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(dev_name_, &parsed_name);

    OP_REQUIRES(ctx, parsed_name.has_id,
                errors::InvalidArgument("Invalid device name %s", dev_name_));

    OP_REQUIRES_OK(ctx,
                   p->ResetSeed(parsed_name.id, seed_, identical_replicas_));
  }

 private:
  std::string dev_name_;
  int seed_;
  bool identical_replicas_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuResetSeedOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuResetSeed").Device(DEVICE_CPU),
                        IpuResetSeedOp);

class IpuModelUsedOp : public OpKernel {
 public:
  explicit IpuModelUsedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~IpuModelUsedOp() override{};

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("out", TensorShape({1}), &output_tensor));
    auto output_flat = output_tensor->flat<bool>();
    output_flat(0) = xp::PoplarXlaFlags::Get().use_ipu_model;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuModelUsedOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuModelUsed").Device(DEVICE_CPU),
                        IpuModelUsedOp);

class IpuGetConfigurationOp : public OpKernel {
 public:
  explicit IpuGetConfigurationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~IpuGetConfigurationOp() override {}

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

    // Get IpuOptions from poplar executors.
    std::vector<xp::IpuOptions> out_opts;
    OP_REQUIRES_OK(ctx, p->GetIpuOptions(out_opts));

    // Serialize and write out.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("out", TensorShape({out_opts.size()}),
                                        &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    for (size_t i = 0; i < out_opts.size(); i++) {
      std::string opt_str;
      out_opts[i].SerializeToString(&opt_str);
      output_flat(i) = opt_str;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuGetConfigurationOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuGetConfiguration").Device(DEVICE_CPU),
                        IpuGetConfigurationOp);

class IpuGetNumDevicesOp : public OpKernel {
 public:
  explicit IpuGetNumDevicesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device", &dev_name_));
  }
  ~IpuGetNumDevicesOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

    DeviceNameUtils::ParsedName parsed_name;
    DeviceNameUtils::ParseFullName(dev_name_, &parsed_name);

    OP_REQUIRES(ctx, parsed_name.has_id,
                errors::InvalidArgument("Invalid device name %s", dev_name_));
    auto status_or = p->GetNumIpusForDevice(parsed_name.id);
    OP_REQUIRES_OK(ctx, status_or.status());

    // Serialize and write out.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("out", TensorShape({}), &output_tensor));
    auto output_flat = output_tensor->flat<int64>();
    output_flat(0) = status_or.ValueOrDie();
  }

 private:
  std::string dev_name_;
  TF_DISALLOW_COPY_AND_ASSIGN(IpuGetNumDevicesOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuGetNumDevices").Device(DEVICE_CPU),
                        IpuGetNumDevicesOp);

class IpuUseSyntheticDataForOp : public OpKernel {
 public:
  explicit IpuUseSyntheticDataForOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    int category_int;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("synthetic_data_category", &category_int));

    if (!xp::SyntheticDataCategory_IsValid(category_int)) {
      ctx->CtxFailure(errors::InvalidArgument(
          "Invalid synthetic data category: %d", category_int));
      return;
    }
    category_ = static_cast<xp::SyntheticDataCategory>(category_int);
  }

  ~IpuUseSyntheticDataForOp() = default;

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("out", TensorShape({1}), &output_tensor));
    auto output_flat = output_tensor->flat<bool>();
    output_flat(0) = xp::UseSyntheticDataFor(category_);
  }

 private:
  xp::SyntheticDataCategory category_;
  TF_DISALLOW_COPY_AND_ASSIGN(IpuUseSyntheticDataForOp);
};

REGISTER_KERNEL_BUILDER(Name("IpuUseSyntheticDataFor").Device(DEVICE_CPU),
                        IpuUseSyntheticDataForOp);
}  // namespace tensorflow
