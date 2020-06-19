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
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
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
    auto output_flat = output_tensor->flat<string>();

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

class IpuResetSeedOp : public OpKernel {
 public:
  explicit IpuResetSeedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device", &dev_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
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

    OP_REQUIRES_OK(ctx, p->ResetSeed(parsed_name.id, seed_));
  }

 private:
  std::string dev_name_;
  int seed_;

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
    auto output_flat = output_tensor->flat<string>();

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
}  // namespace tensorflow
