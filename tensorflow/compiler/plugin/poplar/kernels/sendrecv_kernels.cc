/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

namespace {

Rendezvous::DoneCallback MakeRecvCallback(OpKernelContext* ctx,
                                          AsyncOpKernel::DoneCallback done) {
  return [ctx, done](const Status& s, const Rendezvous::Args& send_args,
                     const Rendezvous::Args& recv_args, const Tensor& val,
                     bool is_dead) {
    ctx->SetStatus(s);
    if (s.ok()) {
      if (!is_dead) {
        ctx->set_output(0, val);
      }
    }
    done();
  };
}

}  // namespace

class IpuSendToHostOp : public XlaOpKernel {
 public:
  explicit IpuSendToHostOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string send_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
    string recv_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
    uint64 send_device_incarnation;
    OP_REQUIRES_OK(
        ctx,
        ctx->GetAttr("send_device_incarnation",
                     reinterpret_cast<int64_t*>(&send_device_incarnation)));
    string tensor_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));

    rendezvous_key_ =
        Rendezvous::CreateKey(send_device, send_device_incarnation, recv_device,
                              tensor_name, FrameAndIter{0, 0});
  }

  void Compile(XlaOpKernelContext* ctx) override {
    XlaCompiler* compiler = ctx->compiler();
    xla::XlaBuilder* builder = ctx->builder();

    const xla::XlaOp token = CreateToken(builder);
    const xla::XlaOp input = ctx->Input(0);
    const TensorShape input_shape = ctx->InputShape(0);

    const DataType dtype = ctx->input_type(0);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, input_shape, &xla_shape));

    xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap attributes;
    attributes.AddAttribute("rendezvous_key", rendezvous_key_);

    const xla::XlaOp send_to_host = xla::CustomCall(
        ctx->builder(), PoplarOp_Name(PoplarOp::SendToHost), {input},
        xla::ShapeUtil::MakeNil(), attributes.Serialise());
  }

 private:
  string rendezvous_key_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuSendToHostOp);
};

class IpuRecvAtHostOp : public AsyncOpKernel {
 public:
  explicit IpuRecvAtHostOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));

    string send_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
    string recv_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
    uint64 send_device_incarnation;
    OP_REQUIRES_OK(
        ctx,
        ctx->GetAttr("send_device_incarnation",
                     reinterpret_cast<int64_t*>(&send_device_incarnation)));
    string tensor_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));

    const string full_key =
        Rendezvous::CreateKey(send_device, send_device_incarnation, recv_device,
                              tensor_name, FrameAndIter{0, 0});

    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(full_key, &parsed_key_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());

    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());

    auto* stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();

    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    poplar_executor->GetRendezvous()->RecvAsync(
        parsed_key_, Rendezvous::Args{},
        MakeRecvCallback(ctx, std::move(done)));
  }

 private:
  int device_ordinal_;
  Rendezvous::ParsedKey parsed_key_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuRecvAtHostOp);
};

REGISTER_XLA_OP(Name("IpuSendToHost").Device(DEVICE_IPU_XLA_JIT),
                IpuSendToHostOp);

REGISTER_KERNEL_BUILDER(Name("IpuRecvAtHost").Device(DEVICE_CPU),
                        IpuRecvAtHostOp);

}  // namespace tensorflow
