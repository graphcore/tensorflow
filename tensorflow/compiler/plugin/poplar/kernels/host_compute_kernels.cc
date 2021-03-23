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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace tensorflow {

namespace {

xla::StatusOr<xla::poplarplugin::PoplarExecutor*> GetPoplarExecutor(
    int device_ordinal) {
  TF_ASSIGN_OR_RETURN(auto* platform,
                      se::MultiPlatformManager::PlatformWithName("Poplar"));

  auto* poplar_platform =
      static_cast<xla::poplarplugin::PoplarPlatform*>(platform);

  TF_ASSIGN_OR_RETURN(auto* stream_executor,
                      poplar_platform->ExecutorForDevice(device_ordinal));

  return static_cast<xla::poplarplugin::PoplarExecutor*>(
      stream_executor->implementation());
}

string CreateRendezvousKey(const string& key, const string& send_device,
                           const string& recv_device) {
  const uint64 send_device_incarnation = 0;
  return Rendezvous::CreateKey(send_device, send_device_incarnation,
                               recv_device, key, FrameAndIter{0, 0});
}

// TODO(hakons): These device strings do not need to match the actual
// devices, they only have to be unique for each stream. The unique
// "key" for each host computation should make sure this is the case.
// Should maybe make this more clear somehow.

string CreateSendRendezvousKey(const string& key) {
  return CreateRendezvousKey(key, "/device:IPU:0", "/device:CPU:0");
}

string CreateRecvRendezvousKey(const string& key) {
  return CreateRendezvousKey(key, "/device:CPU:0", "/device:IPU:0");
}

string CreateSendRendezvousKey(const string& key, int index) {
  return CreateSendRendezvousKey(strings::StrCat(key, ":", index));
}

string CreateRecvRendezvousKey(const string& key, int index) {
  return CreateRecvRendezvousKey(strings::StrCat(key, ":", index));
}

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

class XlaHostComputeOp : public XlaOpKernel {
 public:
  explicit XlaHostComputeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0 || ctx->num_outputs() > 0,
        errors::InvalidArgument(
            "Outside compilation scope must have either input or output"));

    std::vector<TensorShape> shapes;
    std::vector<tensorflow::DataType> types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutputs", &types));

    OP_REQUIRES(
        ctx, shapes.size() == ctx->num_outputs(),
        errors::InvalidArgument("All output shapes from outside compilation "
                                "scope must be statically known"));

    OP_REQUIRES(
        ctx, shapes.size() == types.size(),
        errors::InvalidArgument("Must have same number of shapes and types"));

    for (size_t i = 0; i < shapes.size(); ++i) {
      xla::PrimitiveType xla_type;
      OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(types[i], &xla_type));
      recv_shapes_.push_back(TensorShapeToXLAShape(xla_type, shapes[i]));
    }

    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::Shape> send_shapes;
    for (size_t i = 0; i < ctx->num_inputs(); ++i) {
      xla::Shape shape;
      OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(ctx->input_type(i),
                                                ctx->InputShape(i), &shape));
      send_shapes.push_back(shape);
      BuildSendToHost(ctx, i, shape);
    }

    for (int i = 0; i < ctx->num_outputs(); ++i) {
      if (i < send_shapes.size() && send_shapes[i] == recv_shapes_[i]) {
        // Pass the matching input tensor that can potentially be used in-place.
        BuildRecvFromHost(ctx, i, {ctx->Input(i)});
      } else {
        BuildRecvFromHost(ctx, i);
      }
    }
  }

 private:
  void BuildSendToHost(XlaOpKernelContext* ctx, int index,
                       const xla::Shape& shape) {
    const xla::XlaOp input = ctx->Input(index);

    const auto rendezvous_key = CreateSendRendezvousKey(key_, index);

    xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap attributes;
    attributes.AddAttribute("rendezvous_key", rendezvous_key);

    const xla::XlaOp send_to_host = xla::CustomCall(
        ctx->builder(), PoplarOp_Name(PoplarOp::SendToHost), {input},
        xla::ShapeUtil::MakeNil(), attributes.Serialise());
  }

  void BuildRecvFromHost(XlaOpKernelContext* ctx, int index,
                         absl::Span<const xla::XlaOp> inputs = {}) {
    const auto rendezvous_key = CreateRecvRendezvousKey(key_, index);

    xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap attributes;
    attributes.AddAttribute("rendezvous_key", rendezvous_key);

    const xla::XlaOp output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(PoplarOp::RecvFromHost),
                        inputs, recv_shapes_[index], attributes.Serialise());

    ctx->SetOutput(index, output);
  }

  string key_;
  std::vector<xla::Shape> recv_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaHostComputeOp);
};

class XlaRecvAtHostOp : public AsyncOpKernel {
 public:
  explicit XlaRecvAtHostOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));

    OP_REQUIRES(ctx, ctx->num_outputs() == 1,
                errors::InvalidArgument("Must have 1 output, got ",
                                        ctx->num_outputs()));

    string key;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key));

    const string full_key = CreateSendRendezvousKey(key);
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(full_key, &parsed_key_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto poplar_executor = GetPoplarExecutor(device_ordinal_);
    OP_REQUIRES_OK_ASYNC(ctx, poplar_executor.status(), done);
    auto* rendezvous = poplar_executor.ValueOrDie()->GetRendezvous();

    rendezvous->RecvAsync(parsed_key_, Rendezvous::Args{},
                          MakeRecvCallback(ctx, std::move(done)));
  }

 private:
  int device_ordinal_;
  Rendezvous::ParsedKey parsed_key_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaRecvAtHostOp);
};

class XlaSendFromHostOp : public OpKernel {
 public:
  explicit XlaSendFromHostOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));

    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        errors::InvalidArgument("Must have 2 inputs, got ", ctx->num_inputs()));

    string key;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key));

    const string full_key = CreateRecvRendezvousKey(key);
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(full_key, &parsed_key_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto poplar_executor = GetPoplarExecutor(device_ordinal_);
    OP_REQUIRES_OK(ctx, poplar_executor.status());
    auto* rendezvous = poplar_executor.ValueOrDie()->GetRendezvous();

    const Tensor& tensor = ctx->input(0);
    rendezvous->Send(parsed_key_, Rendezvous::Args{}, tensor,
                     /*is_dead=*/false);
  }

 private:
  int device_ordinal_;
  Rendezvous::ParsedKey parsed_key_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaSendFromHostOp);
};

REGISTER_XLA_OP(Name("XlaHostCompute").Device(DEVICE_IPU_XLA_JIT),
                XlaHostComputeOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRecvAtHost").Device(DEVICE_CPU),
                        XlaRecvAtHostOp);

REGISTER_KERNEL_BUILDER(Name("_XlaSendFromHost").Device(DEVICE_CPU),
                        XlaSendFromHostOp);

}  // namespace tensorflow
