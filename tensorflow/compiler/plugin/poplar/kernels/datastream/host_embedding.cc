/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"

namespace tensorflow {

template <typename T>
class IpuHostEmbeddingOp : public AsyncOpKernel {
 public:
  explicit IpuHostEmbeddingOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_id", &embedding_id_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replication_factor", &replication_factor_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lookup_count", &lookup_count_init_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_count", &update_count_init_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    ctx->forward_ref_input_to_ref_output(0, 0);

    // If we are either using synthetica data or never performa a lookup/update,
    // then immediately complete the async op.
    if (xla::poplarplugin::UseSyntheticData() ||
        (lookup_count_init_ + update_count_init_ == 0)) {
      done();
    } else {
      auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
      OP_REQUIRES(ctx, platform.ok(), platform.status());
      auto* p = static_cast<xla::poplarplugin::PoplarPlatform*>(
          platform.ValueOrDie());
      auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
      auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
          stream_executor->implementation());

      Tensor inp = ctx->mutable_input(0, true);

      poplar_executor->RegisterHostEmbedding(
          embedding_id_, absl::make_unique<HostEmbeddingImpl>(
                             inp, replication_factor_, lookup_count_init_,
                             update_count_init_, done));
    }
  }

 private:
  int device_ordinal_;
  int replication_factor_;
  int lookup_count_init_;
  int update_count_init_;
  std::string embedding_id_;

  class HostEmbeddingImpl
      : public xla::poplarplugin::PoplarExecutor::HostEmbeddingInterface<T> {
   public:
    HostEmbeddingImpl(Tensor embedding, int replication_factor,
                      int lookup_count, int update_count,
                      AsyncOpKernel::DoneCallback done)
        : encoding_width_(embedding.dim_size(1)),
          access_count_(replication_factor * (lookup_count + update_count)),
          embedding_(std::move(embedding)),
          done_(std::move(done)),
          lookup_indices_(replication_factor),
          update_indices_(replication_factor) {
      embedding_rows_.reserve(embedding_.dim_size(0));

      for (std::size_t i = 0; i < embedding_.dim_size(0); ++i) {
        auto slice = embedding_.SubSlice(i);
        auto buffer = tensorflow::DMAHelper::buffer(&slice);

        embedding_rows_.push_back(buffer->base<T>());
      }
    }

    ~HostEmbeddingImpl() { done_(); }

    Status EnqueueLookupIndices(int replica, const int* indices,
                                int index_count) {
      lookup_indices_[replica].resize(index_count);

      std::memcpy(lookup_indices_[replica].data(), indices,
                  index_count * sizeof(int));

      return Status::OK();
    }

    Status DequeueLookupActivations(int replica, T* destination) {
      for (std::size_t i = 0; i < lookup_indices_[replica].size(); ++i) {
        std::memcpy(destination + i * encoding_width_,
                    embedding_rows_[lookup_indices_[replica][i]],
                    encoding_width_ * sizeof(T));
      }

      access_count_.fetch_sub(1);

      return Status::OK();
    }

    Status EnqueueUpdateIndices(int replica, const int* indices,
                                int index_count) {
      update_indices_[replica].resize(index_count);

      std::memcpy(update_indices_[replica].data(), indices,
                  index_count * sizeof(int));

      return Status::OK();
    }

    Status EnqueueUpdateGrads(int replica, const T* grads) {
      std::size_t index_count = update_indices_[replica].size();

      for (std::size_t i = 0; i < index_count; ++i) {
        const T* src_ptr = grads + (encoding_width_ * i);

        std::transform(src_ptr, src_ptr + encoding_width_,
                       embedding_rows_[update_indices_[replica][i]],
                       embedding_rows_[update_indices_[replica][i]],
                       std::plus<T>{});
      }

      access_count_.fetch_sub(1);

      return Status::OK();
    }

    bool Done() const { return access_count_ <= 0; }

   private:
    int encoding_width_;

    std::atomic<int> access_count_;

    Tensor embedding_;
    std::vector<T*> embedding_rows_;
    AsyncOpKernel::DoneCallback done_;

    std::vector<std::vector<int>> lookup_indices_;
    std::vector<std::vector<int>> update_indices_;
  };

  TF_DISALLOW_COPY_AND_ASSIGN(IpuHostEmbeddingOp);
};

#define REGISTER_HOST_EMBEDDING_KERNEL(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("IpuHostEmbedding").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      IpuHostEmbeddingOp<T>);

TF_CALL_half(REGISTER_HOST_EMBEDDING_KERNEL);
TF_CALL_float(REGISTER_HOST_EMBEDDING_KERNEL);
TF_CALL_int32(REGISTER_HOST_EMBEDDING_KERNEL);
TF_CALL_uint32(REGISTER_HOST_EMBEDDING_KERNEL);

template <int IndicesPosition>
class IpuDeviceEmbeddingLookupOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit IpuDeviceEmbeddingLookupOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_id", &embedding_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_shape", &embedding_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const xla::XlaOp& indices = ctx->Input(IndicesPosition);
    const DataType dtype = output_type(0);
    TensorShape output_shape = ctx->InputShape(IndicesPosition);

    output_shape.AppendShape(embedding_shape_);
    output_shape.RemoveDim(1);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, output_shape, &xla_shape));

    attribute_map_.AddAttribute("embedding_id", embedding_id_);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::HostEmbeddingLookup),
                        {indices}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  std::string embedding_id_;
  TensorShape embedding_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuDeviceEmbeddingLookupOp);
};

REGISTER_XLA_OP(Name("IpuDeviceEmbeddingLookup").Device(DEVICE_IPU_XLA_JIT),
                IpuDeviceEmbeddingLookupOp<0>);

REGISTER_XLA_OP(
    Name("IpuDeviceEmbeddingLookupTrainable").Device(DEVICE_IPU_XLA_JIT),
    IpuDeviceEmbeddingLookupOp<1>);

class IpuDeviceEmbeddingUpdateAddOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit IpuDeviceEmbeddingUpdateAddOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_id", &embedding_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_shape", &embedding_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const xla::XlaOp& grads = ctx->Input(0);
    const xla::XlaOp& indices = ctx->Input(1);

    xla::Shape xla_shape = xla::ShapeUtil::MakeTokenShape();

    attribute_map_.AddAttribute("embedding_id", embedding_id_);

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp output = xla::CustomCall(
        b, PoplarOp_Name(PoplarOp::HostEmbeddingUpdate), {grads, indices},
        xla_shape, attribute_map_.Serialise());
  }

 private:
  std::string embedding_id_;
  TensorShape embedding_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuDeviceEmbeddingUpdateAddOp);
};

REGISTER_XLA_OP(Name("IpuDeviceEmbeddingUpdateAdd").Device(DEVICE_IPU_XLA_JIT),
                IpuDeviceEmbeddingUpdateAddOp);
}  // namespace tensorflow
