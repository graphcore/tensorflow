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

#define OP_REQUIRES_OK_WITH_HANDLER(CTX, HANDLER, ...)       \
  do {                                                       \
    ::tensorflow::Status _s(__VA_ARGS__);                    \
    if (!TF_PREDICT_TRUE(_s.ok())) {                         \
      HANDLER();                                             \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return;                                                \
    }                                                        \
  } while (0)

namespace tensorflow {

namespace {
uint64 DeviceIncarnation(int device_ordinal, int replica) {
  return (device_ordinal << 5) | replica;
}
}  // namespace

template <typename T>
class IpuHostEmbeddingOp : public AsyncOpKernel {
 public:
  explicit IpuHostEmbeddingOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), device_ordinal_(0), reader_count_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_id", &embedding_id_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replication_factor", &replication_factor_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lookup_count", &lookup_count_init_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_count", &update_count_init_));

    const std::vector<std::string> names = {"lookup_indices",
                                            "lookup_activations",
                                            "update_indices", "update_grads"};
    const bool to_hosts[] = {true, false, true, true};

    const std::string cpu_device_name = "/device:CPU:0";
    const std::string ipu_device_name =
        "/device:IPU:" + std::to_string(device_ordinal_);

    // Create rendezvous keys for each replica
    for (int replica = 0; replica < replication_factor_; ++replica) {
      for (int i = 0; i < names.size(); ++i) {
        auto& name = names[i];
        if (to_hosts[i]) {
          auto key = Rendezvous::CreateKey(
              ipu_device_name, DeviceIncarnation(device_ordinal_, replica),
              cpu_device_name, embedding_id_ + "_" + name, {0, 0});
          OP_REQUIRES_OK(
              ctx, Rendezvous::ParseKey(key, &rendezvous_keys_[replica][name]));
        } else {
          auto key = Rendezvous::CreateKey(
              cpu_device_name, DeviceIncarnation(device_ordinal_, replica),
              ipu_device_name, embedding_id_ + "_" + name, {0, 0});
          OP_REQUIRES_OK(
              ctx, Rendezvous::ParseKey(key, &rendezvous_keys_[replica][name]));
        }
      }
    }
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    auto rendezvous = poplar_executor->GetRendezvous();

    Tensor inp = ctx->mutable_input(0, true);

    lookup_count_.store(lookup_count_init_);
    update_count_.store(update_count_init_);

    auto done_wrapper = [this, done]() mutable {
      if (lookup_count_ == 0 && update_count_ == 0) {
        done();
      }
    };

    for (int replica = 0; replica < replication_factor_; ++replica) {
      rendezvous->RecvAsync(
          rendezvous_keys_[replica]["lookup_indices"], {},
          CreateIndexLookupRecvCallback(
              ctx, rendezvous, rendezvous_keys_[replica]["lookup_indices"],
              rendezvous_keys_[replica]["lookup_activations"], inp,
              done_wrapper));

      rendezvous->RecvAsync(
          rendezvous_keys_[replica]["update_indices"], {},
          CreateIndexUpdateRecvCallback(
              ctx, rendezvous, rendezvous_keys_[replica]["update_indices"],
              rendezvous_keys_[replica]["update_grads"], inp, done_wrapper));
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  int device_ordinal_;
  int replication_factor_;
  int lookup_count_init_;
  int update_count_init_;
  std::atomic<int> lookup_count_;
  std::atomic<int> update_count_;
  std::atomic<int> reader_count_;
  std::string embedding_id_;
  std::mutex block;
  std::condition_variable block_cv;
  absl::flat_hash_map<int,
                      absl::flat_hash_map<std::string, Rendezvous::ParsedKey>>
      rendezvous_keys_;

  /**
   * Construct a Rendezvous::DoneCallback which handles incoming indices from
   * the IPU.
   *
   * @param ctx The OpKernelContext.
   * @param rendezvous A pointer to the Poplar Executor Rendezvous instance.
   * @param recv_key The rendezvous key for the host to recv on.
   * @param send_key The rendezvous key for the host to send on.
   * @param inp The embedding tensor.
   *
   * @returns A DoneCallback that implements our embedding lookup.
   */
  Rendezvous::DoneCallback CreateIndexLookupRecvCallback(
      OpKernelContext* ctx, Rendezvous* rendezvous,
      Rendezvous::ParsedKey recv_key, Rendezvous::ParsedKey send_key,
      Tensor inp, AsyncOpKernel::DoneCallback done) {
    return [this, ctx, rendezvous, recv_key, send_key, inp, done](
               const Status& status, const Rendezvous::Args& sender_args,
               const Rendezvous::Args& reciever_args, const Tensor& tensor,
               const bool is_dead) mutable {
      auto error_handler = [this, done]() mutable {
        lookup_count_.store(0);
        block_cv.notify_all();

        done();
      };

      OP_REQUIRES_OK_WITH_HANDLER(ctx, error_handler, status);

      // Get the indices
      int32* indices =
          static_cast<int32*>(tensorflow::DMAHelper::buffer(&tensor)->data());

      // Create a vector for holding the slices.
      std::vector<Tensor> slices(tensor.NumElements());
      {
        // Aquire the access mutex.
        std::unique_lock<std::mutex> lk(block);
        reader_count_.fetch_add(1);
      }

      // For each index, slice the elements.
      for (int64 i = 0; i < tensor.NumElements(); ++i) {
        slices[i] = inp.SubSlice(indices[i]);
      }

      // Concatenate the slices.
      Tensor result;
      OP_REQUIRES_OK_WITH_HANDLER(ctx, error_handler,
                                  tensor::Concat(slices, &result));

      // Send the slices to the device.
      OP_REQUIRES_OK_WITH_HANDLER(
          ctx, error_handler, rendezvous->Send(send_key, {}, result, false));

      // Decrement the lookup count. If it's greater than 0 (check greater than
      // 1), initiate another recv with the same callback.
      if (lookup_count_.fetch_sub(1) > 1) {
        rendezvous->RecvAsync(
            recv_key, {},
            CreateIndexLookupRecvCallback(ctx, rendezvous, recv_key, send_key,
                                          inp, done));
      }

      reader_count_.fetch_sub(1);
      done();

      // Notify the waiter that the lookup_count_ and reader_count_ has changed.
      block_cv.notify_all();
    };
  }

  Rendezvous::DoneCallback CreateIndexUpdateRecvCallback(
      OpKernelContext* ctx, Rendezvous* rendezvous,
      Rendezvous::ParsedKey indices_key, Rendezvous::ParsedKey grad_key,
      Tensor inp, AsyncOpKernel::DoneCallback done) {
    return [this, ctx, rendezvous, indices_key, grad_key, inp, done](
               const Status& status, const Rendezvous::Args& sender_args,
               const Rendezvous::Args& reciever_args, const Tensor& tensor,
               const bool is_dead) mutable {
      auto error_handler = [this, done]() mutable {
        update_count_.store(0);
        block_cv.notify_all();

        done();
      };
      OP_REQUIRES_OK_WITH_HANDLER(ctx, error_handler, status);

      // Start to recv the updated values.
      rendezvous->RecvAsync(
          grad_key, {},
          CreateGradUpdateRecvCallback(ctx, rendezvous, indices_key, grad_key,
                                       inp, tensor, done));
    };
  }

  Rendezvous::DoneCallback CreateGradUpdateRecvCallback(
      OpKernelContext* ctx, Rendezvous* rendezvous,
      Rendezvous::ParsedKey indices_key, Rendezvous::ParsedKey grad_key,
      Tensor inp, Tensor indices, AsyncOpKernel::DoneCallback done) {
    return [this, ctx, rendezvous, indices_key, grad_key, inp, indices, done](
               const Status& status, const Rendezvous::Args& sender_args,
               const Rendezvous::Args& reciever_args, const Tensor& grads,
               const bool is_dead) mutable {
      auto error_handler = [this, done]() mutable {
        update_count_.store(0);
        block_cv.notify_all();

        done();
      };

      OP_REQUIRES_OK_WITH_HANDLER(ctx, error_handler, status);

      // Get the indices
      int32* idxs =
          static_cast<int32*>(tensorflow::DMAHelper::buffer(&indices)->data());
      {
        // Aquire the access mutex.
        // Wait until there are no readers
        std::unique_lock<std::mutex> lk(block);
        auto condition = [this]() -> bool { return (reader_count_ == 0); };
        block_cv.wait(lk, condition);

        for (int64 i = 0; i < indices.NumElements(); ++i) {
          auto slice = inp.SubSlice(idxs[i]);
          auto update = grads.SubSlice(i);

          for (int k = 0; k < slice.NumElements(); ++k) {
            T* slice_ptr =
                static_cast<T*>(tensorflow::DMAHelper::buffer(&slice)->data());
            T* update_ptr =
                static_cast<T*>(tensorflow::DMAHelper::buffer(&update)->data());

            slice_ptr[k] += update_ptr[k];
          }
        }
      }

      // Decrement the update count. If it's greater than 0 (check greater than
      // 1), initiate another recv with the same update indices callback.
      if (update_count_.fetch_sub(1) > 1) {
        rendezvous->RecvAsync(
            indices_key, {},
            CreateIndexUpdateRecvCallback(ctx, rendezvous, indices_key,
                                          grad_key, inp, done));
      }

      done();

      // Notify the waiter that the update_count_ has changed.
      block_cv.notify_all();
    };
  }

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

class IpuDeviceEmbeddingLookupOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit IpuDeviceEmbeddingLookupOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_id", &embedding_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_shape", &embedding_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const xla::XlaOp& indices = ctx->Input(0);
    const DataType dtype = output_type(0);
    TensorShape output_shape = ctx->InputShape(0);

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
                IpuDeviceEmbeddingLookupOp);

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
