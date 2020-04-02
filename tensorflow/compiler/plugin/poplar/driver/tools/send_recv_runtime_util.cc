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

#include "tensorflow/compiler/plugin/poplar/driver/tools/send_recv_runtime_util.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace xla {
namespace poplarplugin {

namespace {

// Non-owning view of a Poplar tensor buffer. Can be used to make
// a `tensorflow::Tensor` backed by this buffer without copying
// memory. The backing memory must outlive the `Tensor`.
class PoplarTensorBufferView : public tensorflow::TensorBuffer {
 public:
  PoplarTensorBufferView(void* data, std::size_t size)
      : TensorBuffer(data), size_(size) {}

  ~PoplarTensorBufferView() override {}

  std::size_t size() const override { return size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {}

  bool OwnsMemory() const override { return false; }

 private:
  std::size_t size_;
};

tensorflow::TensorShape ConcatenatedShape(const tensorflow::TensorShape& shape,
                                          int64 factor) {
  tensorflow::TensorShape concat_shape(shape);
  CHECK_GT(shape.dims(), 0);
  concat_shape.set_dim(0, shape.dim_size(0) * factor);
  return concat_shape;
}

}  // namespace

std::function<poplar::StreamCallbackHandle(int64)>
SendFromFirstReplicaCallbackCreator(const tensorflow::TensorShape& shape,
                                    tensorflow::DataType type,
                                    tensorflow::Rendezvous::ParsedKey key,
                                    tensorflow::Rendezvous* rendezvous,
                                    int64 num_replicas,
                                    bool can_avoid_buffer_copy) {
  return [=](int64 replica_id) -> poplar::StreamCallbackHandle {
    if (replica_id == 0) {
      if (can_avoid_buffer_copy) {
        // Use a non-owning buffer view to avoid the copy in the callback.
        // Also, maintain a reference to the TensorBuffer in order to
        // prohibit any TensorFlow ops from doing in-place modifications
        // of the buffer (since they will observe the refcount being
        // greater than one), see i.e. OpKernelContext::forward_input.
        using Buffer = tensorflow::core::RefCountPtr<PoplarTensorBufferView>;

        return [rendezvous, key, type, shape,
                buf = Buffer()](void* src) mutable {
          if (!buf) {
            // Allocate the buffer view the first time.
            buf.reset(new PoplarTensorBufferView(
                src, shape.num_elements() * tensorflow::DataTypeSize(type)));
          } else {
            // Should get the same pointer from Poplar every time.
            CHECK_EQ(buf->data(), src);
            // And we should be the only buffer owner here.
            CHECK(buf->RefCountIsOne());
          }

          auto tensor = tensorflow::Tensor(type, shape, buf.get());

          // Sending here increases the refcount until it is consumed.
          rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                           /*is_dead=*/false);
        };
      }

      // Must copy the data in the callback.
      return [rendezvous, key,
              tensor = tensorflow::Tensor(type, shape)](void* src) {
        auto* dst = tensorflow::DMAHelper::buffer(&tensor);

        // We reuse the same tensor every time to avoid allocating in this
        // callback. This should be safe since every Send op must be matched
        // by a corresponding Recv op in the same graph, so the tensor must
        // be consumed before the next execution of the graph. Verify this
        // assumption here by checking that we are the only owner.
        CHECK(dst->RefCountIsOne());
        std::memcpy(dst->data(), src, dst->size());

        // Sending here increases the refcount until it is consumed.
        rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                         /*is_dead=*/false);
      };
    } else {
      // Discard the output from the remaining replicas.
      return [](void*) {};
    }
  };
}

std::function<poplar::StreamCallbackHandle(int64)>
SendConcatenatedCallbackCreator(const tensorflow::TensorShape& shape,
                                tensorflow::DataType type,
                                tensorflow::Rendezvous::ParsedKey key,
                                tensorflow::Rendezvous* rendezvous,
                                int64 num_replicas) {
  // We create one shared tensor and reuse it every time to avoid allocating
  // in the callback. This should be safe since every Send op must be
  // matched by a corresponding Recv op in the same graph, so the tensor
  // must be consumed before the next execution of the graph. We verify this
  // assumption before sending by checking that we are the only owner.
  // All the lambdas must capture the shared_ptr to make sure it is alive.
  auto tensor = std::make_shared<tensorflow::Tensor>(
      type, ConcatenatedShape(shape, num_replicas));

  const int64 num_bytes_per_replica =
      shape.num_elements() * tensorflow::DataTypeSize(type);

  auto* tensor_buf = tensorflow::DMAHelper::buffer(tensor.get());
  CHECK_EQ(tensor_buf->size(), num_bytes_per_replica * num_replicas);
  CHECK(tensor_buf->RefCountIsOne());

  // Keep one additional reference for each callback.
  for (int64 i = 0; i < num_replicas; ++i) {
    tensor_buf->Ref();
  }

  // Since the callbacks for the replicas are invoked by Poplar sequentially
  // in replica index order, we can send the tensor from the last replica.
  // Verify this assumption by slightly abusing the refcount to count the
  // number of callbacks invoked.
  return [=](int64 replica_id) -> poplar::StreamCallbackHandle {
    char* dst = tensor_buf->base<char>() + replica_id * num_bytes_per_replica;
    if (replica_id < num_replicas - 1) {
      return [tensor, tensor_buf, dst, num_bytes_per_replica](void* src) {
        std::memcpy(dst, src, num_bytes_per_replica);
        tensor_buf->Unref();
      };
    } else {
      return [tensor, tensor_buf, dst, num_bytes_per_replica, num_replicas,
              rendezvous, key](void* src) {
        std::memcpy(dst, src, num_bytes_per_replica);
        tensor_buf->Unref();

        // Check that this was the last callback to be invoked, and that
        // the tensor was consumed from the previous iteration.
        CHECK(tensor_buf->RefCountIsOne());

        // Sending here increases the refcount until it is consumed.
        rendezvous->Send(key, tensorflow::Rendezvous::Args{}, *tensor,
                         /*is_dead=*/false);

        // Prepare the refcount for the next iteration.
        for (int64 i = 0; i < num_replicas; ++i) {
          tensor_buf->Ref();
        }
      };
    }
  };
}

bool CanPoplarSendBuffersOverlap(const poplar::OptionFlags& flags,
                                 const IpuOptions& options) {
  // Check if there is an environment variable override.
  char* env_flags = std::getenv("POPLAR_ENGINE_OPTIONS");
  if (env_flags) {
    if (std::string(env_flags).find("streamBufferOverlap") !=
        std::string::npos) {
      // Don't bother parsing the value, assume the worst.
      return true;
    }
  }

  // Otherwise, check the option we passed in.
  const auto stream_buffer_overlap =
      absl::c_find_if(flags, [](const poplar::OptionFlags::OptionFlag& flag) {
        return flag.first == "exchange.streamBufferOverlap";
      });

  if (stream_buffer_overlap == flags.end()) {
    return true;
  }

  if (stream_buffer_overlap->second == "none") {
    return false;
  }

  if (stream_buffer_overlap->second == "hostRearrangeOnly") {
    // Can overlap if we are doing rearrangement on the host.
    return options.speed_size_config().always_rearrange_copies_on_the_host();
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
