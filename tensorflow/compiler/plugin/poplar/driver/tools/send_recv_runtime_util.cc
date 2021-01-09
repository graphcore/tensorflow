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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

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
        return [rendezvous, key, type, shape](void* src) mutable {
          // Create a non-owning view of the source data, avoiding copying.
          auto buffer = tensorflow::core::RefCountPtr<PoplarTensorBufferView>(
              new PoplarTensorBufferView(
                  src, shape.num_elements() * tensorflow::DataTypeSize(type)));
          auto tensor = tensorflow::Tensor(type, shape, buffer.get());

          rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                           /*is_dead=*/false);
        };
      } else {
        // In this case we must copy the data in the callback.
        return [rendezvous, key, type, shape](void* src) {
          auto tensor = tensorflow::Tensor(type, shape);
          auto* dst = tensorflow::DMAHelper::buffer(&tensor);
          std::memcpy(dst->data(), src, dst->size());

          rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                           /*is_dead=*/false);
        };
      }
    } else {
      // Discard the output from the remaining replicas.
      return [](void*) {};
    }
  };
}

bool CanPoplarSendBuffersOverlap(const poplar::OptionFlags& flags,
                                 const IpuOptions& options) {
  // Check if there is an environment variable override.
  if (GetPoplarEngineOption("exchange.streamBufferOverlap").has_value()) {
    return true;
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
