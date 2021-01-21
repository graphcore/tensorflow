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

std::function<poplar::StreamCallbackHandle(int64)>
SendFromFirstReplicaCallbackCreator(const tensorflow::TensorShape& shape,
                                    tensorflow::DataType type,
                                    tensorflow::Rendezvous::ParsedKey key,
                                    tensorflow::Rendezvous* rendezvous,
                                    int64 num_replicas) {
  return [=](int64 replica_id) -> poplar::StreamCallbackHandle {
    if (replica_id == 0) {
      return [rendezvous, key, type, shape](void* src) {
        auto tensor = tensorflow::Tensor(type, shape);
        auto* dst = tensorflow::DMAHelper::buffer(&tensor);
        std::memcpy(dst->data(), src, dst->size());

        rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                         /*is_dead=*/false);
      };
    } else {
      // Discard the output from the remaining replicas.
      return [](void*) {};
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
