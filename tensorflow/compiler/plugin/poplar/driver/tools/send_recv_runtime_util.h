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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SEND_RECV_RUNTIME_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SEND_RECV_RUNTIME_UTIL_H_

#include <poplar/OptionFlags.hpp>
#include <poplar/StreamCallback.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/rendezvous.h"

namespace xla {
namespace poplarplugin {

std::function<poplar::StreamCallbackHandle(int64)>
SendFromFirstReplicaCallbackCreator(const tensorflow::TensorShape& shape,
                                    tensorflow::DataType type,
                                    tensorflow::Rendezvous::ParsedKey key,
                                    tensorflow::Rendezvous* rendezvous,
                                    int64 num_replicas,
                                    bool can_avoid_buffer_copy);

std::function<poplar::StreamCallbackHandle(int64)>
SendConcatenatedCallbackCreator(const tensorflow::TensorShape& shape,
                                tensorflow::DataType type,
                                tensorflow::Rendezvous::ParsedKey key,
                                tensorflow::Rendezvous* rendezvous,
                                int64 num_replicas);

bool CanPoplarSendBuffersOverlap(const poplar::OptionFlags& flags,
                                 const IpuOptions& options);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SEND_RECV_RUNTIME_UTIL_H_
