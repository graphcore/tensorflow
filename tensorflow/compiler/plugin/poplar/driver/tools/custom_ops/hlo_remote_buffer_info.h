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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_REMOTE_BUFFER_INFO_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_REMOTE_BUFFER_INFO_H_

#include <string>
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

namespace xla {
namespace poplarplugin {

struct HloRemoteBufferInfo {
  std::string name;
  int64_t num_merged;
  int64_t merge_offset;
};

}  // namespace poplarplugin
}  // namespace xla

namespace std {
template <>
struct hash<xla::poplarplugin::HloRemoteBufferInfo> {
  size_t operator()(const xla::poplarplugin::HloRemoteBufferInfo& info) const {
    return xla::poplarplugin::hash_util::hash(info.name, info.num_merged,
                                              info.merge_offset);
  }
};
}  // namespace std

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_REMOTE_BUFFER_INFO_H_
