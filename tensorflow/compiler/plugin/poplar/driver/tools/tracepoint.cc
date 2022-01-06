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
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"

namespace xla {
namespace poplarplugin {

// Initialize static class member
pvti::TraceChannel TensorflowPoplarPluginTracepoint::trace_tensorflow = {
    "tensorflow_plugin"};

void TensorflowPoplarPluginTracepoint::BeginTrace(
    const absl::string_view trace_label) {
  pvti::Tracepoint::begin(&trace_tensorflow,
                          static_cast<std::string>(trace_label));
}

void TensorflowPoplarPluginTracepoint::EndTrace(
    const absl::string_view trace_label) {
  pvti::Tracepoint::end(&trace_tensorflow,
                        static_cast<std::string>(trace_label));
}

}  // namespace poplarplugin
}  // namespace xla
