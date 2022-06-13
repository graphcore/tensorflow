/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_STREAM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_STREAM_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

class PopItStream : public se::internal::StreamInterface {};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_STREAM_H_
