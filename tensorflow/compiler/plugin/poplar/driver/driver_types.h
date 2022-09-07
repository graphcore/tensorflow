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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_

#include <utility>
#include <vector>

#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

using DriverGraph = poplar::Graph;
using DriverTensor = poplar::Tensor;
using DriverDataStream = poplar::DataStream;
using DriverRemoteBuffer = poplar::RemoteBuffer;

using DriverProgram = poplar::program::Program;
using DriverProgramSequence = poplar::program::Sequence;
using DriverProgramCopy = poplar::program::Copy;
using DriverProgramSync = poplar::program::Sync;
using DriverProgramRepeat = poplar::program::Repeat;
using DriverProgramCall = poplar::program::Call;
using DriverProgramWriteUndef = poplar::program::WriteUndef;

using DriverFunction = poplar::Function;

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_DRIVER_TYPES_H_
