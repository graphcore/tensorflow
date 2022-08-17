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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_PROGRAM_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_PROGRAM_H_

#include <utility>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

using ExtendedProgram = poplar::program::Program;
using ExtendedProgramCopy = poplar::program::Copy;
using ExtendedProgramSync = poplar::program::Sync;
using ExtendedProgramRepeat = poplar::program::Repeat;
using ExtendedProgramCall = poplar::program::Call;
using ExtendedProgramWriteUndef = poplar::program::WriteUndef;

// Wrapper class for poplar (will be removed in T67791)
class ExtendedProgramSequence : public poplar::program::Sequence {
 public:
  ExtendedProgramSequence(poplar::Graph& graph,
                          const poplar::DebugContext& debugContext = {})
      : poplar::program::Sequence(debugContext) {}
  ExtendedProgramSequence(
      std::initializer_list<poplar::program::Program> programs,
      poplar::Graph& graph, const poplar::DebugContext& debugContext = {})
      : poplar::program::Sequence(programs, debugContext) {}

  void add(const poplar::program::Program& p) {
    poplar::program::Sequence::add(p);
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_PROGRAM_H_
