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

#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>

namespace xla {
namespace poplarplugin {

using ExtendedProgram = snap::program::Program;
using ExtendedProgramCopy = snap::program::Copy;
using ExtendedProgramSync = snap::program::Sync;
using ExtendedProgramRepeat = snap::program::Repeat;
using ExtendedProgramCall = snap::program::Call;
using ExtendedProgramWriteUndef = snap::program::WriteUndef;

// Wrapper class to abstract migration from poplar to snap
class ExtendedProgramSequence : public snap::program::Sequence {
 public:
  ExtendedProgramSequence(snap::Graph& graph,
                          const poplar::DebugContext& debugContext = {})
      : snap::program::Sequence(debugContext, graph) {}
  ExtendedProgramSequence(poplar::ArrayRef<snap::program::Program> programs,
                          snap::Graph& graph,
                          const poplar::DebugContext& debugContext = {})
      : snap::program::Sequence(programs, debugContext, graph) {}

  operator poplar::program::Sequence&() { return getPoplarSequence(); }
  operator const poplar::program::Sequence&() const {
    return getPoplarSequence();
  }

  void add(const snap::program::Program& p) { snap::program::Sequence::add(p); }
  void add(const poplar::program::Program& p) { getPoplarSequence().add(p); }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_PROGRAM_H_
