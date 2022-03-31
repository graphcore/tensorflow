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

#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

class ExtendedProgramCopy;
class ExtendedProgramSync;

// Wrapper class to abstract migration from poplar to snap
class ExtendedProgramSequence {
 public:
  ExtendedProgramSequence(poplar::Graph&) : sequence_() {}  // NOLINT
  explicit ExtendedProgramSequence(poplar::Graph&,
                                   const poplar::DebugContext& debugContext)
      : sequence_(debugContext) {}

  operator poplar::program::Sequence&() { return getPoplarSequence(); }
  operator const poplar::program::Sequence&() const {
    return getPoplarSequence();
  }

  void add(const ExtendedProgramSequence& p) {
    getPoplarSequence().add((poplar::program::Program&)p);
  }
  void add(const ExtendedProgramCopy& p) {
    getPoplarSequence().add((poplar::program::Program&)p);
  }
  void add(const ExtendedProgramSync& p) {
    getPoplarSequence().add((poplar::program::Program&)p);
  }

  void add(const poplar::program::Program& p) { getPoplarSequence().add(p); }

  poplar::program::Sequence& getPoplarSequence() { return sequence_; }
  const poplar::program::Sequence& getPoplarSequence() const {
    return sequence_;
  }

 private:
  poplar::program::Sequence sequence_;
};

// Wrapper class to abstract migration from poplar to snap
class ExtendedProgramCopy {
 public:
  ExtendedProgramCopy(poplar::Tensor src, const poplar::DataStream& stream,
                      bool optimiseMemory = false,
                      const poplar::DebugContext& debugContext = {})
      : copy_(src, stream, optimiseMemory, debugContext) {}

  ExtendedProgramCopy(const poplar::DataStream& stream, poplar::Tensor dst,
                      bool optimiseMemory = false,
                      const poplar::DebugContext& debugContext = {})
      : copy_(stream, dst, optimiseMemory, debugContext) {}

  ExtendedProgramCopy(poplar::Tensor src, poplar::Tensor dst,
                      bool dontOutline = false,
                      const poplar::DebugContext& debugContext = {})
      : copy_(src, dst, dontOutline, debugContext) {}

  operator poplar::program::Program&() { return copy_; }
  operator const poplar::program::Program&() const { return copy_; }

 private:
  poplar::program::Copy copy_;
};

// Wrapper class to abstract migration from poplar to snap
class ExtendedProgramSync {
 public:
  ExtendedProgramSync(poplar::Graph&, poplar::SyncType type,
                      const poplar::DebugContext& debugContext = {})
      : sync_(type, debugContext) {}

  operator poplar::program::Program&() { return sync_; }
  operator const poplar::program::Program&() const { return sync_; }

 private:
  poplar::program::Sync sync_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXTENDED_PROGRAM_H_
