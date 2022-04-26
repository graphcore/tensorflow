/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_OVERLAP_IO_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_OVERLAP_IO_VISITOR_H_

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/visitors/repeat_loop_visitor.h"

namespace xla {
namespace poplarplugin {

class RepeatLoopOverlapIOVisitor : public RepeatLoopVisitor {
 public:
  using RepeatLoopVisitor::RepeatLoopVisitor;

  DriverProgramSequence GetRepeatLoopSequence(
      const HloInstruction* inst) override;

 protected:
  Status AddSequenceForInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  Status AppendSequenceGroupedByInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  Status PrependSequenceGroupedByInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

 private:
  StatusOr<poplar::program::Sequence*> GetSequenceForInstruction(
      const HloInstruction* inst);

  poplar::program::Sequence infeed_sequence_;
  poplar::program::Sequence outfeed_sequence_;
  poplar::program::Sequence io_tile_copy_in_sequence_;
  poplar::program::Sequence io_tile_copy_out_sequence_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_OVERLAP_IO_VISITOR_H_
