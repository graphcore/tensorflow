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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_VISITOR_H_

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

class RepeatLoopVisitor : public InplaceDeferredVisitor {
 public:
  RepeatLoopVisitor(CompilerResources& res, const DeferredArgVectors& inputs,
                    bool reallocate_inputs, const std::string& name);

  Status HandleDeferredAllocationCall(HloInstruction* inst) override;

  Status FinishDeferedAllocationVisit(HloInstruction* inst) override;

  poplar::program::Sequence GetRepeatLoopSequence(const HloInstruction* inst);

  const TensorVector& GetLoopState() const;

 protected:
  StatusOr<poplar::program::Sequence*> GetSequenceForInstruction(
      const HloInstruction* inst);

  poplar::program::Sequence& GetSequenceForAliasingCopy() override;

 private:
  // Sequence which is executed once before the loop starts executing.
  poplar::program::Sequence pre_loop_sequence_;

  // The tensors representing the inputs/outputs of the loops (they have to
  // alias).
  TensorVector loop_state_;

  // Information used for the resource update (if there is one).
  bool has_resource_update_ = false;
  int64 num_mini_batches_to_accumulate_ = -1;
  poplar::program::Sequence tensors_zeroing_sequence_;
  poplar::program::Sequence resource_update_sequence_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_VISITOR_H_
