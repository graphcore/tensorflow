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

#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

class RepeatLoopVisitor : public InplaceDeferredVisitor {
 public:
  RepeatLoopVisitor(CompilerResources& res, const DeferredArgRBVectors& inputs,
                    const HloPoplarInplaceDescription& description,
                    const ReallocateInputsInfo& reallocate_inputs_info,
                    const poplar::DebugNameAndId& debug_name_and_id);

  Status HandleDeferredAllocationCall(HloInstruction* inst) override;

  Status FinishDeferedAllocationVisit(HloInstruction* inst) override;

  virtual DriverProgramSequence GetRepeatLoopSequence(
      const HloInstruction* inst);

  const TensorOrRemoteBufferVector& GetLoopState() const;

 protected:
  Status AddSequenceForInstruction(const HloInstruction* inst,
                                   const DriverProgramSequence& seq) override;

  void AddSequenceForAliasingCopy(const HloInstruction* inst,
                                  const DriverProgramSequence& seq) override;

  // Sequence which is executed once before the loop starts executing.
  DriverProgramSequence pre_loop_sequence_;

  // Information used for the resource update (if there is one).
  bool has_resource_update_ = false;
  int64 num_mini_batches_to_accumulate_ = -1;
  DriverProgramSequence tensors_zeroing_sequence_;
  DriverProgramSequence resource_update_sequence_;

 private:
  // The tensors representing the inputs/outputs of the loops (they have to
  // alias).
  TensorOrRemoteBufferVector loop_state_;

  // Track the SR method at points in the loop. These are used
  // to make sure the seed remains consistent as the loop restarts and
  // as we go in/out of the resource update function.
  StochasticRoundingMethod loop_start_sr_method_;
  StochasticRoundingMethod ru_end_sr_method_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_REPEAT_LOOP_VISITOR_H_
