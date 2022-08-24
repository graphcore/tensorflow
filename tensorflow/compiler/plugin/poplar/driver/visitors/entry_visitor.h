/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_ENTRY_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_ENTRY_VISITOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

/*
 * This visitor handles inputs and outputs of the entry computation in a module.
 */
class EntryVisitor : public DeferredVisitor {
 public:
  EntryVisitor(CompilerResources& resources, const HloComputation* comp);

  const DriverProgramSequence GetHostToDevice() const;
  const DriverProgramSequence GetDeviceToHost() const;

  // Get the sequence generated by this visitor. Prepends the sequence with
  // computation for initializing the execution counters to zero.
  const DriverProgramSequence GetSequenceAndInitializeCounters();

 protected:
  Status AddSequenceForInstruction(const HloInstruction* inst,
                                   const DriverProgramSequence& seq) override;

  StatusOr<DriverTensor> PostProcessParameterAllocation(
      TensorLocation location, const Shape& shape,
      DriverProgramSequence& sequence, DriverTensor tensor,
      const poplar::DebugNameAndId& debug_name_and_id) override;

  Status FinishDeferedAllocationVisit(HloInstruction* root) override;

  Status PreProcessParameter(HloInstruction* parameter) override;

 private:
  Status StreamOutputs(HloInstruction* inst, uint64 start_idx,
                       TensorVector outputs);

  DriverProgramSequence host_to_device;
  DriverProgramSequence device_to_host;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
