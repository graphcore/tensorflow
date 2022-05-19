/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FORWARD_ALLOCATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FORWARD_ALLOCATION_H_

#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class CallGraph;
class HloModule;
class HloComputation;
class HloReachabilityMap;

namespace poplarplugin {

struct ForwardAllocationGraphComparator {
  bool operator()(const HloInstruction* const& lhs,
                  const HloInstruction* const& rhs) const;
};

using ForwardAllocationGraph =
    MetaGraph<HloInstruction*, ForwardAllocationGraphComparator>;

class ForwardAllocation : public HloModulePass {
 public:
  ForwardAllocation(CompilerAnnotations& annotations);

  absl::string_view name() const override { return "forward-allocation"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool CreateForwardAllocationTarget(
      HloReachabilityMap* reachability_map, HloInstruction* source,
      HloInstruction* target, const int64_t input_index,
      HloInstruction* layout_producer, const int64_t layout_output_index,
      const std::vector<HloInstruction*>& other_targets,
      const std::vector<HloInstruction*>& forward_path,
      const std::vector<HloInstruction*>& backward_path);

  StatusOr<bool> FindLayoutSensativeTargets(
      HloComputation* comp, std::set<const HloInstruction*>& ops_with_layout,
      CallGraph* call_graph);

  StatusOr<bool> FindLayoutDependentTargets(HloComputation* comp,
                                            CallGraph* call_graph);

  StatusOr<ForwardAllocationGraph::MetaGraphSet> FindInputs(
      HloComputation* comp, CallGraph* call_graph);

  void FlattenInputs(HloInstruction* inst,
                     ForwardAllocationGraph::MetaGraphSet& deferred_inputs);

  TensorAllocationMap& tensor_allocation_map;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
