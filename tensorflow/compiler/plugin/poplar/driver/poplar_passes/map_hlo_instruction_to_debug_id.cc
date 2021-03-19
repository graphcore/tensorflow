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

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/map_hlo_instruction_to_debug_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace xla {
namespace poplarplugin {

MapHloInstructionToDebugIdPass::MapHloInstructionToDebugIdPass(
    absl::flat_hash_map<const HloInstruction*, std::uint64_t>& map)
    : map_(map) {}

StatusOr<bool> MapHloInstructionToDebugIdPass::Run(HloModule* module) {
  bool result = true;

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  // Consider adding debug info for a HloModule and HloComputation
  // here so all instructions can be shown as part of a Module/Computation

  for (auto comp : module->MakeComputationPostOrder()) {
    for (auto inst : comp->MakeInstructionPostOrder()) {
      // Create the top level XlaOp Debug Context.
      poplar::DebugContext xla_op_debug_context(inst->metadata().op_name());
      XlaOpDebugInfo xla_op_debug_info(xla_op_debug_context, inst->metadata());

      // Create the next level HloInstruction Debug Context.
      poplar::DebugContext hlo_instruction_debug_context(xla_op_debug_info,
                                                         inst->name());
      HloInstructionDebugInfo hlo_instruction_debug_info(
          hlo_instruction_debug_context, inst);

      // We create a single debug context for each hlo instruction
      // that will be written to the debug.cbor file (if the poplar engine
      // option has outputDebugInfo enabled). Then we create mapping from
      // HloInstruction to debug info (id) so that later when we get a
      // HloInstruction in a visitor we can determine the debug info id (create
      // a poplar::DebugNameAndId) and pass that down to the PoplarOpDef to
      // associate all poplar/poplibs calls with this HloInstruction.
      map_.insert({inst, hlo_instruction_debug_info.getId()});
    }
  }

  return result;
}

}  // namespace poplarplugin
}  // namespace xla
