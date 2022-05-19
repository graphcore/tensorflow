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

// Analysis for determining the possible set of values for all positions
// (instructions and ShapeIndexes) in the HLO module. Analysis is module-scoped
// tracking values across computation boundaries.
// This class is based on:
// tensorflow/compiler/xla/service/hlo_dataflow_analysis.h

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_DATAFLOW_ANALYSIS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_DATAFLOW_ANALYSIS_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;

// Analysis which identifies all Poplar buffers and their uses in an HLO module.
class HloPoplarDataflowAnalysis {
 public:
  // Run dataflow analysis on the given module. Requires the module to be
  // flattened.
  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloComputation* entry, const CallGraph& call_graph,
      const CompilerAnnotations* annotations = nullptr);

  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloComputation* entry,
      const CompilerAnnotations* annotations = nullptr);

  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloModule* module, const CompilerAnnotations& annotations);

  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloModule* module, const CompilerAnnotations& annotations,
      const CallGraph& call_graph);

  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloModule* module,
      const CompilerAnnotations* annotations = nullptr);

  static StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>> Run(
      const HloModule* module, const CompilerAnnotations* annotations,
      const CallGraph& call_graph);

  // Returns a new HloPoplarBuffer defined at the given instruction and shape
  // index.
  HloPoplarBuffer* NewHloPoplarBuffer(HloInstruction* instruction,
                                      const ShapeIndex& index,
                                      BufferLocality locality);

  // Returns true if 'instruction' defines a buffer at the given shape index
  // of its output.
  bool BufferIsDefinedAt(const HloInstruction* instruction,
                         const ShapeIndex& index = {}) const;

  // Return the HloPoplarBuffer defined by 'instruction' at the given shape
  // index of its output.
  //
  // Precondition: BufferIsDefinedAt is true for this instruction and index.
  const HloPoplarBuffer& GetBufferDefinedAt(const HloInstruction* instruction,
                                            const ShapeIndex& index = {}) const;
  HloPoplarBuffer& GetBufferDefinedAt(const HloInstruction* instruction,
                                      const ShapeIndex& index = {});

  // Sets the buffer set output for an instruction.
  void SetInstructionBufferSetOutput(const HloInstruction* instruction,
                                     const ShapeIndex& index,
                                     const HloPoplarBufferSet& buffer_set);

  // Sets the whole instruction buffer set for an instruction.
  void SetInstructionBufferSet(
      const HloInstruction* instruction,
      const InstructionPoplarBufferSet& instruction_buffer_set);

  // Return the InstructionPoplarBufferSet for the given instruction.
  const InstructionPoplarBufferSet& GetInstructionBufferSet(
      const HloInstruction* instruction) const;
  InstructionPoplarBufferSet& GetInstructionBufferSet(
      const HloInstruction* instruction);

  const InstructionBufferSets& GetInstructionBufferSets() const {
    return buffer_sets_;
  }

  // Return the HloPoplarBufferSet for the given instruction at the given index
  // or the given position.
  const HloPoplarBufferSet& GetBufferSet(const HloInstruction* instruction,
                                         const ShapeIndex& index = {}) const;
  const HloPoplarBufferSet& GetBufferSet(
      const HloPoplarPosition& position) const;
  HloPoplarBufferSet& GetBufferSet(const HloPoplarPosition& position);
  HloPoplarBufferSet& GetBufferSet(const HloInstruction* instruction,
                                   const ShapeIndex& index = {});

  // Return the unique buffer in the HloPoplarBufferSet at the given instruction
  // and shape index. CHECKs if the buffer set does not contain a exactly one
  // buffer.
  const HloPoplarBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                                           const ShapeIndex& index = {}) const {
    return GetBufferSet(instruction, index).GetUniqueBuffer();
  }
  HloPoplarBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                                     const ShapeIndex& index = {}) {
    return GetBuffer(GetBufferSet(instruction, index).GetUniqueBuffer().id());
  }

  // Return the HloPoplarBuffer with the given Id.
  const HloPoplarBuffer& GetBuffer(HloPoplarBuffer::Id buffer_id) const;
  HloPoplarBuffer& GetBuffer(HloPoplarBuffer::Id buffer_id);

  // Return the total number of HloPoplarBuffers.
  int64_t buffer_count() const { return buffers_.size(); }

  std::string ToString() const;

 private:
  explicit HloPoplarDataflowAnalysis(const HloModule* module);

  // Constructs and initializes the InstructionPoplarBufferSets of all
  // instructions and then propagates them.
  Status InitializeAndPropagate(const CompilerAnnotations* annotations,
                                const HloComputation* entry);

  const HloModule* module_;

  // The map of all HloPoplarBuffers in the module. We pass around pointers to
  // the mapped HloPoplarBuffers, so the underlying container must keep them
  // valid despite mutations touching other map entries.
  std::map<HloPoplarBuffer::Id, HloPoplarBuffer> buffers_;

  // A map from instruction to InstructionPoplarBufferSet.
  std::unordered_map<const HloInstruction*, InstructionPoplarBufferSet>
      buffer_sets_;

  // The Id to use for the next HloPoplarBuffer.
  HloPoplarBuffer::Id next_buffer_id_ = 0;

  absl::flat_hash_set<const HloComputation*> visited_computations_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_DATAFLOW_ANALYSIS_H_
