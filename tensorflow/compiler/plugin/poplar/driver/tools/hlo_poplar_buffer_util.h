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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_UTIL_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

class HloInstruction;

namespace poplarplugin {
// Common functions for describing uses/aliasing/buffers of inputs/outputs for
// instructions.

// Returns descriptions where there is no aliasing between inputs and outputs.
HloPoplarUseDescriptions UseDescriptionsNoInputOutputAlias();

// Returns descriptions where all the inputs/outputs are non-tuples, and the
// output aliases the first 'num_operands' operands.
HloPoplarUseDescriptions UseDescriptionsSimpleNoTupleAliasing(
    const HloInstruction* inst, int64_t num_operands, BufferUseKind kind);

// Same as above, however fixes num_operands=1, which is common for
// elementwise-like operations.
HloPoplarUseDescriptions UseDescriptionsSimpleNoTuple0thOperandAliasing(
    const HloInstruction* inst,
    BufferUseKind kind = BufferUseKind::USE_ALIAS_READ_WRITE);

// Returns descriptions where all the buffers from the first 'num_operands' are
// forwarded as output buffers at corresponding shape indices.
// Requires the shape of inst to match the shape obtained by combining the
// shapes of the 'num_operands' operands.
HloPoplarUseDescriptions UseDescriptionsForwardsBuffers(
    const HloInstruction* inst, int64_t num_operands, BufferUseKind kind);

// Returns descriptions where no allocations were made.
HloPoplarBufferDescriptions BufferDescriptionsNoAllocations();

// Returns buffer allocation descriptions for all the outputs.
HloPoplarBufferDescriptions BufferDescriptionsAllocatesAllOutputs(
    const HloInstruction* inst,
    BufferLocality locality = BufferLocality::kDeviceMemory);

// Returns buffer allocation descriptions for all the output indicies which
// do not have an aliased buffer assigned to them.
HloPoplarBufferDescriptions BufferDescriptionsAllocatesAllUnaliasedBuffers(
    const HloInstruction* inst, const HloPoplarUseDescriptions& descriptions,
    BufferLocality locality = BufferLocality::kDeviceMemory);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_UTIL_H_
