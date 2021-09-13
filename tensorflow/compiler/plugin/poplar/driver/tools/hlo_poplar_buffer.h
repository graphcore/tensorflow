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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/alias_info.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

// This file is based on tensorflow/compiler/xla/service/hlo_value.h but adopted
// to meet Poplar specific needs (for example the static allocations).

namespace xla {
namespace poplarplugin {

// Abstraction which identifies a specific point in the XLA graph. An
// HloPoplarPosition specifies a ShapeIndex within the output of a specific
// instruction.
struct HloPoplarPosition {
  HloInstruction* instruction;
  ShapeIndex index;

  // Returns the shape at this position.
  const Shape& shape() const;

  std::string ToString() const;

  bool operator==(const HloPoplarPosition& other) const;
  bool operator!=(const HloPoplarPosition& other) const;
  bool operator<(const HloPoplarPosition& other) const;
};

std::ostream& operator<<(std::ostream& out, const HloPoplarPosition& position);

// Class used to describe a single buffer alias.
class HloPoplarUseDescription {
 public:
  HloPoplarUseDescription(int64 operand_number, const ShapeIndex& operand_index,
                          const ShapeIndex& output_index, BufferUseKind kind);

  // The operand number in which the buffer appears.
  int64 operand_number() const { return operand_number_; }

  // The shape index within the operand in which the buffer appears.
  const ShapeIndex& operand_index() const { return operand_index_; }

  // The shape index within the instruction in which the buffer appears as an
  // output.
  const ShapeIndex& output_index() const { return output_index_; }

  // Get what kind of usage this is.
  BufferUseKind kind() const { return kind_; }

  // Convert to protobuf version of this class.
  PoplarUseDescription ToProto() const;

  // Get the class instance from the protobuf.
  static HloPoplarUseDescription FromProto(const PoplarUseDescription& proto);

  std::string ToString() const;

  bool operator==(const HloPoplarUseDescription& other) const;
  bool operator!=(const HloPoplarUseDescription& other) const;

 private:
  const int64 operand_number_;
  const ShapeIndex operand_index_;
  const ShapeIndex output_index_;
  const BufferUseKind kind_;
};

using HloPoplarUseDescriptions = std::vector<HloPoplarUseDescription>;

// Base class for defining a single use of a buffer.
class HloPoplarUse {
 public:
  // Instruction at which the buffer is used.
  HloInstruction* instruction() const { return instruction_; }

  // The operand number in which the buffer appears.
  int64 operand_number() const { return operand_number_; }

  // The shape index within the operand in which the buffer appears.
  const ShapeIndex& operand_index() const { return operand_index_; }

  // Get what kind of usage this is.
  BufferUseKind kind() const { return kind_; }

  virtual std::string ToString() const = 0;

 protected:
  HloPoplarUse(HloInstruction* instruction, int64 operand_number,
               const ShapeIndex& operand_index, BufferUseKind kind);

 private:
  HloInstruction* instruction_;
  const int64 operand_number_;
  const ShapeIndex operand_index_;
  const BufferUseKind kind_;
};

std::ostream& operator<<(std::ostream& out, const HloPoplarUse& use);

// A class for defining a single use of a buffer which is not aliased by any
// output.
class HloPoplarNoAliasUse : public HloPoplarUse {
 public:
  HloPoplarNoAliasUse(HloInstruction* instruction, int64 operand_number,
                      const ShapeIndex& operand_index);

  std::string ToString() const override;
};

// A base class for defining a single use of a buffer which is aliased by some
// outputs.
class HloPoplarAliasUseBase : public HloPoplarUse {
 public:
  // The shape index within the instruction in which the buffer appears as an
  // output.
  const std::vector<ShapeIndex>& output_indices() const {
    return output_indices_;
  }

  std::string ToString() const override;

 protected:
  HloPoplarAliasUseBase(HloInstruction* instruction, int64 operand_number,
                        const ShapeIndex& operand_index,
                        const std::vector<ShapeIndex> output_indices,
                        BufferUseKind kind);

 private:
  const std::vector<ShapeIndex> output_indices_;
};

// A class for defining a single use of a buffer which is aliased by outputs
// however the values are *not* modified.
class HloPoplarAliasReadOnlyUse : public HloPoplarAliasUseBase {
 public:
  HloPoplarAliasReadOnlyUse(HloInstruction* instruction, int64 operand_number,
                            const ShapeIndex& operand_index,
                            const std::vector<ShapeIndex> output_indices);
};

// A class for defining a single use of a buffer which is aliased by outputs
// however the values are modified.
class HloPoplarAliasReadWriteUse : public HloPoplarAliasUseBase {
 public:
  HloPoplarAliasReadWriteUse(HloInstruction* instruction, int64 operand_number,
                             const ShapeIndex& operand_index,
                             const std::vector<ShapeIndex> output_indices);
};

enum class BufferLocality {
  // Indicates that the buffer is stored in device memory.
  kDeviceMemory = 0,
  // Indicated that the buffer is stored in remote memory managed by Poplar.
  kRemoteMemory,
};

// Class used to describe a buffer being generated at a particular position.
class HloPoplarBufferDescription {
 public:
  HloPoplarBufferDescription(const ShapeIndex& output_index,
                             BufferLocality locality);

  // The shape index within the instruction in which the buffer appears as an
  // output.
  const ShapeIndex& output_index() const { return output_index_; }

  // The memory space this buffer will be stored in.
  BufferLocality locality() const { return locality_; }

  std::string ToString() const;

  bool operator==(const HloPoplarBufferDescription& other) const;
  bool operator!=(const HloPoplarBufferDescription& other) const;

 private:
  const ShapeIndex output_index_;
  const BufferLocality locality_;
};

using HloPoplarBufferDescriptions = std::vector<HloPoplarBufferDescription>;

// Class used to represent a single buffer tensor.
class HloPoplarBuffer {
 public:
  using Id = int64;
  // Predicate comparing HloPoplarBuffers by increasing id, useful for
  // std::sort.
  static bool IdLessThan(const HloPoplarBuffer* a, const HloPoplarBuffer* b) {
    return a->id() < b->id();
  }

  // Predicate comparing HloPoplarBuffers by equal id, useful for std::unique.
  static bool IdEqual(const HloPoplarBuffer* a, const HloPoplarBuffer* b) {
    return a->id() == b->id();
  }

  HloPoplarBuffer(Id id, const HloPoplarPosition& defining_position,
                  BufferLocality locality);

  Id id() const { return id_; }

  const HloPoplarPosition& defining_position() const {
    return defining_position_;
  }

  BufferLocality locality() const { return locality_; }

  HloInstruction* instruction() const {
    return defining_position().instruction;
  }

  const ShapeIndex& index() const { return defining_position().index; }

  const Shape& shape() const { return instruction()->shape(); }

  bool operator==(const HloPoplarBuffer& other) const;
  bool operator!=(const HloPoplarBuffer& other) const;

  std::string ToString() const;

 private:
  const Id id_;
  const HloPoplarPosition defining_position_;
  const BufferLocality locality_;
};

std::ostream& operator<<(std::ostream& out, const HloPoplarBuffer& buffer);

// A class representing the possible set of HloPoplarBuffer at a particular
// position in the XLA graph.
class HloPoplarBufferSet {
 public:
  HloPoplarBufferSet() = default;
  explicit HloPoplarBufferSet(absl::Span<const HloPoplarBuffer* const> buffers);

  // Return the vector of HloPoplarBuffer in the set. Buffers in the vector are
  // unique and stably sorted by buffer id.
  const std::vector<const HloPoplarBuffer*>& buffers() const {
    return buffers_;
  }

  int64 size() const { return buffers_.size(); }

  // Returns a unique buffer (if there is one).
  const HloPoplarBuffer& GetUniqueBuffer() const;

  // Add a buffer to the current set - returns true iff a buffer was added.
  bool AddBuffer(const HloPoplarBuffer* buffer);

  // Sets this buffer set to the union of the given buffer sets. Returns whether
  // this value set changed.
  bool AssignUnionOf(absl::Span<const HloPoplarBufferSet* const> buffer_sets);

  bool operator==(const HloPoplarBufferSet& other) const;
  bool operator!=(const HloPoplarBufferSet& other) const;

  std::string ToString() const;

 private:
  // Sorts buffers_ and removes duplicates. This should be called after adding
  // any elements to buffers_.
  void SortAndUniquifyBuffers();

  // HloPoplarBuffers sorted by HloPoplarBuffer::Id.
  std::vector<const HloPoplarBuffer*> buffers_;
};

std::ostream& operator<<(std::ostream& out,
                         const HloPoplarBufferSet& buffer_set);

// A class collecting the HloPoplarBuffers which might be contained in the
// output of an HLO instruction. For array-shaped instructions, an
// InstructionPoplarBufferSet trivially holds a single HloPoplarBufferSet.
// Tuple-shaped InstructionPoplarBufferSets hold multiple HloPoplarBufferSets.
class InstructionPoplarBufferSet {
 public:
  explicit InstructionPoplarBufferSet(const Shape& shape);

  // Set the buffer set for a particular output index.
  void SetOutputBufferSet(const ShapeIndex& output_index,
                          const HloPoplarBufferSet& buffer_set);
  // Gets the output buffer set at the output index, and reassigns it to the
  // union of the current set and the input set.
  void SetOutputToBufferSetUnion(const ShapeIndex& output_index,
                                 const HloPoplarBufferSet& buffer_set);

  // Get the buffer set for a particular output index.
  const HloPoplarBufferSet& GetOutputBufferSet(
      const ShapeIndex& output_index) const;
  HloPoplarBufferSet& GetMutableOutputBufferSet(const ShapeIndex& output_index);

  bool operator==(const InstructionPoplarBufferSet& other) const;
  bool operator!=(const InstructionPoplarBufferSet& other) const;

  std::string ToString() const;

 private:
  const Shape shape_;
  ShapeTree<HloPoplarBufferSet> buffer_sets_;
};

std::ostream& operator<<(
    std::ostream& out,
    const InstructionPoplarBufferSet& instruction_buffer_set);
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_BUFFER_H_
