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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H

#include <map>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloUserOpInstruction : public HloPoplarInstruction {
 public:
  using MetadataFn = void (*)(std::vector<std::int64_t>&,
                              std::vector<std::int64_t>&,
                              std::map<std::int64_t, std::int64_t>&, bool&,
                              bool&, bool&, std::uint32_t);

  explicit HloUserOpInstruction(absl::Span<HloInstruction* const> operands,
                                const Shape& shape, const std::string& gp_path,
                                void*, void*, void*, int64 gradient_size,
                                int64 partial_derivative_index,
                                bool is_user_read_write,
                                const std::string& attributes);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_set<int64> ReplicaIdenticalOutputIndices() const;
  bool AllocatingOutput() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  HloPoplarUseDescriptions GetUseDescriptions() const override;
  HloPoplarBufferDescriptions GetBufferDescriptions() const override;

  const FindConsumersExtensionResults FindConsumers(
      FindConsumersExtensionParams params) const override;

  bool IsPopOpsElementwise() const override;

  size_t NumInputs() const { return num_inputs_; }

  void* GetPointerToFunc() const { return function_ptr_; }

  void* GetAllocatorFunc() const { return allocator_function_ptr_; }

  const std::string& GetPath() const { return gp_path_; }

  bool IsGradient() const { return gradient_size_ != 0; }

  int GetGradientSize() const { return gradient_size_; }

  int PartialDerivativeIndex() const { return partial_derivative_index_; }

  bool IsReadWrite() const { return is_user_read_write_; }

  const std::string& GetAttributes() const { return attributes_; }

  bool IsHashable() const { return metadata_.is_hashable_; }

 protected:
  std::vector<string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const override;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  // The pointer to the function provided by the user via the shared library.
  void* function_ptr_;

  // The pointer to the metadata, if provided by the user, via the shared
  // library.
  void* metadata_function_ptr_;

  // The pointer to the allocation function, if provided by the user, via the
  // shared library.
  void* allocator_function_ptr_;

  // The number of inputs to this operation.
  size_t num_inputs_;

  // The path to the codelet GP file
  std::string gp_path_;

  struct MetadataStructure {
    std::vector<std::int64_t> allocating_indices_;
    std::vector<std::int64_t> replica_identical_output_indices_;
    std::map<std::int64_t, std::int64_t> input_to_output_tensor_aliasing_;
    bool is_elementwise_ = false;
    bool is_hashable_ = false;
  };

  MetadataStructure metadata_;

  // When greater than zero, this is a gradient operation with given gradients,
  // rather than a forward pass operation.
  int64 gradient_size_;

  // If this is a grad function, and it is the partial derivative for only one
  // input, then this is the index of that input.
  int64 partial_derivative_index_;

  // Is this a read/write user op. That is an operation which streams the
  // tensors to host, executes some processing, then streams the outputs back.
  bool is_user_read_write_;

  std::string attributes_;
};

std::unique_ptr<HloInstruction> CreateUserOp(
    absl::Span<HloInstruction* const> operands, const Shape& shape,
    const std::string& gp_path, void*, void*, void*, int64 gradient_size,
    int64 partial_derivative_index, bool is_user_read_write,
    const std::string& attributes);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
