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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloUserOpInstruction : public HloPoplarInstruction {
 public:
  explicit HloUserOpInstruction(absl::Span<HloInstruction* const> operands,
                                const Shape& shape, const std::string& gp_path,
                                void*, void*, void*, bool is_gradient,
                                int partial_derivative_index,
                                bool is_user_read_write);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const override;

  size_t NumInputs() const { return num_inputs_; }

  void* GetPointerToFunc() const { return function_ptr_; }

  void* GetAllocatorFunc() const { return allocator_function_ptr_; }

  const std::string& GetPath() const { return gp_path_; }

  bool IsGradient() const { return is_gradient_; }

  int PartialDerivativeIndex() const { return partial_derivative_index_; }

  bool IsReadWrite() const { return is_user_read_write_; }

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
    MetadataStructure()
        : allocating_indices_({}), num_inplace_(0), is_elementwise_(false) {}
    std::vector<std::int64_t> allocating_indices_;
    std::uint32_t num_inplace_;
    bool is_elementwise_;
  };

  MetadataStructure metadata_;

  // When true, this is a gradient operation, rather than a forward pass
  // operation.
  bool is_gradient_;

  // If this is a grad function, and it is the partial derivative for only one
  // input, then this is the index of that input.
  int partial_derivative_index_;

  // Is this a read/write user op. That is an operation which streams the
  // tensors to host, executes some processing, then streams the outputs back.
  bool is_user_read_write_;
};

std::unique_ptr<HloInstruction> CreateUserOp(
    absl::Span<HloInstruction* const> operands, const Shape& shape,
    const std::string& gp_path, void*, void*, void*, bool is_gradient,
    int partial_derivative_index, bool is_user_read_write);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
