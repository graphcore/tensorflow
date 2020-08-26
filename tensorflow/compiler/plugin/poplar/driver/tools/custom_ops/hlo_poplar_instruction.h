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

// All HloInstruction subclasses are put in this file.

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_CUSTOM_OP_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_CUSTOM_OP_H_

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "tensorflow/core/platform/str_util.h"

namespace xla {
namespace poplarplugin {

Shape GetHloPoplarInstructionShape(absl::Span<HloInstruction* const> operands);

// Base class for a poplar hlo instruction
class HloPoplarInstruction : public HloCustomCallInstruction {
 public:
  // Note that the variadic arguments "attributes" is used to create the opaque
  // field.
  template <typename... Args>
  HloPoplarInstruction(const Shape& shape,
                       absl::Span<HloInstruction* const> operands, PoplarOp op,
                       Args&&... attributes)
      : HloCustomCallInstruction(shape, operands, PoplarOp_Name(op), "") {
    // Hash all attributes to prevent comparisons of ops from false positives
    int64 hash = hash_util::hash(std::forward<Args>(attributes)...);

    // Poke the hash of the attributes into the backend config
    PoplarBackendConfig backend_config;
    backend_config.set_hash_of_custom_attributes(hash);
    set_backend_config(backend_config);

    // Give it a name based on the Poplar op name.
    auto name = tensorflow::str_util::ArgDefCase(PoplarOp_Name(op));
    std::replace(name.begin(), name.end(), '_', '-');
    SetAndSanitizeName(name);
  }

  // Allocating indexes used by the Allocation Finder - op specific.
  virtual absl::flat_hash_set<int64> AllocatingIndices() const = 0;

  // Layout dependent indexes used by the Forward Allocation Finder - op
  // specific.
  // Example - custom op has 3 inputs a(0), b(1), c(2) - a is an input, and b
  // and c are dependent on allocation of a, then
  // this map would look as follows:
  // { {1, 0}, {2, 0} }
  // Note that the dependent allocation cannot be an Allocating index or another
  // layout dependency.
  virtual absl::flat_hash_map<int64, int64> LayoutDependencies() const = 0;

  // Return how many of the first n operands are updated in place. If 0, the op
  // is treated as NotInplace.
  virtual uint64 NumberOfInplaceOperands() const = 0;

  // Returns whether this is an elementwise instruction.
  virtual bool IsPopOpsElementwise() const = 0;

 protected:
  virtual std::vector<string> ExtraPoplarAttributesToStringImpl(
      const HloPrintOptions& options) const = 0;

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext*) const override = 0;

  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override final;
};

class HloPoplarInstructionFactory {
 public:
  using FactoryType = std::function<StatusOr<std::unique_ptr<HloInstruction>>(
      HloCustomCallInstruction*)>;

  HloPoplarInstructionFactory(PoplarOp op, FactoryType factory);

  static bool IsCreatable(HloCustomCallInstruction* inst);
  static StatusOr<std::unique_ptr<HloInstruction>> Create(
      HloCustomCallInstruction* inst);

 private:
  static std::unordered_map<std::string, FactoryType>& GetFactoryMap();
};

// Returns true if inst is a call to a custom hlo op for Poplibs
const bool IsPoplibsHloCustomOp(const HloInstruction* inst);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_CUSTOM_OP_H_
