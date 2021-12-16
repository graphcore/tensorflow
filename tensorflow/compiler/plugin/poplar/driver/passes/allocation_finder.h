/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_FINDER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ALLOCATION_FINDER_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_location.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

struct CompilerAnnotations;

struct TensorTarget {
  // The node in the graph which consumes the tensor
  const HloInstruction* tgt;

  // The input on the target node which consumes the tensor
  int64 input_index;

  // A node in the graph which produces a tensor that influences the
  // construction of the tensor.  Example: bias tensors should match the layout
  // of a convolution output.  'layout' points to the convolution parameter.
  absl::optional<const HloInstruction*> layout;

  // Layout can have multiple output tensors - this index identifies which
  // output tensor to use.
  absl::optional<int64> layout_output_idx;

  // A vector of operations between the source and target operations.  Sometimes
  // it is possible to allocate a tensor for consumption by a target, and then
  // transform it into the tensor as it should be allocated by the source.
  std::vector<const HloInstruction*> forward_path;

  // A path from the layout influencing operation and the target operation.
  // Sometimes it is possible to take the output of the target and transform
  // it into something that can be used to make a layout-dependent allocation
  // at the target site.
  std::vector<const HloInstruction*> backward_path;

  // Optional permutation of the dimensions at the input to the dimensions at
  // the allocation location created by traversing the path from the input to
  // the allocation.
  absl::optional<std::vector<int64>> permutation;

  // Optional indicator for requesting a particular input dimension to be
  // sliceable.
  absl::optional<int64> sliceable_dimension = absl::nullopt;

  // Set of instructions which use compatible plans/allocated tensors.
  absl::flat_hash_set<const HloInstruction*> compatible_slice_plans;

  TensorTarget(const HloInstruction* tgt, int64 input_index,
               absl::optional<const HloInstruction*> layout,
               absl::optional<int64> layout_output_idx,
               const std::vector<const HloInstruction*>& forward_path = {},
               const std::vector<const HloInstruction*>& backward_path = {},
               absl::optional<std::vector<int64>> permutation = absl::nullopt)
      : tgt(tgt),
        input_index(input_index),
        layout(layout),
        layout_output_idx(layout_output_idx),
        forward_path(forward_path),
        backward_path(backward_path),
        permutation(permutation) {}

  TensorTarget(const HloInstruction* tgt, int64 input_index,
               const std::vector<const HloInstruction*>& backward_path = {},
               absl::optional<std::vector<int64>> permutation = absl::nullopt)
      : TensorTarget(tgt, input_index, absl::nullopt, absl::nullopt, {},
                     backward_path, permutation) {}

  TensorTarget() = default;
};

using TensorAllocationMap = std::map<TensorLocation, TensorTarget>;

// Dumps tensor allocation map with given log level
void DumpTensorAllocationMap(const TensorAllocationMap& map, int32 log_level);

/**
 * This class finds all instructions that explicitly add tensors to the
 * graph.  For each one of them, it locates the downstream consumers of that
 * tensor, and if any of those instructions require a specific tensor allocation
 * method (e.g. convolution), then it notes the downstream instruction
 */
class AllocationFinder : public HloModulePass {
 public:
  AllocationFinder(CompilerAnnotations& annotations,
                   bool always_rearrange_copies_on_host = false);

  ~AllocationFinder() = default;

  absl::string_view name() const override { return "allocation-finder"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool CanInferTarget(const TensorLocation& src,
                      const HloInstruction* tgt) const;
  bool IsInferredTargetCompatible(const TensorLocation& src,
                                  const TensorTarget& inferred_target) const;

  TensorTarget InferTarget(
      int64 index, const absl::optional<std::vector<int64>>& permutation,
      const TensorTarget& tgt_tensor_target,
      std::vector<const HloInstruction*>& path) const;

  void FindAllocation(const TensorLocation& location, const Shape& shape);

  void FindConsumers(const TensorLocation&, const HloInstruction* tgt, int64,
                     absl::optional<std::vector<int64>>,
                     std::vector<const HloInstruction*>&);

  int64 GetAllocationPriority(const TensorTarget& target) const;

  // Should return true when target 'a' should be used over 'b'
  bool ReplaceTarget(const TensorTarget& a, const TensorTarget& b) const;
  bool SlicePlanCompatibleTarget(const TensorTarget& a,
                                 const TensorTarget& b) const;

  void AddTensorTarget(const TensorLocation& source,
                       const TensorTarget& new_target);

  std::unique_ptr<CallGraph> call_graph;

  // The set of TensorLocations which allocation finding has been completed for.
  absl::flat_hash_set<TensorLocation> completed;

  const CompilerAnnotations& annotations;
  TensorAllocationMap& tensor_allocation_map;
  const bool always_rearrange_copies_on_host;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
