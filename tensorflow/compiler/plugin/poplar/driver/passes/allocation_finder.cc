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

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remap_deduce.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_instruction_extensions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_location.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

void DumpTensorAllocationMap(const TensorAllocationMap& map, int32 log_level) {
  if (!VLOG_IS_ON(log_level)) {
    return;
  }

  VLOG(log_level) << "Tensor allocation map:";
  if (map.empty()) {
    VLOG(log_level) << " --empty--";
    return;
  }
  const HloComputation* last_comp = nullptr;
  for (auto& entry : map) {
    auto& location = entry.first;
    const auto* inst = location.instruction;
    auto& target = entry.second;
    auto* comp = inst->parent();
    if (comp != last_comp) {
      VLOG(log_level) << " Computation " << comp->name() << ":";
      last_comp = comp;
    }
    VLOG(log_level) << "  " << inst->name() << ":"
                    << location.flattened_output_tuple_index
                    << ", target: " << target.tgt->name() << ":"
                    << target.input_index;
  }
}

namespace {

struct AllocationLocation {
  TensorLocation location;
  Shape shape;
};

std::vector<Shape> GetAllocationShapes(const HloInstruction* inst) {
  auto shape = inst->shape();
  if (auto* infeed = DynCast<HloInfeedInstruction>(inst)) {
    shape = infeed->infeed_shape();
  }

  return FlattenedXlaShape(shape);
}

std::vector<AllocationLocation> GetAllocationLocations(
    const HloInstruction* inst) {
  std::vector<AllocationLocation> allocation_locations;

  // Return empty list if instruction is not allocating.
  if (!CallHloInstructionExtension<AllocatingOutputExtension>(inst)) {
    return allocation_locations;
  }

  const auto shapes = GetAllocationShapes(inst);
  allocation_locations.reserve(shapes.size());

  for (unsigned int i = 0; i < shapes.size(); i++) {
    allocation_locations.push_back({TensorLocation{inst, i}, shapes[i]});
  }

  return allocation_locations;
}
}  // namespace

bool AllocationFinder::CanInferTarget(const TensorLocation& src,
                                      const HloInstruction* tgt) const {
  // An instruction cannot infer a target from itself.
  if (src.instruction == tgt) {
    return false;
  }
  // Instructions inside fusion computations do not get targets.
  if (IsPopOpsFusion(src.instruction->parent())) {
    return false;
  }

  // If instruction is not allocating it will not have a target.
  return CallHloInstructionExtension<AllocatingOutputExtension>(tgt);
}

// An inferred target is compatible if it contains all of the information we
// would accumulate if we continued traversing the graph normally.
bool AllocationFinder::IsInferredTargetCompatible(
    const TensorLocation& src, const TensorTarget& inferred_target) const {
  auto itr = tensor_allocation_map.find(src);
  if (itr == tensor_allocation_map.end()) {
    // If we have no target yet, then the inferred target is trivially
    // compatible as src is in same state as tgt was when it traversed the
    // graph.
    return true;
  }
  TensorTarget current_target = itr->second;

  // If neither target is a multi-slice/update then both will update
  // purely based on priority, so the inferred target is compatible.
  if (!IsMultiSliceOrUpdate(current_target.tgt) &&
      !IsMultiSliceOrUpdate(inferred_target.tgt)) {
    return true;
  }
  // If the targets are slice-plan compatible then they are both
  // multi-slice/updates and are accumulating the same slice plans.
  if (SlicePlanCompatibleTarget(inferred_target, current_target)) {
    return true;
  }
  // If target and new_target are not slice-plan compatible then they would
  // accumulate different slice plans when traversing the graph.
  // If new_target is higher priority than target, then this doesn't matter
  // because target will be entirely overridden anyway.
  return ReplaceTarget(inferred_target, current_target);
}

TensorTarget AllocationFinder::InferTarget(
    int64_t index, const absl::optional<std::vector<int64_t>>& permutation,
    const TensorTarget& tgt_tensor_target,
    std::vector<const HloInstruction*>& path) const {
  // Create backward path.
  std::vector<const HloInstruction*> backward_path;
  backward_path.reserve(path.size() + tgt_tensor_target.backward_path.size() -
                        1);
  backward_path.insert(backward_path.end(), path.begin() + 1, path.end());
  backward_path.insert(backward_path.end(),
                       tgt_tensor_target.backward_path.begin(),
                       tgt_tensor_target.backward_path.end());

  // Merge permutations by applying tgt permutation to src permutation.
  absl::optional<std::vector<int64_t>> combined_permutation;
  if (permutation.has_value() && tgt_tensor_target.permutation.has_value()) {
    int64_t size = tgt_tensor_target.permutation.value().size();
    combined_permutation = std::vector<int64_t>(size);
    for (int64_t i = 0; i < size; ++i) {
      combined_permutation.value()[i] =
          permutation.value()[tgt_tensor_target.permutation.value()[i]];
    }
  }

  TensorTarget tensor_target =
      TensorTarget(tgt_tensor_target.tgt, tgt_tensor_target.input_index,
                   backward_path, combined_permutation);

  // Take sliceable dimension.
  tensor_target.sliceable_dimension = tgt_tensor_target.sliceable_dimension;

  // Take slice plans.
  tensor_target.compatible_slice_plans =
      absl::flat_hash_set<const HloInstruction*>(
          tgt_tensor_target.compatible_slice_plans);

  return tensor_target;
}

int64_t AllocationFinder::GetAllocationPriority(
    const TensorTarget& target) const {
  // Allocation priority (highest to lowest):
  // * Used by Send/Recv,
  // * Used as a slice which is updating a bigger tensor,
  // * Used as input to matmul/convolution,
  // * Everything else.

  const bool is_host_copy =
      IsPoplarInstruction(PoplarOp::SendToHost)(target.tgt) ||
      IsPoplarInstruction(PoplarOp::RecvFromHost)(target.tgt);
  if (is_host_copy && !always_rearrange_copies_on_host) {
    // The memory cost of doing on-device stream copy rearrangment is large
    // enough that it is usually beneficial to prioritise the allocation layout
    // desired by the host exchange to avoid this rearrangement.
    return 3;
  }
  const bool is_multi_update =
      IsPoplarInstruction(PoplarOp::MultiUpdate)(target.tgt) ||
      IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(target.tgt);
  if (is_multi_update) {
    return target.input_index == 2 ? 2 : 0;
  }

  switch (target.tgt->opcode()) {
    case HloOpcode::kDynamicUpdateSlice: {
      return target.input_index == 1 ? 2 : 0;
    }
    case HloOpcode::kConvolution:
    case HloOpcode::kDot: {
      return 1;
    }
    case HloOpcode::kCustomCall:
    case HloOpcode::kFusion: {
      return IsPopOpsConvolution(target.tgt) ? 1 : 0;
    }
    default: { return 0; }
  }
}

bool IsPreferablePath(absl::Span<const HloInstruction* const> a,
                      absl::Span<const HloInstruction* const> b) {
  // Returns true if path a is preferable to path b.
  // Prefer paths without slice operations in them as they can create a high
  // tile imbalance.
  auto is_slice = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kSlice;
  };

  if (absl::c_any_of(b, is_slice) && !absl::c_any_of(a, is_slice)) {
    return true;
  }
  return false;
}

bool AllocationFinder::ReplaceTarget(
    const TensorTarget& new_target, const TensorTarget& existing_target) const {
  const int64_t new_target_priority = GetAllocationPriority(new_target);
  const int64_t existing_target_priority =
      GetAllocationPriority(existing_target);
  if (new_target_priority > existing_target_priority) {
    // New target has higher priority.
    return true;
  } else if (new_target_priority == existing_target_priority) {
    // Replace if one instruction is marked as fwd and the other isn't.
    if (IsTrainingForward(new_target.tgt) &&
        !IsTrainingForward(existing_target.tgt)) {
      return true;
    }
    if (IsPreferablePath(new_target.backward_path,
                         existing_target.backward_path)) {
      return true;
    }
    return false;
  } else {
    // Existing target has higher priority.
    return false;
  }
}

bool AllocationFinder::SlicePlanCompatibleTarget(const TensorTarget& a,
                                                 const TensorTarget& b) const {
  // Consider only allocating indices.
  if (IsMultiSliceOrUpdate(a.tgt) && IsMultiSliceOrUpdate(b.tgt) &&
      a.input_index == b.input_index && a.input_index < 2) {
    return true;
  }

  return false;
}

// Returns true if the tensor target for source was affected by the new target.
void AllocationFinder::AddTensorTarget(const TensorLocation& source,
                                       const TensorTarget& new_target) {
  // Check whether we should replace the tensor target.
  auto itr = tensor_allocation_map.find(source);
  if (itr != tensor_allocation_map.end()) {
    TensorTarget& target = itr->second;

    // Combine the sliceable dimension.
    absl::optional<int64_t> sliceable_dimension = target.sliceable_dimension;
    if (new_target.sliceable_dimension) {
      sliceable_dimension = new_target.sliceable_dimension;
    }

    if (SlicePlanCompatibleTarget(new_target, target)) {
      // We have new target with compatible plan. Add it and its compatible
      // plans instructions if any.
      target.compatible_slice_plans.insert(
          new_target.compatible_slice_plans.begin(),
          new_target.compatible_slice_plans.end());
      target.compatible_slice_plans.insert(new_target.tgt);
    } else if (ReplaceTarget(new_target, target)) {
      target = new_target;
    }

    target.sliceable_dimension = sliceable_dimension;
  } else {
    tensor_allocation_map[source] = new_target;
  }
}

void AllocationFinder::FindConsumers(
    const TensorLocation& src, const HloInstruction* tgt, int64_t index,
    absl::optional<std::vector<int64_t>> permutation,
    std::vector<const HloInstruction*>& path,
    absl::flat_hash_set<TensorLocation>& explored) {
  VLOG(10) << "Finding consumers of " << tgt->name();
  path.emplace_back(tgt);
  auto tgt_location = TensorLocation{tgt, index};

  if (!explored.contains(tgt_location)) {
    explored.insert(tgt_location);
  } else {
    // The path has already been explored from this point. The only benefit we
    // could get from reexploring the rest of the path is if we find
    // preferable overall paths (e.g. without slices) to the same targets we
    // have already found.
    // Compare the current path to the path of the existing target. If the
    // current path is prefereable then it's worth continuing to explore it.

    auto itr = tensor_allocation_map.find(src);
    if (itr == tensor_allocation_map.end()) {
      // No target has been found yet, meaning there are no targets on this
      // path.
      path.pop_back();
      VLOG(10) << "No target has been previously found.";
      return;
    }
    auto& allocation_path = itr->second.backward_path;

    // If the existing allocation path contains the current target node, look at
    // the subpath up to that node, otherwise look at the entire path.
    auto existing_path = absl::MakeConstSpan(allocation_path)
                             .first(absl::c_find(allocation_path, tgt) -
                                    allocation_path.begin());

    // The paths in targets do not have the first element (the source inst).
    auto current_path = absl::MakeConstSpan(path).subspan(1);

    if (!IsPreferablePath(current_path, existing_path)) {
      // If the current path is not already preferable to the existing path,
      // there is no point reexploring the rest of it. The current path can
      // only get less preferable as it gets longer.
      path.pop_back();
      VLOG(10) << "This path isn't preferable.";
      return;
    }
  }

  // Targets can usually be inferred from allocating instructions without
  // needing to re-traverse the graph.
  if (CanInferTarget(src, tgt)) {
    // If allocation finding has not been run for tgt yet then run it now.
    if (!completed.contains(tgt_location)) {
      const auto shape = GetAllocationShapes(tgt)[index];
      FindAllocation(tgt_location, shape);
    }

    auto itr = tensor_allocation_map.find(tgt_location);
    if (itr == tensor_allocation_map.end()) {
      // If we have run allocation finding there is no target then we can
      // infer that there are no targets in this part of the graph so we do not
      // need to traverse it.
      path.pop_back();
      VLOG(10) << "This path isn't preferable.";
      return;
    }
    TensorTarget new_target =
        InferTarget(index, permutation, itr->second, path);
    if (IsInferredTargetCompatible(src, new_target)) {
      AddTensorTarget(src, new_target);
      path.pop_back();
      VLOG(10) << "Inferred target is not compatible.";
      return;
    }
  }

  for (auto user : tgt->users()) {
    VLOG(10) << "Inspecting user " << user->name();
    int repeated_operand_offset = 0;
    auto& operands = user->operands();

    // How many times `user` has `tgt` as an operand.
    const std::size_t operand_count = absl::c_count(operands, tgt);

    for (std::size_t i = 0u; i < operand_count; ++i) {
      // Find the next position where tgt is an operand of user.
      auto op_itr = std::find(operands.begin() + repeated_operand_offset,
                              operands.end(), tgt);
      int64_t op_index = std::distance(operands.begin(), op_itr);
      repeated_operand_offset = op_index + 1;

      // The backward path does not contain the source.
      std::vector<const HloInstruction*> backward_path = path;
      backward_path.erase(std::begin(backward_path));

      auto tensor_target =
          TensorTarget(user, op_index, backward_path, permutation);

      switch (user->opcode()) {
        case HloOpcode::kCholesky:
        case HloOpcode::kConvolution:
        case HloOpcode::kDot:
        case HloOpcode::kScatter:
        case HloOpcode::kTriangularSolve:
        case HloOpcode::kGather: {
          const auto allocating_indices =
              CallHloInstructionExtension<AllocatingIndicesExtension>(user);
          if (allocating_indices.count(op_index)) {
            AddTensorTarget(src, tensor_target);
          }
          break;
        }
        case HloOpcode::kDynamicSlice: {
          if (op_index == 0) {
            const SliceInfo slice_info =
                GetSliceInfo(user->operand(0)->shape(), user->shape());
            // If the DynamicSlice has a valid permutation, and a single slice
            // dimension, then request for it to be sliceable in that dimension.
            if (tensor_target.permutation &&
                slice_info.sliced_dims.size() == 1) {
              tensor_target.sliceable_dimension =
                  (*tensor_target.permutation)[slice_info.sliced_dims[0]];
            }
            AddTensorTarget(src, tensor_target);
          }
          break;
        }
        case HloOpcode::kDynamicUpdateSlice: {
          if (op_index == 0 || op_index == 1) {
            const SliceInfo slice_info =
                GetSliceInfo(user->shape(), user->operand(1)->shape());

            // If the DynamicUpdateSlice has a valid permutation, and a single
            // slice dimension, then request for it to be sliceable in that
            // dimension.
            if (op_index == 0 && tensor_target.permutation &&
                slice_info.sliced_dims.size() == 1) {
              tensor_target.sliceable_dimension =
                  (*tensor_target.permutation)[slice_info.sliced_dims[0]];
            }
            AddTensorTarget(src, tensor_target);
          }
          break;
        }
        case HloOpcode::kCustomCall: {
          if (IsPoplibsHloCustomOp(user)) {
            auto poplar_inst = Cast<HloPoplarInstruction>(user);
            auto allocating_indexes = poplar_inst->AllocatingIndices();

            if (allocating_indexes.count(op_index)) {
              // Request that the tensor for operand 0 should be allocated to be
              // sliceable on dimension 0 if we have a valid dimension
              // permutation.
              const bool is_multi_update =
                  IsPoplarInstruction(PoplarOp::MultiUpdate)(user) ||
                  IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(user);
              if (op_index == 0 && tensor_target.permutation &&
                  (IsPoplarInstruction(PoplarOp::MultiSlice)(user) ||
                   is_multi_update)) {
                CHECK(!tensor_target.permutation->empty());
                tensor_target.sliceable_dimension =
                    (*tensor_target.permutation)[0];
              }
              AddTensorTarget(src, tensor_target);
            }
          }
          break;
        }
      }

      // FindConsumer look through.
      auto result = CallHloInstructionExtension<FindConsumersExtension,
                                                FindConsumersExtensionParams>(
          user,
          FindConsumersExtensionParams{src, tgt, index, op_index, permutation});
      switch (result.do_find_consumers) {
        case DoFindConsumers::UNSPECIFIED: {
          if (ShapeUtil::Compatible(tgt->shape(), user->shape())) {
            VLOG(10) << "FindConsumer extension returned UNSPECIFIED and "
                        "shapes are compatible.";
            FindConsumers(src, user, index, permutation, path, explored);
          } else {
            VLOG(10) << "FindConsumer extension returned UNSPECIFIED and "
                        "shapes are not compatible.";
          }
          break;
        }
        case DoFindConsumers::TRUE: {
          VLOG(10) << "FindConsumer extension returned TRUE.";
          FindConsumers(src, result.tgt, result.index, result.permutation, path,
                        explored);
          break;
        }
        case DoFindConsumers::FALSE:
        default: {
          VLOG(10) << "FindConsumer extension returned FALSE.";
          break;
        }
      }
    }
  }

  // If the target is the root instruction, look through it's caller.
  const auto comp = tgt->parent();
  if (comp->root_instruction() == tgt) {
    auto callsites = call_graph->GetNode(comp).caller_callsites();
    if (callsites.size() == 1) {
      auto caller = callsites.front().instruction();
      if (caller->shape() == tgt->shape()) {
        FindConsumers(src, caller, index, permutation, path, explored);
      }
    }
  }

  path.pop_back();
  return;
}

void AllocationFinder::FindAllocation(const TensorLocation& location,
                                      const Shape& shape) {
  if (completed.contains(location)) {
    return;
  }
  // Skip over opaque types
  if (shape.IsOpaque()) {
    return;
  }

  // Starting dimensions permutation is just all the dimensions mapping to
  // themselves.
  std::vector<int64_t> permutation(shape.rank());
  absl::c_iota(permutation, 0);
  std::vector<const HloInstruction*> path;
  absl::flat_hash_set<TensorLocation> explored;
  FindConsumers(location, location.instruction,
                location.flattened_output_tuple_index, permutation, path,
                explored);

  // Mark this location as completed (has had allocation finding run).
  completed.insert(location);
}

StatusOr<bool> AllocationFinder::Run(HloModule* module) {
  call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition("Call graph must be flattened.");
  }

  for (const auto& comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      for (auto& allocation_location : GetAllocationLocations(inst)) {
        FindAllocation(allocation_location.location, allocation_location.shape);
      }
    }
  }

  VLOG(2) << "After the AllocationFinder:";
  DumpTensorAllocationMap(tensor_allocation_map, 2);

  return true;
}

AllocationFinder::AllocationFinder(CompilerAnnotations& annotations,
                                   bool always_rearrange_copies_on_host)
    : annotations(annotations),
      tensor_allocation_map(annotations.tensor_allocation_map),
      always_rearrange_copies_on_host(always_rearrange_copies_on_host) {}

}  // namespace poplarplugin
}  // namespace xla
