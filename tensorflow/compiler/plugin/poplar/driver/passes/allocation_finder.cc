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

namespace {

struct AllocationLocation {
  TensorLocation location;
  Shape shape;
};

std::vector<AllocationLocation> FindAllocatingInstructions(
    const HloComputation* comp) {
  std::vector<AllocationLocation> allocation_locations;

  for (auto* inst : comp->MakeInstructionPostOrder()) {
    const bool allocating =
        CallHloInstructionExtension<AllocatingOutputExtension>(inst);
    if (allocating) {
      auto shape = inst->shape();
      if (auto* infeed = DynCast<HloInfeedInstruction>(inst)) {
        shape = infeed->infeed_shape();
      }

      const auto shapes = FlattenedXlaShape(shape);
      for (unsigned int i = 0; i < shapes.size(); i++) {
        allocation_locations.push_back({TensorLocation{inst, i}, shapes[i]});
      }
    }
  }

  return allocation_locations;
}
}  // namespace

int64 AllocationFinder::GetAllocationPriority(
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

bool AllocationFinder::ReplaceTarget(
    const TensorTarget& new_target, const TensorTarget& existing_target) const {
  const int64 new_target_priority = GetAllocationPriority(new_target);
  const int64 existing_target_priority = GetAllocationPriority(existing_target);
  if (new_target_priority > existing_target_priority) {
    // New target has higher priority.
    return true;
  } else if (new_target_priority == existing_target_priority) {
    // Replace if one instruction is marked as fwd and the other isn't.
    if (IsTrainingForward(new_target.tgt) &&
        !IsTrainingForward(existing_target.tgt)) {
      return true;
    }

    // Prefer paths without slice operations in them as they can create a high
    // tile imbalance.
    auto is_slice = [](const HloInstruction* inst) {
      return inst->opcode() == HloOpcode::kSlice;
    };

    if (absl::c_any_of(existing_target.backward_path, is_slice) &&
        !absl::c_any_of(new_target.backward_path, is_slice)) {
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

void AllocationFinder::AddTensorTarget(const TensorLocation& source,
                                       const TensorTarget& new_target) {
  // Check whether we should replace the tensor target.
  auto itr = tensor_allocation_map.find(source);
  if (itr != tensor_allocation_map.end()) {
    TensorTarget& target = itr->second;

    // Combine the sliceable dimension.
    absl::optional<int64> sliceable_dimension = target.sliceable_dimension;
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
    const TensorLocation& src, const HloInstruction* tgt, int64 index,
    absl::optional<std::vector<int64>> permutation) {
  path.emplace_back(tgt);

  for (auto user : tgt->users()) {
    int repeated_operand_offset = 0;
    auto& operands = user->operands();

    // How many times `user` has `tgt` as an operand.
    const std::size_t operand_count = absl::c_count(operands, tgt);

    for (std::size_t i = 0u; i < operand_count; ++i) {
      // Find the next position where tgt is an operand of user.
      auto op_itr = std::find(operands.begin() + repeated_operand_offset,
                              operands.end(), tgt);
      int64 op_index = std::distance(operands.begin(), op_itr);
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
          auto shapes = FlattenedXlaShape(src.instruction->shape());
          // Ignore the element type (paths can contain casts).
          if (shapes[src.flattened_output_tuple_index].dimensions() ==
              user->shape().dimensions()) {
            FindConsumers(src, user, index, permutation);
          }
          break;
        }
        case DoFindConsumers::TRUE: {
          FindConsumers(src, result.tgt, result.index, result.permutation);
          break;
        }
        case DoFindConsumers::FALSE:
        default: { break; }
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
        FindConsumers(src, caller, index, permutation);
      }
    }
  }

  path.pop_back();
  return;
}

StatusOr<bool> AllocationFinder::Run(HloModule* module) {
  call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition("Call graph must be flattened.");
  }

  for (const auto& comp : module->MakeComputationPostOrder()) {
    if (!IsPopOpsFusion(comp)) {
      for (auto allocation_location : FindAllocatingInstructions(comp)) {
        // Skip over opaque types
        if (allocation_location.shape.IsOpaque()) {
          continue;
        }

        // Starting dimensions permutation is just all the dimensions mapping to
        // themselves.
        std::vector<int64> permutation(allocation_location.shape.rank());
        absl::c_iota(permutation, 0);
        FindConsumers(allocation_location.location,
                      allocation_location.location.instruction,
                      allocation_location.location.flattened_output_tuple_index,
                      permutation);
      }
    }
  }

  return true;
}

AllocationFinder::AllocationFinder(CompilerAnnotations& annotations,
                                   bool always_rearrange_copies_on_host)
    : annotations(annotations),
      tensor_allocation_map(annotations.tensor_allocation_map),
      always_rearrange_copies_on_host(always_rearrange_copies_on_host) {}

}  // namespace poplarplugin
}  // namespace xla
