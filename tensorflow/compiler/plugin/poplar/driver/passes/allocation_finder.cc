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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
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

// Find the index of a tensor after extracting it (or a tuple containing it)
// from a tuple. tuple_index is the index of one of the elements of the tuple,
// and original_index is the tensor position within the original tuple.
int64 ExtractFromTuple(const Shape& tuple, int64 tuple_index,
                       int64 original_index) {
  int64 index = original_index;
  for (int64 i = 0; i < tuple_index; i++) {
    index -= CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  int64 n = CountShapes(ShapeUtil::GetTupleElementShape(tuple, tuple_index));
  if (index < 0 || index >= n) {
    return -1;
  }
  return index;
}

struct AllocationLocation {
  TensorLocation location;
  Shape shape;
};

class FindAllocatingInstructions : public DfsHloVisitorWithDefault {
 public:
  FindAllocatingInstructions() {}

  ~FindAllocatingInstructions() override = default;

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* inst) override {
    allocation_locations.push_back({TensorLocation{inst, 0}, inst->shape()});
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* inst) override {
    allocation_locations.push_back({TensorLocation{inst, 0}, inst->shape()});
    return Status::OK();
  }

  Status HandleRng(HloInstruction* inst) override {
    allocation_locations.push_back({TensorLocation{inst, 0}, inst->shape()});
    return Status::OK();
  }

  Status HandleParameter(HloInstruction* inst) override {
    auto shapes = FlattenedXlaShape(inst->shape());
    for (unsigned int i = 0; i < shapes.size(); i++) {
      allocation_locations.push_back({TensorLocation{inst, i}, shapes[i]});
    }
    return Status::OK();
  }

  Status HandleInfeed(HloInstruction* inst) override {
    HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);
    auto shapes = FlattenedXlaShape(infeed->infeed_shape());
    for (unsigned int i = 0; i < shapes.size(); i++) {
      allocation_locations.push_back({TensorLocation{inst, i}, shapes[i]});
    }
    return Status::OK();
  }

  Status HandleCustomCall(HloInstruction* inst) override {
    const bool is_remap_deduce =
        IsPoplarInstruction(PoplarOp::RemapDeduce)(inst);
    const bool is_host_embedding_lookup =
        IsPoplarInstruction(PoplarOp::HostEmbeddingLookup)(inst);
    const bool is_remote_buffer_load =
        IsPoplarInstruction(PoplarOp::RemoteParameterLoad)(inst);
    const bool is_rw_user_op =
        IsPoplarInstruction(PoplarOp::UserOp)(inst)
            ? Cast<HloUserOpInstruction>(inst)->IsReadWrite()
            : false;
    const bool is_recv_from_host =
        IsPoplarInstruction(PoplarOp::RecvFromHost)(inst);
    const bool is_gradient_accumulator_create =
        IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst);

    if (is_remap_deduce || is_host_embedding_lookup || is_remote_buffer_load ||
        is_rw_user_op || is_recv_from_host || is_gradient_accumulator_create) {
      auto shapes = FlattenedXlaShape(inst->shape());
      for (unsigned int i = 0; i < shapes.size(); i++) {
        allocation_locations.push_back({TensorLocation{inst, i}, shapes[i]});
      }
    }

    return Status::OK();
  }

  Status HandleFusion(HloInstruction* inst) override {
    if (IsPopOpsFusion(inst, "wide_const") ||
        IsPopOpsFusion(inst, "reduction_fp16_input")) {
      allocation_locations.push_back({TensorLocation{inst, 0}, inst->shape()});
    }
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* inst) override {
    allocation_locations.push_back({TensorLocation{inst, 0}, inst->shape()});
    return Status::OK();
  }
  std::vector<AllocationLocation> allocation_locations;
};
}  // namespace

int64 AllocationFinder::GetAllocationPriority(
    const TensorTarget& target) const {
  const bool is_host_copy =
      IsPoplarInstruction(PoplarOp::SendToHost)(target.tgt) ||
      IsPoplarInstruction(PoplarOp::RecvFromHost)(target.tgt);
  if (is_host_copy && !always_rearrange_copies_on_host) {
    // The memory cost of doing on-device stream copy rearrangment is large
    // enough that it is usually beneficial to prioritise the allocation layout
    // desired by the host exchange to avoid this rearrangement.
    return 2;
  }

  switch (target.tgt->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kDot: {
      return 1;
    }
    case HloOpcode::kFusion: {
      return IsPopOpsConvolution(target.tgt) ? 1 : 0;
    }
    default: { return 0; }
  }
}

bool AllocationFinder::ReplaceTarget(const TensorTarget& new_target,
                                     const TensorTarget& existing_target) {
  const int64 new_target_priority = GetAllocationPriority(new_target);
  const int64 existing_target_priority = GetAllocationPriority(existing_target);
  if (new_target_priority > existing_target_priority) {
    // New target has higher priority.
    return true;
  } else if (new_target_priority == existing_target_priority) {
    // Replace if one instruction is marked as fwd and the other isn't.
    return IsTrainingForward(new_target.tgt) &&
           !IsTrainingForward(existing_target.tgt);

  } else {
    // Existing target has higher priority.
    return false;
  }
}

void AllocationFinder::AddTensorTarget(const TensorLocation& source,
                                       const TensorTarget& new_target) {
  TensorTarget target;
  // Check whether we should replace the tensor target.
  auto itr = tensor_allocation_map.find(source);
  if (itr != tensor_allocation_map.end()) {
    target = itr->second;

    // Combine the sliceable dimension.
    absl::optional<int64> sliceable_dimension = target.sliceable_dimension;
    if (new_target.sliceable_dimension) {
      sliceable_dimension = new_target.sliceable_dimension;
    }

    if (ReplaceTarget(new_target, target)) {
      target = new_target;
    }

    target.sliceable_dimension = sliceable_dimension;
  } else {
    target = new_target;
  }

  tensor_allocation_map[source] = target;
}

void AllocationFinder::FindConsumers(
    const TensorLocation& src, const HloInstruction* tgt, int64 index,
    absl::optional<std::vector<int64>> permutation) {
  path.emplace_back(tgt);
  for (auto user : tgt->users()) {
    if (visited.count(user) == 0) {
      visited.insert(user);
      int64 op_index = user->operand_index(tgt);
      auto tensor_target = TensorTarget(user, op_index, path, permutation);
      switch (user->opcode()) {
        case HloOpcode::kConvolution:
        case HloOpcode::kDot: {
          AddTensorTarget(src, tensor_target);
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
        case HloOpcode::kScatter: {
          if (op_index == 0 || op_index == 1 || op_index == 2) {
            AddTensorTarget(src, tensor_target);
          }
          break;
        }
        case HloOpcode::kGather: {
          if (op_index == 0 || op_index == 1) {
            AddTensorTarget(src, tensor_target);
          }
          break;
        }
        case HloOpcode::kCall: {
          // This also handles repeat loops which are represented as a Call
          // operation.
          HloComputation* comp = user->to_apply();
          HloInstruction* param = comp->parameter_instruction(op_index);
          FindConsumers(src, param, index, permutation);
          break;
        }
        case HloOpcode::kFusion: {
          HloComputation* comp = user->fused_instructions_computation();
          if (IsPopOpsFusion(user)) {
            if (IsPopOpsFusion(user, "depthwise_conv")) {
              AddTensorTarget(src, tensor_target);
            } else if (IsPopOpsFusion(user, "zero_pad")) {
              FindConsumers(src, user, index, permutation);
            } else if (IsPopOpsFusion(user, "scaled_inplace") && op_index < 2) {
              // Look through the scaled inplace op.
              FindConsumers(src, user, index, permutation);
            } else if (IsPopOpsFusion(user, "implicit")) {
              // Look through implicit elementwise ops if the shapes match.
              auto shapes = FlattenedXlaShape(src.instruction->shape());
              if (shapes[src.flattened_output_tuple_index] == user->shape() &&
                  user->shape() == user->operand(op_index)->shape()) {
                FindConsumers(src, user, index, permutation);
              }
            }
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
              if (op_index == 0 && tensor_target.permutation &&
                  (IsPoplarInstruction(PoplarOp::MultiSlice)(user) ||
                   IsPoplarInstruction(PoplarOp::MultiUpdate)(user) ||
                   IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(user))) {
                CHECK(!tensor_target.permutation->empty());
                tensor_target.sliceable_dimension =
                    (*tensor_target.permutation)[0];
              }
              AddTensorTarget(src, tensor_target);
            }
          } else {
            auto shapes = FlattenedXlaShape(src.instruction->shape());
            if (shapes[src.flattened_output_tuple_index] == user->shape()) {
              FindConsumers(src, user, index, permutation);
            }
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* comp = user->while_body();
          HloInstruction* param = comp->parameter_instruction(op_index);
          FindConsumers(src, param, index, permutation);
          break;
        }
        case HloOpcode::kTuple: {
          int64 new_index = InsertIntoTuple(user->shape(), op_index, index);
          FindConsumers(src, user, new_index, permutation);
          break;
        }
        case HloOpcode::kGetTupleElement: {
          int64 tuple_index = user->tuple_index();
          int64 new_index = ExtractFromTuple(tgt->shape(), tuple_index, index);
          if (new_index != -1) {
            FindConsumers(src, user, new_index, permutation);
          }
          break;
        }
        case HloOpcode::kReshape: {
          // Can look through reshapes, but cannot track dimension permutation.
          FindConsumers(src, user, index, absl::nullopt);
          break;
        }
        case HloOpcode::kTranspose: {
          absl::optional<std::vector<int64>> new_permutation;
          if (permutation) {
            // Permute the dimensions according to the transpose.
            new_permutation = std::vector<int64>(permutation->size());
            const std::vector<int64> transpose_permutation = user->dimensions();
            for (int64 d = 0; d != permutation->size(); ++d) {
              (*new_permutation)[d] = (*permutation)[transpose_permutation[d]];
            }
          }
          FindConsumers(src, user, index, new_permutation);
          break;
        }
        case HloOpcode::kConvert: {
          FindConsumers(src, user, index, permutation);
          break;
        }
        case HloOpcode::kConcatenate: {
          FindConsumers(src, user, index, permutation);
          break;
        }
        case HloOpcode::kSlice: {
          if (IsUniformSingleDimSlice(user)) {
            FindConsumers(src, user, index, permutation);
          }
          break;
        }
        case HloOpcode::kPad: {
          if (op_index == 0) {
            FindConsumers(src, user, index, permutation);
          }
          break;
        }
        default: {
          auto shapes = FlattenedXlaShape(src.instruction->shape());
          if (shapes[src.flattened_output_tuple_index] == user->shape()) {
            FindConsumers(src, user, index, permutation);
          }
          break;
        }
      }
    }
  }
  path.pop_back();
  return;
}

StatusOr<bool> AllocationFinder::Run(HloModule* module) {
  FindAllocatingInstructions finder;

  for (const auto& comp : module->MakeComputationPostOrder()) {
    if (!IsPopOpsFusion(comp)) {
      TF_RETURN_IF_ERROR(comp->Accept(&finder));
    }
  }

  for (auto allocation_location : finder.allocation_locations) {
    visited.clear();
    // Starting dimensions permutation is just all the dimensions mapping to
    // themselves.
    std::vector<int64> permutation(allocation_location.shape.rank());
    absl::c_iota(permutation, 0);
    FindConsumers(
        allocation_location.location, allocation_location.location.instruction,
        allocation_location.location.flattened_output_tuple_index, permutation);
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
