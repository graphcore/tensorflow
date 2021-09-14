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

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.h"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla::poplarplugin::algebraic_simplifier::dot {
namespace m = match;

namespace {
// Transposes a dot operand such that the batch dimensions are the most major,
// and the contracting dimensions are most minor.
StatusOr<HloInstruction*> NormalizeDotOperandToBatchMajorAndContractingMinor(
    HloInstruction* dot_operand, absl::Span<const int64> batch_dimensions,
    absl::Span<const int64> contracting_dimensions) {
  std::vector<int64> transpose_dimensions(batch_dimensions.begin(),
                                          batch_dimensions.end());
  for (int64 i = 0; i < dot_operand->shape().rank(); ++i) {
    if (!(absl::c_linear_search(batch_dimensions, i) ||
          absl::c_linear_search(contracting_dimensions, i))) {
      transpose_dimensions.push_back(i);
    }
  }
  transpose_dimensions.insert(transpose_dimensions.end(),
                              contracting_dimensions.begin(),
                              contracting_dimensions.end());
  if (absl::c_is_sorted(transpose_dimensions)) {
    return dot_operand;
  }
  return util::PreserveFrontendAttributesIfNeeded(
      MakeTransposeHlo(dot_operand, transpose_dimensions), dot_operand);
}
}  // namespace

StatusOr<HloInstruction*> OptimizeDotOfConcat(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
    return nullptr;
  }

  const int64 lhs_contracting_dim = dnums.lhs_contracting_dimensions(0);
  const int64 rhs_contracting_dim = dnums.rhs_contracting_dimensions(0);
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * optimized_lhs_concat,
      OptimizeDotOfConcatHelper(visitor, *dot, lhs, lhs_contracting_dim, rhs,
                                rhs_contracting_dim, /*swapped=*/false));
  if (optimized_lhs_concat) {
    return optimized_lhs_concat;
  }

  return OptimizeDotOfConcatHelper(visitor, *dot, rhs, rhs_contracting_dim, lhs,
                                   lhs_contracting_dim, /*swapped=*/true);
}

StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
    AlgebraicSimplifierVisitor* visitor, const HloInstruction& dot,
    HloInstruction* lhs, int64 lhs_contracting_dim, HloInstruction* rhs,
    int64 rhs_contracting_dim, bool swapped) {
  bool can_optimize = lhs->opcode() == HloOpcode::kConcatenate &&
                      lhs->concatenate_dimension() == lhs_contracting_dim &&
                      rhs->opcode() == HloOpcode::kConstant;
  if (!can_optimize) {
    return nullptr;
  }

  // We're replacing this:
  //
  //   +-----+-----+-----+      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_0        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   | L_0 | L_1 | L_2 |   *  |        R_1        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_2        |
  //   |     |     |     |      |                   |
  //   +-----+-----+-----+      +-------------------+
  //
  // with this:
  //
  // [Sum over i]
  //
  //   +-----+     +-------------------+
  //   |     |     |                   |
  //   |     |  *  |        R_i        |
  //   |     |     |                   |
  //   |     |     +-------------------+
  //   |     |
  //   | L_i |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   +-----+
  //
  // where the LHS is a concatenate operation (so we can "split" the LHS tensor
  // for free) and the RHS is a constant tensor (and thus can be split at
  // compile time).  In the future, we may also want to do this when both the
  // LHS and the RHS are concatenate operations that line up along the dimension
  // being contracted over.
  //
  // We should be able to generalize this transform to work on a non-constant
  // RHS when/if we have in-place slices or support input-fusing slices into
  // Dots.

  // Dimension numbers for the new dot instructions we'll create (L_i * R_i in
  // the diagram above).
  DotDimensionNumbers new_dot_dnums;
  new_dot_dnums.add_lhs_contracting_dimensions(swapped ? rhs_contracting_dim
                                                       : lhs_contracting_dim);
  new_dot_dnums.add_rhs_contracting_dimensions(swapped ? lhs_contracting_dim
                                                       : rhs_contracting_dim);

  // Here we use the MKN notation, where the contracted dimension has K
  // elements and the two non-contracted dimensions have M and N elements.
  HloInstruction* add_result = nullptr;
  int64 rhs_contracting_dim_offset = 0;
  int64 n = rhs->shape().dimensions(1 - rhs_contracting_dim);
  for (HloInstruction* concat_op : lhs->operands()) {
    int64 sub_k = concat_op->shape().dimensions(lhs_contracting_dim);
    Shape rhs_slice_shape(rhs->shape());
    rhs_slice_shape.set_dimensions(rhs_contracting_dim, sub_k);
    visitor->simplifier_->UpdateLayout(&rhs_slice_shape);

    std::array<int64, 2> start_indices;
    start_indices[rhs_contracting_dim] = rhs_contracting_dim_offset;
    start_indices[1 - rhs_contracting_dim] = 0;

    std::array<int64, 2> limit_indices;
    limit_indices[rhs_contracting_dim] = rhs_contracting_dim_offset + sub_k;
    limit_indices[1 - rhs_contracting_dim] = n;

    HloInstruction* rhs_slice = util::PreserveFrontendAttributesIfNeeded(
        visitor->computation_->AddInstruction(HloInstruction::CreateSlice(
            rhs_slice_shape, rhs, /*start_indices=*/start_indices,
            /*limit_indices=*/limit_indices, /*strides=*/{1, 1})),
        rhs);

    // TODO(b/69062148): We can get rid of `swapped` once all backends support
    // "non-canonical" contraction dimensions (that contracts dimension 1 of the
    // LHS with dimension 0 of the RHS).  But for now we keep the same
    // contraction dimensions as the incoming dot operation to ensure the new
    // dot operations can be lowered.
    HloInstruction *new_dot_lhs, *new_dot_rhs;
    if (swapped) {
      new_dot_lhs = rhs_slice;
      new_dot_rhs = concat_op;
    } else {
      new_dot_lhs = concat_op;
      new_dot_rhs = rhs_slice;
    }

    auto* new_dot = util::PreserveFrontendAttributesIfNeeded(
        visitor->computation_->AddInstruction(
            HloInstruction::CreateDot(dot.shape(), new_dot_lhs, new_dot_rhs,
                                      new_dot_dnums, dot.precision_config())),
        &dot);

    if (add_result) {
      add_result = util::PreserveFrontendAttributesIfNeeded(
          visitor->computation_->AddInstruction(HloInstruction::CreateBinary(
              dot.shape(), HloOpcode::kAdd, add_result, new_dot)),
          &dot);
    } else {
      add_result = new_dot;
    }

    rhs_contracting_dim_offset += sub_k;
  }

  return add_result;
}

StatusOr<HloInstruction*> OptimizeDotOfGather(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
    VLOG(10) << "DotOfGather: Can only optimize 2D, non-batch dot operations.";
    return nullptr;
  }

  // Optimize either dot(DS(ctA), ctB)) or dot(ctB, DS(ctA)).
  // Currently a Gather is a DynamicSlice.
  auto is_dynamic_slice_constant_combination =
      [](HloInstruction* a, HloInstruction* b, int a_contracting_dimension) {
        // First operand is a DynamicSlice(Constant).
        if (a->opcode() != HloOpcode::kDynamicSlice) {
          return false;
        }
        auto* dynamic_slice_op = a->operand(0);
        if (dynamic_slice_op->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // Second operand is a Constant.
        if (b->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // The DynamicSlice output is a vector.
        const Shape& dynamic_slice_shape = a->shape();
        if (dynamic_slice_shape.dimensions(1 - a_contracting_dimension) != 1) {
          return false;
        }
        // Constant size is the same before and after slice in the contracting
        // dimension, otherwise we either must precompute for all possible slice
        // indices or dot is invalid.
        const Shape& dynamic_slice_op_shape = dynamic_slice_op->shape();
        if (dynamic_slice_op_shape.dimensions(a_contracting_dimension) !=
            dynamic_slice_shape.dimensions(a_contracting_dimension)) {
          return false;
        }
        return true;
      };

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  int lhs_contracting_dimension = dnums.lhs_contracting_dimensions(0);
  int rhs_contracting_dimension = dnums.rhs_contracting_dimensions(0);

  if (!is_dynamic_slice_constant_combination(
          lhs, rhs, /*a_contracting_dimension=*/lhs_contracting_dimension) &&
      !is_dynamic_slice_constant_combination(
          rhs, lhs, /*a_contracting_dimension=*/rhs_contracting_dimension)) {
    VLOG(10) << "DotOfGather: Can only optimize dot(DS(ctA), ctB)) or "
                "dot(ctB, DS(ctA)), where the two constants have equal "
                "contracting dimensions.";
    return nullptr;
  }

  // LHS is DynamicSlice:
  // input: dot(DS(ctA), ctB))
  // where DS(ctA) = DS({M x K}, {start, 0}, {1, K}) and ctB = {K x N}.
  // => input dimensions: dot({1 x K}, {K x N}) => {1 x N}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {start, 0}, {1, N}) => {1 x N}.

  // RHS is DynamicSlice:
  // input: dot(ctA, DS(ctB))
  // where ctA = {M x K} and DS(ctB) = DS({K x N}, {0, start}, {K, 1}).
  // => input dimensions: dot({M x K}, {K x 1}) => {M x 1}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {0, start}, {M, 1}) => {M x 1}.

  bool lhs_is_dynamic_slice = lhs->opcode() == HloOpcode::kDynamicSlice;
  HloDynamicSliceInstruction* dynamic_slice =
      lhs_is_dynamic_slice ? Cast<HloDynamicSliceInstruction>(lhs)
                           : Cast<HloDynamicSliceInstruction>(rhs);

  // ctA:
  HloInstruction* left_operand =
      lhs_is_dynamic_slice ? lhs->mutable_operand(0) : lhs;
  // ctB:
  HloInstruction* right_operand =
      lhs_is_dynamic_slice ? rhs : rhs->mutable_operand(0);
  // Build ctA x ctB.
  const int m = left_operand->shape().dimensions(1 - lhs_contracting_dimension);
  const int n =
      right_operand->shape().dimensions(1 - rhs_contracting_dimension);
  auto memoized_shape =
      ShapeUtil::MakeShape(dot->shape().element_type(), {m, n});
  visitor->simplifier_->UpdateLayout(&memoized_shape);
  auto* memoized_inst = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(
          HloInstruction::CreateDot(memoized_shape, left_operand, right_operand,
                                    dnums, dot->precision_config())),
      dot);
  // Get pair {start, 0} or {0, start}.
  // Position of start:
  int index_of_non_zero_start = lhs_is_dynamic_slice
                                    ? 1 - lhs_contracting_dimension
                                    : 1 - rhs_contracting_dimension;
  // Position of zero:
  int index_of_zero_start = 1 - index_of_non_zero_start;

  // Slice out start and 0 components and reorder if necessary.
  auto indices_type = dynamic_slice->operand(1)->shape().element_type();
  Shape s_shape = ShapeUtil::MakeShape(indices_type, {1});
  visitor->simplifier_->UpdateLayout(&s_shape);
  Shape d_shape = ShapeUtil::MakeShape(indices_type, {2});
  visitor->simplifier_->UpdateLayout(&d_shape);
  HloInstruction* non_zero_start =
      dynamic_slice->mutable_operand(1 + index_of_non_zero_start);
  HloInstruction* zero_start =
      dynamic_slice->mutable_operand(1 + index_of_zero_start);
  std::vector<HloInstruction*> new_start_indices;
  if (lhs_is_dynamic_slice) {
    new_start_indices = {non_zero_start, zero_start};
  } else {
    new_start_indices = {zero_start, non_zero_start};
  }

  // Build DynamicSlice(ctA x ctB).
  const int new_slice_m = lhs_is_dynamic_slice ? 1 : m;
  const int new_slice_n = lhs_is_dynamic_slice ? n : 1;
  auto* memoized_lookup = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
          dot->shape(), memoized_inst, new_start_indices,
          {new_slice_m, new_slice_n})),
      dot);

  return memoized_lookup;
}

// This function tries to transform
//   dot(reshape(transpose(A)), Const) to
//   dot(reshape(A), reshape(transpose(reshape(Const)))),
// so that the reshape and transpose on the Const side can be constant folded.
//
// The basic idea is that since the accumulation in the dot operation is
// associative, so as long as we permute the elements of the contracting
// dimensions on both sides of the dot in the same way, the result of the
// dot is not affected.
StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot) {
  // Canonicalize dot(<constant>, rhs) to dot(rhs, <constant>) to make the
  // remainder of this function easier.
  auto dnums = dot->dot_dimension_numbers();
  auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
  auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();
  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
    std::swap(lhs_contracting_dims, rhs_contracting_dims);
  }

  // Require single contracting dim to make the implementation easier to
  // track contracting dims.
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return nullptr;
  }

  // Pattern match Dot(reshape(transpose(input), constant))
  HloInstruction* reshape;
  HloInstruction* transpose;
  HloInstruction* input;
  HloInstruction* constant;
  if (!Match(lhs,
             m::Reshape(&reshape, m::Transpose(&transpose, m::Op(&input)))) ||
      !Match(rhs, m::Constant(&constant))) {
    return nullptr;
  }

  // Check that reshape squishes some dims into one dim and that this one
  // dim is the dot's lhs contracting dim. The size of unmodified_dims should
  // be N - 1, where N is the rank of the reshape output. This means that the
  // reshape squishes some dims into one dim. lhs contracting dim should not
  // be in unmodified_dims. This means that the squishing target dim is the
  // lhs contracting dim.
  auto unmodified_dims = ShapeUtil::DimensionsUnmodifiedByReshape(
      reshape->operand(0)->shape(), reshape->shape());
  CHECK_EQ(lhs_contracting_dims.size(), 1);
  if ((static_cast<int64>(unmodified_dims.size()) !=
       reshape->shape().rank() - 1) ||
      absl::c_any_of(unmodified_dims, [&](const std::pair<int64, int64>& p) {
        return p.second == lhs_contracting_dims[0];
      })) {
    return nullptr;
  }

  // Virtually pull the reshape into the dot so the dot operates on the
  // transpose, with "unsquished" lhs contracting dims.  The new contracting
  // dims are all of the dims that are modified by the reshape -- that is, every
  // dimension that's not in `unmodified_dims[i].first`.
  //
  // (We don't need to actually create a new dot instruction. We can just keep
  // track of lhs and lhs_contracting_dims.)
  absl::flat_hash_set<int64> unmodified_transpose_dims;
  for (const auto& pair : unmodified_dims) {
    unmodified_transpose_dims.insert(pair.first);
  }
  lhs_contracting_dims.Clear();
  for (int64 i = 0; i < transpose->shape().dimensions_size(); ++i) {
    if (!unmodified_transpose_dims.contains(i)) {
      lhs_contracting_dims.Add(i);
    }
  }
  // We require the "unsquished" lhs contracting dims to be consecutive.
  auto is_iota = [](absl::Span<const int64> dims) {
    return absl::c_adjacent_find(dims, [](const int64 a, const int64 b) {
             return (b != a + 1);
           }) == dims.end();
  };
  if (!is_iota(AsInt64Slice(lhs_contracting_dims))) {
    return nullptr;
  }
  lhs = lhs->mutable_operand(0);

  // Check that the transpose only permutes the contracting dims.
  const std::vector<int64>& transpose_dims = transpose->dimensions();
  for (int64 i = 0; i < static_cast<int64>(transpose_dims.size()); ++i) {
    if (transpose_dims[i] != i &&
        !absl::c_linear_search(lhs_contracting_dims, i)) {
      return nullptr;
    }
  }
  // Virtually pull the transpose into the dot. Now the dot is equivalent to
  // a new dot with "permuted" lhs contracting dims.
  std::vector<int64> permutation;
  for (auto dim : lhs_contracting_dims) {
    permutation.push_back(transpose_dims[dim] - lhs_contracting_dims[0]);
  }
  CHECK(IsPermutation(permutation, permutation.size()));
  auto new_lhs_contracting_dims =
      ComposePermutations(AsInt64Slice(lhs_contracting_dims), permutation);
  lhs_contracting_dims.Clear();
  for (auto dim : new_lhs_contracting_dims) {
    lhs_contracting_dims.Add(dim);
  }
  lhs = lhs->mutable_operand(0);

  // All checks are passed at this point.
  //
  // Transform lhs. Remove the transpose and reshape by sorting the lhs
  // contracting dims and squishing them into a single one. We don't actually
  // squish the lhs_contracting_dims here because we still need the unsquished
  // contracting dims to invert reshape and transpose.
  absl::c_sort(lhs_contracting_dims);
  lhs = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(
          HloInstruction::CreateReshape(reshape->shape(), lhs)),
      lhs);

  // Transform rhs. Say the input HLO is:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t1 = f32[2, 3, 2] transpose(t0) dimensions={0, 2, 1}
  //   t2 = f32[2, 6] reshape(t1)
  //   t3 = f32[6, 2] constant(...)
  //   dot = f32[2, 2] dot(t2, t3) lhs_contracting_dims={1},
  //                               rhs_contracting_dims={0}
  //
  // At this point in the function, we have decided that the second and third
  // dims of t0 can be switched to remove the transpose, and we have
  // "virtually decomposed" the input HLO to:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t2' = f32[2, 6] reshape(t0)
  //   t3' = f32[6, 2] ops-to-be-filled ...
  //   dot = f32[2, 2] dot(t2', t3') lhs_contracting_dims={1},
  //                                 rhs_contracting_dims={0}
  //
  // The rest of this function is to fill in the ops of t3'. To do this, we
  // unsquish the contracting dimensions in t3 and then apply the inverse of
  // the transpose from t1.

  // Invert reshape.
  CHECK_EQ(rhs_contracting_dims.size(), 1);
  auto rhs_unsquished_shape_dims = constant->shape().dimensions();
  auto it = rhs_unsquished_shape_dims.erase(rhs_unsquished_shape_dims.begin() +
                                            rhs_contracting_dims[0]);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_unsquished_shape_dims.insert(it,
                                          transpose->shape().dimensions(dim));
    ++it;
  }
  HloInstruction* rhs_reshape = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_unsquished_shape_dims),
          constant)),
      rhs);
  rhs = rhs_reshape;

  // Rhs reshape "unsquishes" the single rhs contracting dim into multiple dims.
  rhs_contracting_dims.Resize(lhs_contracting_dims.size(),
                              rhs_contracting_dims[0]);
  absl::c_iota(rhs_contracting_dims, rhs_contracting_dims[0]);

  // Invert transpose. First compute the shape.
  auto rhs_transpose_shape_dims = rhs_reshape->shape().dimensions();
  it = rhs_transpose_shape_dims.erase(
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_shape_dims.insert(it, input->shape().dimensions(dim));
    ++it;
  }
  // Then compute the transpose dims.
  std::vector<int64> rhs_transpose_dims(rhs_reshape->shape().rank());
  absl::c_iota(rhs_transpose_dims, 0);
  it = rhs_transpose_dims.erase(
      rhs_transpose_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  auto inverse_lhs_transpose_dims = InversePermutation(transpose_dims);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_dims.insert(it, inverse_lhs_transpose_dims[dim] -
                                           lhs_contracting_dims[0] +
                                           rhs_contracting_dims[0]);
    ++it;
  }
  HloInstruction* rhs_transpose = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_transpose_shape_dims),
          rhs_reshape, rhs_transpose_dims)),
      rhs_reshape);
  rhs = rhs_transpose;

  // Squish the multiple rhs contracting dims into a single one.
  rhs = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(
          HloInstruction::CreateReshape(constant->shape(), rhs)),
      rhs);

  // If we virtually swapped lhs and rhs, we need to swap it back before
  // creating new dot.
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
  }

  HloInstruction* new_dot = util::PreserveFrontendAttributesIfNeeded(
      visitor->computation_->AddInstruction(HloInstruction::CreateDot(
          dot->shape(), lhs, rhs, dnums, dot->precision_config())),
      dot);
  return new_dot;
}

namespace {
typedef google::protobuf::RepeatedField<google::protobuf::int64> RepeatedField;
std::unique_ptr<HloInstruction> CreateBroadcastCompatibleWithOperand(
    HloInstruction* operand_source, HloInstruction* operand_target,
    const RepeatedField& batch_dimensions,
    const RepeatedField& contracting_dimensions,
    uint64_t target_outer_dimensions) {
  // Initialize broadcast dimensions with batch dimensions starting from 0.
  std::vector<int64> broadcast_dimensions(batch_dimensions.size());
  absl::c_iota(broadcast_dimensions, 0);

  // Add the rest of the dimensions and fill the batch dimensions starting with
  // the number of batch dimensions + outer dimensions of the target operand.
  broadcast_dimensions.resize(operand_source->shape().rank());
  std::iota(broadcast_dimensions.begin() + batch_dimensions.size(),
            broadcast_dimensions.end(),
            batch_dimensions.size() + target_outer_dimensions);

  // Create a broadcast with the shape of the target operand and the previously
  // calculated dimensions.
  return HloInstruction::CreateBroadcast(operand_target->shape(),
                                         operand_source, broadcast_dimensions);
}

PrimitiveType DeterminePrimitiveTypeForDotProduct(HloInstruction* dot) {
  if (ShapeUtil::ElementIsFloating(dot->shape())) {
    return dot->shape().element_type() == F64 ? F64 : F32;
  } else {
    return dot->shape().element_type();
  }
}
}  // namespace

StatusOr<HloInstruction*> OptimizeDotStrengthReduction(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot) {
  if (!visitor->config_.enable_dot_strength()) {
    return nullptr;
  }

  const auto& dot_dimension_numbers = dot->dot_dimension_numbers();
  const auto& lhs_batch_dimensions =
      dot_dimension_numbers.lhs_batch_dimensions();
  const auto& lhs_contracting_dimensions =
      dot_dimension_numbers.lhs_contracting_dimensions();
  const auto& rhs_batch_dimensions =
      dot_dimension_numbers.rhs_batch_dimensions();
  const auto& rhs_contracting_dimensions =
      dot_dimension_numbers.rhs_contracting_dimensions();

  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  if (lhs_batch_dimensions.size() + lhs_contracting_dimensions.size() !=
          lhs->shape().rank() &&
      rhs_batch_dimensions.size() + rhs_contracting_dimensions.size() !=
          rhs->shape().rank()) {
    return nullptr;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          lhs, AsInt64Slice(lhs_batch_dimensions),
                          AsInt64Slice(lhs_contracting_dimensions)));

  if (!ShapeUtil::SameElementType(dot->shape(), new_lhs->shape())) {
    new_lhs = MakeConvertToHlo(new_lhs, dot->shape().element_type());
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          rhs, AsInt64Slice(rhs_batch_dimensions),
                          AsInt64Slice(rhs_contracting_dimensions)));

  if (!ShapeUtil::SameElementType(dot->shape(), new_rhs->shape())) {
    new_rhs = MakeConvertToHlo(new_rhs, dot->shape().element_type());
  }

  int64 lhs_outer_dimensions =
      lhs->shape().rank() -
      (lhs_batch_dimensions.size() + lhs_contracting_dimensions.size());
  int64 rhs_outer_dimensions =
      rhs->shape().rank() -
      (rhs_batch_dimensions.size() + rhs_contracting_dimensions.size());
  const auto outer_dimensions =
      std::max(rhs_outer_dimensions, lhs_outer_dimensions);

  // Sanity check that either lhs or rhs operand contains only batch and
  // contracting dimensions.
  CHECK(lhs_outer_dimensions == 0 || rhs_outer_dimensions == 0);

  // Create a broadcast for the operand with only batch and contracting
  // dimensions, making it compatible for a matmul with the other operand.
  if (rhs_outer_dimensions > 0) {
    auto broadcast = CreateBroadcastCompatibleWithOperand(
        new_lhs, new_rhs, lhs_batch_dimensions, lhs_contracting_dimensions,
        rhs_outer_dimensions);
    new_lhs = visitor->computation_->AddInstruction(std::move(broadcast));
  } else if (lhs_outer_dimensions > 0) {
    auto broadcast = CreateBroadcastCompatibleWithOperand(
        new_rhs, new_lhs, rhs_batch_dimensions, rhs_contracting_dimensions,
        lhs_outer_dimensions);
    new_rhs = visitor->computation_->AddInstruction(std::move(broadcast));
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                      MakeBinaryHlo(HloOpcode::kMultiply, new_lhs, new_rhs));

  // We want to reduce the source operand to its original rank.
  std::vector<int64> reduce_dimensions(lhs_contracting_dimensions.size());
  absl::c_iota(reduce_dimensions,
               outer_dimensions + lhs_batch_dimensions.size());

  const auto dot_type = DeterminePrimitiveTypeForDotProduct(dot);
  new_dot = visitor->AsType(new_dot, dot_type);
  new_dot = visitor->AddReduce(new_dot, reduce_dimensions, dot_type);
  new_dot = visitor->AsType(new_dot, dot->shape().element_type());

  return new_dot;
}
}  // namespace xla::poplarplugin::algebraic_simplifier::dot
