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
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"

#include <functional>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
namespace {
template <typename Permutation, typename Dimension>
std::vector<Permutation> LeftMatMulPermutationsInternal(
    const absl::Span<const Dimension>& shape,
    const DotDimensionNumbers& dot_dims) {
  std::vector<Permutation> permutations;
  const auto lhs_reduction_dimensions = dot_dims.lhs_contracting_dimensions();
  const auto lhs_batch_dimensions = dot_dims.lhs_batch_dimensions();

  // Shuffle LHS to [Batch..., M..., Contracting...]
  permutations.reserve(shape.size());

  absl::c_copy(lhs_batch_dimensions, std::back_inserter(permutations));
  for (Permutation i = 0; i < static_cast<Permutation>(shape.size()); ++i) {
    if (absl::c_find(lhs_reduction_dimensions, i) ==
            lhs_reduction_dimensions.end() &&
        absl::c_find(lhs_batch_dimensions, i) == lhs_batch_dimensions.end()) {
      permutations.push_back(i);
    }
  }
  absl::c_copy(lhs_reduction_dimensions, std::back_inserter(permutations));
  return permutations;
}

template <typename Permutation, typename Dimension>
std::vector<Permutation> RightMatMulPermutationsInternal(
    const absl::Span<const Dimension>& shape,
    const DotDimensionNumbers& dot_dims) {
  std::vector<Permutation> permutations;
  const auto rhs_batch_dimensions = dot_dims.rhs_batch_dimensions();
  const auto rhs_reduction_dimensions = dot_dims.rhs_contracting_dimensions();

  // Shuffle RHS to [Batch..., Contracting..., N...]
  permutations.reserve(shape.size());

  absl::c_copy(rhs_batch_dimensions, std::back_inserter(permutations));
  absl::c_copy(rhs_reduction_dimensions, std::back_inserter(permutations));
  for (Permutation i = 0; i < static_cast<Permutation>(shape.size()); ++i) {
    if (absl::c_find(rhs_reduction_dimensions, i) ==
            rhs_reduction_dimensions.end() &&
        absl::c_find(rhs_batch_dimensions, i) == rhs_batch_dimensions.end()) {
      permutations.push_back(i);
    }
  }
  return permutations;
}

template <typename Dimension>
std::vector<Dimension> LeftMatMulPackShapeInternal(
    const absl::Span<const Dimension>& shape,
    const DotDimensionNumbers& dot_dims) {
  const auto lhs_reduction_dimensions = dot_dims.lhs_contracting_dimensions();
  const auto lhs_batch_dimensions = dot_dims.lhs_batch_dimensions();
  const auto lhs_shape_itr_begin = shape.begin();
  const auto lhs_shape_itr_a =
      lhs_shape_itr_begin + lhs_batch_dimensions.size();
  const auto lhs_shape_itr_end = shape.end();
  const auto lhs_shape_itr_b =
      lhs_shape_itr_end - lhs_reduction_dimensions.size();

  const auto lhs_b =
      std::accumulate(lhs_shape_itr_begin, lhs_shape_itr_a,
                      static_cast<Dimension>(1), std::multiplies<Dimension>());
  const auto lhs_m =
      std::accumulate(lhs_shape_itr_a, lhs_shape_itr_b,
                      static_cast<Dimension>(1), std::multiplies<Dimension>());
  const auto lhs_k =
      std::accumulate(lhs_shape_itr_b, lhs_shape_itr_end,
                      static_cast<Dimension>(1), std::multiplies<Dimension>());
  return {lhs_b, lhs_m, lhs_k};
}

template <typename Dimension>
std::vector<Dimension> RightMatMulPackShapeInternal(
    const absl::Span<const Dimension>& shape,
    const DotDimensionNumbers& dot_dims) {
  auto rhs_batch_dimensions = dot_dims.rhs_batch_dimensions();
  auto rhs_reduction_dimensions = dot_dims.rhs_contracting_dimensions();
  const auto rhs_shape_itr_begin = shape.begin();
  const auto rhs_shape_itr_a =
      rhs_shape_itr_begin + rhs_batch_dimensions.size();
  const auto rhs_shape_itr_b =
      rhs_shape_itr_a + rhs_reduction_dimensions.size();
  const auto rhs_shape_itr_end = shape.end();

  const Dimension rhs_b =
      std::accumulate(rhs_shape_itr_begin, rhs_shape_itr_a,
                      static_cast<int64>(1), std::multiplies<int64>());
  const Dimension rhs_k =
      std::accumulate(rhs_shape_itr_a, rhs_shape_itr_b, static_cast<int64>(1),
                      std::multiplies<int64>());
  const Dimension rhs_n =
      std::accumulate(rhs_shape_itr_b, rhs_shape_itr_end, static_cast<int64>(1),
                      std::multiplies<int64>());
  return {rhs_b, rhs_k, rhs_n};
}
}  // namespace

std::vector<int64> LeftMatMulPermutations(const Shape& shape,
                                          const DotDimensionNumbers& dot_dims) {
  // XLA and Poplar use opposite types of permutations so invert the
  // permutations vector.
  return InvertPermutations<int64>(
      LeftMatMulPermutationsInternal<int64>(shape.dimensions(), dot_dims));
}

std::vector<unsigned> LeftMatMulPermutations(
    const absl::Span<const size_t>& shape,
    const DotDimensionNumbers& dot_dims) {
  return LeftMatMulPermutationsInternal<unsigned>(shape, dot_dims);
}

std::vector<int64> RightMatMulPermutations(
    const Shape& shape, const DotDimensionNumbers& dot_dims) {
  // XLA and Poplar use opposite types of permutations so invert the
  // permutations vector.
  return InvertPermutations<int64>(
      RightMatMulPermutationsInternal<int64>(shape.dimensions(), dot_dims));
}

std::vector<unsigned> RightMatMulPermutations(
    const absl::Span<const size_t>& shape,
    const DotDimensionNumbers& dot_dims) {
  return RightMatMulPermutationsInternal<unsigned>(shape, dot_dims);
}

Shape LeftMatMulPackShape(const Shape& shape,
                          const DotDimensionNumbers& dot_dims) {
  return ShapeUtil::MakeShape(
      shape.element_type(),
      LeftMatMulPackShapeInternal(shape.dimensions(), dot_dims));
}

std::vector<size_t> LeftMatMulPackShape(const absl::Span<const size_t>& shape,
                                        const DotDimensionNumbers& dot_dims) {
  return LeftMatMulPackShapeInternal(shape, dot_dims);
}

Shape RightMatMulPackShape(const Shape& shape,
                           const DotDimensionNumbers& dot_dims) {
  return ShapeUtil::MakeShape(
      shape.element_type(),
      RightMatMulPackShapeInternal(shape.dimensions(), dot_dims));
}

std::vector<size_t> RightMatMulPackShape(const absl::Span<const size_t>& shape,
                                         const DotDimensionNumbers& dot_dims) {
  return RightMatMulPackShapeInternal(shape, dot_dims);
}

std::tuple<Shape, Shape, std::vector<int64>> LeftMatMulPrepare(
    const Shape& shape, const DotDimensionNumbers& dot_dims) {
  std::vector<int64> permutations = LeftMatMulPermutations(shape, dot_dims);
  // Collapse the LHS dimensions down to [Batch, M, Contracting]
  Shape shuffled_shape = ShapeUtil::PermuteDimensions(permutations, shape);
  return std::make_tuple(LeftMatMulPackShape(shuffled_shape, dot_dims),
                         shuffled_shape, permutations);
}

std::tuple<Shape, Shape, std::vector<int64>> RightMatMulPrepare(
    const Shape& shape, const DotDimensionNumbers& dot_dims) {
  std::vector<int64> permutations = RightMatMulPermutations(shape, dot_dims);
  // Collapse the LHS dimensions down to [Batch, M, Contracting]
  Shape shuffled_shape = ShapeUtil::PermuteDimensions(permutations, shape);
  return std::make_tuple(RightMatMulPackShape(shuffled_shape, dot_dims),
                         shuffled_shape, permutations);
}
}  // namespace poplarplugin
}  // namespace xla
