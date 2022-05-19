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
#ifndef TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_UTIL_H_
#define TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_UTIL_H_

#include <tuple>
#include <vector>
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {
class DotDimensionNumbers;

namespace poplarplugin {

// Return a permutation vector to convert the input shape into
// [Batch..., M..., Contracting...]
std::vector<int64_t> LeftMatMulPermutations(
    const Shape& shape, const DotDimensionNumbers& dot_dims);
// Return a permutation vector to convert the input shape into
// [Batch..., M..., Contracting...]
std::vector<unsigned> LeftMatMulPermutations(
    const absl::Span<const size_t>& shape, const DotDimensionNumbers& dot_dims);

// Return a permutation vector to convert the input shape into
// [Batch..., Contracting..., N...]
std::vector<int64_t> RightMatMulPermutations(
    const Shape& shape, const DotDimensionNumbers& dot_dims);

// Return a permutation vector to convert the input shape into
// [Batch..., Contracting..., N...]
std::vector<unsigned> RightMatMulPermutations(
    const absl::Span<const size_t>& shape, const DotDimensionNumbers& dot_dims);

// Collapse [Batch..., M..., Contracting...] down to a 3D shape
// [Batch, M, Contracting]
Shape LeftMatMulPackShape(const Shape& shape,
                          const DotDimensionNumbers& dot_dims);

// Collapse [Batch..., M..., Contracting...] down to a 3D shape
// [Batch, M, Contracting]
std::vector<size_t> LeftMatMulPackShape(const absl::Span<const size_t>& shape,
                                        const DotDimensionNumbers& dot_dims);

// Collapse [Batch..., Contracting..., N...] down to a 3D shape
// [Batch, Contracting, N]
Shape RightMatMulPackShape(const Shape& shape,
                           const DotDimensionNumbers& dot_dims);

// Collapse [Batch..., Contracting..., N...] down to a 3D shape
// [Batch, Contracting, N]
std::vector<size_t> RightMatMulPackShape(const absl::Span<const size_t>& shape,
                                         const DotDimensionNumbers& dot_dims);

// The LHS XLA shapes need to be shuffled and collapsed to [Batch, M,
// Contracting] This function returns the final shape, shuffled shape and
// permutations.
std::tuple<Shape, Shape, std::vector<int64_t>> LeftMatMulPrepare(
    const Shape& shape, const DotDimensionNumbers& dot_dims);

// The RHS XLA shapes need to be shuffled and collapsed to [Batch, Contracting,
// N] This function returns the final shape, shuffled shape and permutations.
std::tuple<Shape, Shape, std::vector<int64_t>> RightMatMulPrepare(
    const Shape& shape, const DotDimensionNumbers& dot_dims);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_UTIL_H_
