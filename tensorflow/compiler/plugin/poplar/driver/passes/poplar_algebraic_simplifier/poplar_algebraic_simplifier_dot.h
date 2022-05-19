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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_DOT_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_DOT_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
class AlgebraicSimplifierVisitor;

namespace poplarplugin::algebraic_simplifier::dot {
StatusOr<HloInstruction*> OptimizeDotOfConcat(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
    AlgebraicSimplifierVisitor* visitor, const HloInstruction& dot,
    HloInstruction* lhs, int64_t lhs_contracting_dim, HloInstruction* rhs,
    int64_t rhs_contracting_dim, bool swapped);

StatusOr<HloInstruction*> OptimizeDotOfGather(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

StatusOr<HloInstruction*> OptimizeDotStrengthReduction(
    AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);
}  // namespace poplarplugin::algebraic_simplifier::dot
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_DOT_H_
