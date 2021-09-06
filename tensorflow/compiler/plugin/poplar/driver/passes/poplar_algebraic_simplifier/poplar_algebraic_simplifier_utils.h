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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
class PoplarAlgebraicSimplifier;

namespace poplarplugin::algebraic_simplifier::util {
bool IsAll(const HloInstruction* op, int8 value);

bool IsAllFloat(const HloInstruction* op, float value);

bool IsAllFpConstantPowerOf2(const HloInstruction* op);

HloInstruction* BitcastingOperandOfReshapeOrCopyChainHelper(
    HloInstruction* instr, HloInstruction* operand);

bool IsUnstridedSlice(const HloInstruction* hlo);

HloInstruction* PreserveFrontendAttributesIfNeeded(
    HloInstruction* new_inst, const HloInstruction* old_inst);

StatusOr<HloInstruction*> PreserveFrontendAttributesIfNeeded(
    StatusOr<HloInstruction*> new_inst, const HloInstruction* old_inst);

bool IsGlobalAllReduceWithSum(const HloInstruction* all_reduce);
}  // namespace poplarplugin::algebraic_simplifier::util

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_POPLAR_ALGEBRAIC_SIMPLIFIER_POPLAR_ALGEBRAIC_SIMPLIFIER_UTILS_H_
