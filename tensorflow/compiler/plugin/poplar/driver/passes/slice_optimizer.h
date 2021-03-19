/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_OPTIMIZER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/core/framework/types.h"

namespace xla {
namespace poplarplugin {

/**
 * A pass which tries to minimize the maximum livness by spotting concatenated
 * operations being a RHS of an elementwise operation.
 * Also spots if either/both the LHS or the RHS is multiplied by a scalar and
 * applies that multiplication serially as well.
 *
 * Works on the following patterns:
 * * f(A * Broadcast(x), Concat(B, C, ...) * Broadcast(y)) =>
 *    SliceApplyaXbY(SliceApplyaXbY(A, B, x, y), C, x, y)
 * * f(A, Concat(B, C, ...) * Broadcast(y)) =>
 *    SliceApplyabY(SliceApplyabY(A, B, y), C, y)
 * * f(A * Broadcast(x), Concat(B, C, ...)) =>
 *    SliceApplyaXb(SliceApplyaXb(A, B, x), C, x)
 * * f(A, Concat(B, C, ...)) =>
 *    SliceApply(SliceApply(A, B, y), C)
 *
 * Note that the patterns with any multiplicative scalars are currently limited
 * to only add and subtract elementwise operations.
 */
class SliceOptimizer : public HloMatcher {
 public:
  explicit SliceOptimizer(struct CompilerAnnotations& annotations);

  absl::string_view name() const override { return "slice-optimizer"; }

  static StatusOr<HloInstruction*> ConvertToSliceApply(
      HloOpcode opcode, HloInstruction* const input,
      HloInstruction* const update);

  static StatusOr<HloInstruction*> ConvertToSliceApplyabY(
      HloOpcode opcode, HloInstruction* const input,
      HloInstruction* const update, HloInstruction* const scale_update);

  static StatusOr<HloInstruction*> ConvertToSliceApplyaXb(
      HloOpcode opcode, HloInstruction* const input,
      HloInstruction* const update, HloInstruction* const scale_input);

  static StatusOr<HloInstruction*> ConvertToSliceApplyaXbY(
      HloOpcode opcode, HloInstruction* const input,
      HloInstruction* const update, HloInstruction* const scale_input,
      HloInstruction* const scale_update);

 private:
  StatusOr<bool> HandleMatch(HloMatcherMatched& match,
                             const absl::optional<int64>) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_OPTIMIZER_H_
