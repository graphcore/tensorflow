/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_custom_call.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin::algebraic_simplifier::custom_call {

StatusOr<bool> ElideStatefulGradientAccumulate(
    HloStatefulGradientAccumulate* inst) {
  // We elide gradient accumulation ops with `num_mini_batches=1`.
  if (inst->MiniBatchesToAccumulate() != 1) {
    return false;
  }

  TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(inst->mutable_operand(0)));
  return true;
}

StatusOr<std::unique_ptr<HloInstruction>> FoldPaddingIntoMaxPool(
    HloMaxPoolInstruction* inst) {
  // Attempt to fold padding into max pool.
  HloInstruction* operand = inst->mutable_operand(0);

  // Only padding with zeroes can be folded into max pool.
  if (!IsZeroPad(operand)) {
    return {nullptr};
  }

  const PaddingConfig& padding = operand->padding_config();
  const Window& window = inst->window();

  // The max pool op cannot simulate interior padding.
  if (HasInteriorPadding(padding)) {
    return {nullptr};
  }

  // Sanity check assumptions.
  CHECK(IsPoplibsPoolWindow(window));
  CHECK_EQ(padding.dimensions_size(), window.dimensions_size());

  Window new_window;
  for (int64_t i = 0; i != padding.dimensions_size(); ++i) {
    auto& padding_dim = padding.dimensions(i);
    auto& window_dim = window.dimensions(i);

    int64_t padding_low =
        padding_dim.edge_padding_low() + window_dim.padding_low();
    int64_t padding_high =
        padding_dim.edge_padding_high() + window_dim.padding_high();

    auto* new_window_dim = new_window.add_dimensions();
    new_window_dim->set_size(window_dim.size());
    new_window_dim->set_stride(window_dim.stride());
    new_window_dim->set_padding_low(padding_low);
    new_window_dim->set_padding_high(padding_high);
  }

  HloInstruction* unpadded_operand = operand->mutable_operand(0);
  return CreateMaxPool(inst->shape(), unpadded_operand, new_window);
}

}  // namespace poplarplugin::algebraic_simplifier::custom_call
}  // namespace xla
