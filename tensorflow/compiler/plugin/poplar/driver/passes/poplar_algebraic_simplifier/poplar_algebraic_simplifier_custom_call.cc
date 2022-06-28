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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/normalise_image.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin::algebraic_simplifier::custom_call {
namespace {

Window MergePaddingIntoWindow(const PaddingConfig& padding,
                              const Window& window) {
  // Sanity check assumptions.
  CHECK(IsPoplibsPoolWindow(window));
  CHECK_EQ(padding.dimensions_size(), window.dimensions_size());

  Window new_window;
  for (int64_t dim = 0; dim != padding.dimensions_size(); ++dim) {
    auto& padding_dim = padding.dimensions(dim);
    auto& window_dim = window.dimensions(dim);

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
  return new_window;
}

bool SliceMatchesPad(const HloInstruction* slice,
                     const PaddingConfig& padding) {
  // Sanity check assumption.
  CHECK_EQ(slice->shape().rank(), padding.dimensions_size());

  // Check that the slice is the inverse of the pad.
  for (int64_t dim = 0; dim != padding.dimensions_size(); ++dim) {
    auto& padding_dim = padding.dimensions(dim);
    int64_t slice_low = slice->slice_starts(dim);
    int64_t slice_high =
        slice->operand(0)->shape().dimensions(dim) - slice->slice_limits(dim);

    // Note this function doesn't look at interior padding / slice stride.
    if (slice_low != padding_dim.edge_padding_low()) {
      return false;
    }
    if (slice_high != padding_dim.edge_padding_high()) {
      return false;
    }
  }
  return true;
}

bool IsStrided(const HloInstruction* slice) {
  return absl::c_any_of(slice->slice_strides(),
                        [](int64_t stride) { return stride != 1; });
}

}  // anonymous namespace

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

  HloInstruction* unpadded_operand = operand->mutable_operand(0);
  Window new_window = MergePaddingIntoWindow(padding, window);
  return CreateMaxPool(inst->shape(), unpadded_operand, new_window);
}

StatusOr<bool> FoldPaddingIntoMaxPoolGrad(HloMaxPoolGradInstruction* inst) {
  // Attempt to fold padding into max pool gradient.
  HloInstruction* operand = inst->mutable_operand(0);

  // Only padding with zeroes can be folded into max pool.
  if (!IsZeroPad(operand)) {
    return false;
  }

  const PaddingConfig& padding = operand->padding_config();
  const Window& window = inst->window();

  // The max pool grad op cannot simulate interior padding.
  if (HasInteriorPadding(padding)) {
    return false;
  }

  for (HloInstruction* user : inst->users()) {
    // Padding can only be folded into max pool grad if there is a corresponding
    // slice on the output which undoes the padding.
    // Strided slices cannot be optimised as the corresponding pad would use
    // interior padding which max pool grad cannot simulate.
    if (user->opcode() != HloOpcode::kSlice || IsStrided(user) ||
        !SliceMatchesPad(user, padding)) {
      return false;
    }
  }
  HloInstruction* unpadded_operand = operand->mutable_operand(0);
  Window new_window = MergePaddingIntoWindow(padding, window);

  HloInstruction* replacement =
      inst->parent()->AddInstruction(CreateMaxPoolGrad(
          unpadded_operand->shape(), unpadded_operand, inst->mutable_operand(1),
          inst->mutable_operand(2), new_window));
  for (HloInstruction* user : inst->users()) {
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(replacement));
  }
  return true;
}

StatusOr<std::unique_ptr<HloInstruction>> FoldCastIntoNormaliseImage(
    HloNormaliseImage* inst) {
  // NormaliseImage implicitly casts u8 inputs to f16.
  HloInstruction* operand = inst->mutable_operand(0);
  if (!IsU8ToF16Convert(operand)) {
    return {nullptr};
  }

  HloInstruction* uncasted_operand = operand->mutable_operand(0);
  return inst->CloneWithNewOperands(
      inst->shape(),
      {uncasted_operand, inst->mutable_operand(1), inst->mutable_operand(2)});
}

}  // namespace poplarplugin::algebraic_simplifier::custom_call
}  // namespace xla
