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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REDUNDANT_TRIANGULAR_MASK_REMOVER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REDUNDANT_TRIANGULAR_MASK_REMOVER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/core/framework/types.h"

namespace xla {
namespace poplarplugin {

// A pass which removes masking from cholesky and triangular-solve. In the
// TensorFlow implementation of these ops, the zero part of the result can
// contain junk values which often need masking. This is not the case with the
// poplibs implementations of these ops, so we can safely remove masking.
// Masking is added internally by tf.linalg.cholesky so this pass is required to
// remove it in that case.
class RedundantTriangularMaskRemover : public HloMatcher {
 public:
  explicit RedundantTriangularMaskRemover(
      struct CompilerAnnotations& annotations);

  ~RedundantTriangularMaskRemover() override = default;
  absl::string_view name() const override {
    return "redundant-triangular-mask-remover";
  }

 private:
  StatusOr<bool> HandleMatch(HloMatcherMatched& match,
                             const absl::optional<int64_t>) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_REDUNDANT_TRIANGULAR_MASK_REMOVER_H_
