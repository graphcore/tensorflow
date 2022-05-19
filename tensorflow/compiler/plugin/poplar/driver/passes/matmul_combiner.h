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

#ifndef TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_MATMUL_COMBINER_H_
#define TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_MATMUL_COMBINER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/core/framework/types.h"

namespace xla {
namespace poplarplugin {

// A pass which combines multiple matmul instructions.
class MatmulCombiner : public HloMatcher {
 public:
  explicit MatmulCombiner(struct CompilerAnnotations& annotations);

  ~MatmulCombiner() override = default;
  absl::string_view name() const override { return "matmul-combiner"; }

 private:
  StatusOr<bool> HandleMatch(HloMatcherMatched& match,
                             const absl::optional<int64_t>) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_MATMUL_COMBINER_H_
