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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DYNAMIC_SLICE_REPLACER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DYNAMIC_SLICE_REPLACER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"

namespace xla {

class HloDynamicIndexInstruction;

namespace poplarplugin {

// Replace dynamicSlice with multiSlice.
class DynamicSliceReplacer : public HloMatcher {
 public:
  explicit DynamicSliceReplacer(CompilerAnnotations& annotations);

  absl::string_view name() const override { return "dynamic-slice-replacer"; }

 private:
  StatusOr<bool> HandleMatch(HloMatcherMatched& match,
                             const absl::optional<int64_t> shard) override;

  StatusOr<bool> HandleDynamicUpdateAdd(HloInstruction* inst) const;
  StatusOr<bool> HandleDynamicSlice(HloInstruction* inst) const;
  StatusOr<bool> HandleDynamicUpdate(HloInstruction* inst) const;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DYNAMIC_SLICE_REPLACER_H_
