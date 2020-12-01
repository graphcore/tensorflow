/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUSE_OPS_EARLY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUSE_OPS_EARLY_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/single_hlo_matcher.h"

namespace xla {

namespace poplarplugin {

// The purpose of this pass is to extract patterns created by tf2xla *before*
// other passes attempt to optimize the graph and hence making the pattern
// matching more difficult
class FuseOpsEarly : public SingleHloMatcher {
 public:
  FuseOpsEarly(struct CompilerAnnotations& annotations);

  ~FuseOpsEarly() override = default;

  absl::string_view name() const override { return "fuse-ops-early"; }
};

}  // namespace poplarplugin
}  // namespace xla

#endif
