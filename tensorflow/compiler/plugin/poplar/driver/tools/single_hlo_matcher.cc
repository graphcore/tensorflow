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

#include "tensorflow/compiler/plugin/poplar/driver/tools/single_hlo_matcher.h"

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

bool SingleHloMatcher::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64> sharding_device) {
  auto& pattern = patterns_[match.pattern_idx];
  std::string name = op_prefix_ + pattern.GetType();
  HloInstruction* inst =
      OutlineExpressionFromComputation(match, name, sharding_device);

  PoplarOp tmp;

  if (name.find("_pop_op_") != std::string::npos) {
    name = name.substr(8);
  }
  *name.begin() = std::toupper(*name.begin());

  auto itr = name.find(".");
  if (itr != std::string::npos) {
    name = name.substr(0, itr);
  }

  bool parse = PoplarOp_Parse(name, &tmp);

  if (!parse) {
    LOG(FATAL)
        << "Matched fusion is NOT a known poplar operation registered with XLA "
        << name;
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
