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

// HloInstruction extensions

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_EXTENSIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_EXTENSIONS_H_

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_location.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/str_util.h"

namespace xla {
namespace poplarplugin {

struct FindConsumersExtensionParams {
  FindConsumersExtensionParams() = delete;
  const TensorLocation& src;
  const HloInstruction* tgt;
  int64 index;
  int64 op_index;
  absl::optional<std::vector<int64>> permutation;
};

enum class DoFindConsumers {
  UNSPECIFIED,
  FALSE,
  TRUE,
};

struct FindConsumersExtensionResults {
  DoFindConsumers do_find_consumers;
  const HloInstruction* tgt;
  int64 index;
  absl::optional<std::vector<int64>> permutation;

  FindConsumersExtensionResults()
      : do_find_consumers(DoFindConsumers::UNSPECIFIED),
        tgt(nullptr),
        index(-1),
        permutation(absl::nullopt) {}

  FindConsumersExtensionResults(bool do_find_consumers,
                                const HloInstruction* tgt, int64 index,
                                absl::optional<std::vector<int64>> permutation)
      : do_find_consumers(do_find_consumers ? DoFindConsumers::TRUE
                                            : DoFindConsumers::FALSE),
        tgt(tgt),
        index(index),
        permutation(permutation) {}

  static FindConsumersExtensionResults DoNotFindConsumers() {
    return FindConsumersExtensionResults(false, nullptr, -1, absl::nullopt);
  }

  bool operator==(const FindConsumersExtensionResults& rhs) const {
    return do_find_consumers == rhs.do_find_consumers && tgt == rhs.tgt &&
           index == rhs.index &&
           permutation.has_value() == rhs.permutation.has_value() &&
           (!permutation ||
            absl::c_equal(permutation.value(), rhs.permutation.value()));
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_HLO_EXTENSIONS_H_
