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

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/post_order_scheduler.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"

#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {

IpuSchedulerAlgorithm CreatePostOrderScheduler() {
  return [](HloComputation* computation, const HloPoplarDataflowAnalysis&,
            const absl::flat_hash_map<const HloComputation*, int64_t>&) {
    HloInstructionSequence sequence(computation->MakeInstructionPostOrder());
    return sequence;
  };
}

}  // namespace poplarplugin
}  // namespace xla
