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

#include <absl/memory/memory.h>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloExecutionCounter::HloExecutionCounter()
    : HloPoplarInstruction(ShapeUtil::MakeShape(S32, {}), {},
                           PoplarOp::ExecutionCounter) {}

std::vector<std::string> HloExecutionCounter::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> HloExecutionCounter::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const>,
    HloCloneContext*) const {
  CHECK_EQ(shape.rank(), 0);
  CHECK_EQ(shape.element_type(), S32);
  return CreateExecutionCounter();
}

std::unique_ptr<HloInstruction> CreateExecutionCounter() {
  return absl::make_unique<HloExecutionCounter>();
}

}  // namespace poplarplugin
}  // namespace xla
