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

#include "tensorflow/compiler/plugin/poplar/driver/passes/wide_const_finder.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

WideConstFinder::WideConstFinder() {}

StatusOr<bool> WideConstFinder::Run(HloModule* module) {
  std::vector<HloComputation*> comps = module->MakeComputationPostOrder();

  for (auto* comp : comps) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (!inst->shape().IsToken()) {
        if (inst->IsConstant() && ShapeUtil::ElementsIn(inst->shape()) > 1) {
          const Literal& literal = inst->literal();
          if (literal.IsAll(0)) {
            const auto element_type = inst->shape().element_type();
            auto zero = LiteralUtil::Zero(element_type).Clone();
            HloInstruction* c = comp->AddInstruction(
                HloInstruction::CreateConstant(std::move(zero)));
            HloInstruction* b = comp->AddInstruction(
                HloInstruction::CreateBroadcast(inst->shape(), c, {}));

            if (!inst->ReplaceAllUsesWith(b).ok()) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
