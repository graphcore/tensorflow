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

#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_nan.h"

#include <stdlib.h>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> ConstantNaN::Run(HloModule* module) {
  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kConstant) {
        continue;
      }

      const Shape& shape = inst->shape();
      const PrimitiveType& type = shape.element_type();
      const Literal& literal = inst->literal();

      if (ShapeUtil::ElementIsFloating(shape)) {
        int64 num_elements = ShapeUtil::ElementsIn(shape);
        TF_ASSIGN_OR_RETURN(Literal literal_flat,
                            literal.Reshape({num_elements}));

        for (int64 i = 0; i < num_elements; i++) {
          bool error = false;
          switch (type) {
            case F16: {
              Eigen::half value = literal_flat.Get<Eigen::half>({i});
              error |= Eigen::half_impl::isnan(value);
              break;
            }
            case F32: {
              float value = literal_flat.Get<float>({i});
              error |= std::isnan(value);
              break;
            }
            default:
              break;
          }
          if (error) {
            return xla::FailedPrecondition(
                "Detected a NaN constant instruction `%s` in the Hlo graph "
                "during graph construction. To disable this error message use "
                "`TF_POPLAR_FLAGS=\"--allow_nans\"`.",
                inst->ToString().c_str());
          }
        }
      }
    }
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
