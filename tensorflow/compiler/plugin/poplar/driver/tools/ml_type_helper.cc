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
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

Status SetInstructionMLType(HloInstruction* inst, const MLType& type) {
  auto status_or = inst->backend_config<PoplarBackendConfig>();
  if (status_or.ok()) {
    auto backend_config = status_or.ValueOrDie();
    backend_config.set_ml_type(type);
    TF_RETURN_IF_ERROR(inst->set_backend_config(backend_config));
  }
  return status_or.status();
}

StatusOr<MLType> GetMLType(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  return backend_config.ml_type();
}

StatusOr<std::string> GetMLTypeAsString(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(MLType type, GetMLType(inst));
  return MLType_Name(type);
}

bool IsTrainingForward(const HloInstruction* inst) {
  return GetMLType(inst).ValueOrDie() == MLType::TRAINING_FWD;
}

bool IsTrainingBackward(const HloInstruction* inst) {
  return GetMLType(inst).ValueOrDie() == MLType::TRAINING_BWD;
}

bool IsTrainingWU(const HloInstruction* inst) {
  return GetMLType(inst).ValueOrDie() == MLType::TRAINING_WU;
}

StatusOr<absl::flat_hash_map<const HloInstruction*, MLType>>
GetAllNotNoneMlTypes(const HloModule* module) {
  absl::flat_hash_map<const HloInstruction*, MLType> result;
  for (HloComputation* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(MLType type, GetMLType(inst));
      if (type != MLType::NONE) {
        result[inst] = type;
      }
    }
  }
  return result;
}

}  // namespace poplarplugin
}  // namespace xla
