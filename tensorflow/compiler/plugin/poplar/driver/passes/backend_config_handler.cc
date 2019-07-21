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

#include "tensorflow/compiler/plugin/poplar/driver/passes/backend_config_handler.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

StatusOr<bool> BackendConfigHandler::Run(HloModule* module) {
  bool changed = false;
  for (auto comp : module->MakeNonfusionComputations()) {
    for (auto inst : comp->instructions()) {
      // Go through all the calls and check if they have the type field set from
      // tf2xla.
      if (inst->opcode() == HloOpcode::kCall) {
        auto backend_config_string = inst->raw_backend_config_string();
        if (backend_config_string.empty()) {
          continue;
        }
        // Go through all the calls and check if they have the type field set.
        IPUCustomKernelsUtil::AttributeMap attribute_map(backend_config_string);
        if (attribute_map.HasAttribute("type")) {
          TF_ASSIGN_OR_RETURN(std::string type_string,
                              attribute_map.GetAttributeAsString("type"));
          PoplarBackendConfig backend_config;
          auto* call_config = backend_config.mutable_call_config();
          PoplarBackendConfig::CallConfig::Type type;
          bool type_parsed =
              PoplarBackendConfig_CallConfig_Type_Parse(type_string, &type);
          if (!type_parsed) {
            return xla::FailedPrecondition("Could not parse the call type.");
          }
          call_config->set_type(type);
          inst->set_backend_config(backend_config);
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
