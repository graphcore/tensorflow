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

#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace poplarplugin {

StatusOr<bool> ParsePoplarBackendConfig::Run(HloModule* module) {
  bool changed = false;
  StochasticRounding stochastic_rounding = NOT_SET;

  for (auto* comp : module->computations()) {
    for (auto instr : comp->instructions()) {
      auto attributes = instr->frontend_attributes();
      PoplarBackendConfig poplar_config;
      auto stochastic_rounding_attribute =
          attributes.map().find(FrontendAttributeId_Name(STOCHASTIC_ROUNDING));
      if (stochastic_rounding_attribute != attributes.map().end()) {
        if (!StochasticRounding_Parse(stochastic_rounding_attribute->second,
                                      &stochastic_rounding)) {
          return xla::FailedPrecondition(
              "Could not parse the stochastic rounding value");
        }
        changed = true;
      }
      // Check if the calls they have the type field set from tf2xla.
      if (instr->opcode() == HloOpcode::kCall) {
        auto call_config_type_attribute =
            attributes.map().find(FrontendAttributeId_Name(CALL_CONFIG_TYPE));
        if (call_config_type_attribute != attributes.map().end()) {
          PoplarBackendConfig::CallConfig::Type type;
          bool type_parsed = PoplarBackendConfig_CallConfig_Type_Parse(
              call_config_type_attribute->second, &type);
          if (!type_parsed) {
            return xla::FailedPrecondition("Could not parse the call type.");
          }
          auto* call_config = poplar_config.mutable_call_config();
          call_config->set_type(type);
          changed = true;
        }
      }
      poplar_config.set_stochastic_rounding(stochastic_rounding);
      instr->set_backend_config(poplar_config);
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
