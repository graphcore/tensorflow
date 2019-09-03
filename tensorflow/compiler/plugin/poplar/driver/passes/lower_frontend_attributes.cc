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

#include "tensorflow/compiler/plugin/poplar/driver/passes/lower_frontend_attributes.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace poplarplugin {

StatusOr<bool> LowerFrontendAttributes::Run(HloModule* module) {
  bool changed = false;
  // Note: we expect all the instructions to have a value for these attributes
  // (For example in scopes.py instructions added after the end of a
  // stochastic_rounding scope will explicitely be tagged as NOT_SET). However
  // in some cases optimizers introduce new nodes without preserving the
  // frontend attributes of the node they replace which is why the variables
  // used to lower the frontend attributes in this method are declared outside
  // of the loop. This way we use the last successfully parsed values to
  // approximate the missing values.
  StochasticRounding stochastic_rounding = NOT_SET;
  PrimitiveType partials_type = PRIMITIVE_TYPE_INVALID;

  for (auto* comp : module->computations()) {
    for (auto instr : comp->instructions()) {
      auto attributes = instr->frontend_attributes();
      TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                          instr->backend_config<PoplarBackendConfig>());
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
      auto partials_type_attribute =
          attributes.map().find(FrontendAttributeId_Name(PARTIALS_TYPE));
      if (partials_type_attribute != attributes.map().end()) {
        bool type_parsed = PrimitiveType_Parse(partials_type_attribute->second,
                                               &partials_type);
        if (!type_parsed) {
          return xla::FailedPrecondition("Could not parse the partials type.");
        }
        switch (partials_type) {
          case F32:
          case F16:
          case PRIMITIVE_TYPE_INVALID:  // Switch back to default
            // Allowed partials type
            break;
          default:
            return xla::FailedPrecondition("Unsupported partials type.");
        }
        changed = true;
      }
      poplar_backend_config.set_partials_type(partials_type);
      poplar_backend_config.set_stochastic_rounding(stochastic_rounding);
      instr->set_backend_config(poplar_backend_config);
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
