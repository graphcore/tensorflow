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

#include "tensorflow/compiler/plugin/poplar/driver/passes/add_stochastic_rounding_options.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

AddStochasticRoundingOptions::AddStochasticRoundingOptions(
    const StochasticRoundingBehaviour& default_stochastic_rounding_behaviour)
    : default_stochastic_rounding_behaviour_(
          default_stochastic_rounding_behaviour) {}

StatusOr<bool> AddStochasticRoundingOptions::Run(HloModule* module) {
  bool modified = false;

  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool added_option,
                          ConfigureStochasticRoundingOption(instr));
      modified |= added_option;

      TF_ASSIGN_OR_RETURN(added_option,
                          ConfigureDeterministicWorkersOption(instr));
      modified |= added_option;
    }
  }

  return modified;
}

StatusOr<bool> AddStochasticRoundingOptions::ConfigureStochasticRoundingOption(
    HloInstruction* instr) const {
  TF_ASSIGN_OR_RETURN(ThreeState stochastic_rounding,
                      ParseFrontendStochasticRoundingAttr(instr));

  // We only want to apply the default stochastic rounding option
  // if the instruction did not have one specified via its frontend
  // attributes. In this case stochastic rounding is either turned
  // on or off for all instructions, or just enabled for those that
  // that are replica identical.
  const bool use_default = stochastic_rounding == THREESTATE_UNDEFINED;
  if (use_default) {
    if (default_stochastic_rounding_behaviour_ ==
        StochasticRounding_ReplicaIdenticalOnly) {
      stochastic_rounding =
          IsInstructionReplicaIdentical(instr) ? THREESTATE_ON : THREESTATE_OFF;
    } else {
      stochastic_rounding =
          default_stochastic_rounding_behaviour_ == StochasticRounding_On
              ? THREESTATE_ON
              : THREESTATE_OFF;
    }
  }

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<PoplarBackendConfig>());
  backend_config.set_stochastic_rounding(stochastic_rounding);
  TF_RETURN_IF_ERROR(instr->set_backend_config(backend_config));

  return true;
}

StatusOr<ThreeState>
AddStochasticRoundingOptions::ParseFrontendStochasticRoundingAttr(
    const HloInstruction* instr) const {
  ThreeState stochastic_rounding = THREESTATE_UNDEFINED;

  auto attributes = instr->frontend_attributes();
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      instr->backend_config<PoplarBackendConfig>());

  auto stochastic_rounding_attribute =
      attributes.map().find(FrontendAttributeId_Name(STOCHASTIC_ROUNDING));
  if (stochastic_rounding_attribute != attributes.map().end()) {
    if (!ThreeState_Parse(stochastic_rounding_attribute->second,
                          &stochastic_rounding)) {
      return FailedPrecondition(
          "Could not parse the stochastic rounding value");
    }
  }

  return stochastic_rounding;
}

StatusOr<bool>
AddStochasticRoundingOptions::ConfigureDeterministicWorkersOption(
    HloInstruction* instr) const {
  const ThreeState deterministic_workers = IsInstructionReplicaIdentical(instr)
                                               ? THREESTATE_ON
                                               : THREESTATE_UNDEFINED;

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<PoplarBackendConfig>());
  backend_config.set_deterministic_workers(deterministic_workers);
  TF_RETURN_IF_ERROR(instr->set_backend_config(backend_config));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
