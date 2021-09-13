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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ADD_STOCHASTIC_ROUNDING_OPTIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ADD_STOCHASTIC_ROUNDING_OPTIONS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"

namespace xla {
namespace poplarplugin {

// A HLO pass for adding deterministic workers/stochastic rounding
// backend options to instructions.
class AddStochasticRoundingOptions : public HloModulePass {
 public:
  explicit AddStochasticRoundingOptions(
      const StochasticRoundingBehaviour& default_stochastic_rounding_behaviour);

  absl::string_view name() const override {
    return "add-stochastic-rounding-options";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> ConfigureStochasticRoundingOption(HloInstruction* instr) const;
  StatusOr<ThreeState> ParseFrontendStochasticRoundingAttr(
      const HloInstruction* instr) const;

  StatusOr<bool> ConfigureDeterministicWorkersOption(
      HloInstruction* instr) const;

  StochasticRoundingBehaviour default_stochastic_rounding_behaviour_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_ADD_STOCHASTIC_ROUNDING_OPTIONS_H_
