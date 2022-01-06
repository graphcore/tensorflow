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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_RECOMPUTATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_RECOMPUTATION_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * Pass which copies all the non-stateful computation operations from the
 * forward pass into the backward pass.
 */
class PipelineRecomputation : public HloModulePass {
 public:
  explicit PipelineRecomputation(bool allow_recomputation);

  absl::string_view name() const override { return "pipeline-recomputation"; }

  StatusOr<bool> Run(HloModule* module) override;

  // Find all the instructions which will be recomputed by this pass runs.
  static StatusOr<std::vector<HloInstruction*>> GetInstructionsToRecompute(
      HloModule* module);

 private:
  // Recompute a pipeline.
  StatusOr<bool> RecomputePipeline(HloInstruction* pipeline_op);

  bool allow_recomputation_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_RECOMPUTATION_H_
