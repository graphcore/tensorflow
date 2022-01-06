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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_VERIFIER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_VERIFIER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class CallGraph;
class HloModule;

namespace poplarplugin {

/**
 * Pass which verifies that the Pipeline is ready to be lowered. It verifies
 * that:
 * 1. PipelineStatefulGradientAccumulate instructions are only used in the right
 *    context.
 * 2. Each parameter instruction is only used by a pipeline stage and/or its
 *    corresponding backward pipeline stage. Any other uses are illegal.
 * 3. An output of a pipeline stage is either used by the next pipeline stage
 *    with an inter IPU copy operation between them and/or used by the
 *    corresponding pipeline stage with an FIFO operation between them.
 * 4. Verifies that all the sharding inside the pipeline stage matches.
 */
class PipelineVerifier : public HloModulePass {
 public:
  PipelineVerifier(bool allow_recomputation);

  absl::string_view name() const override { return "pipeline-verifier"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Verify a pipeline can be lowered.
  Status VerifyPipeline(HloInstruction* pipeline_op, CallGraph* call_graph);

  bool allow_recomputation_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
