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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_BATCH_SERIALIZATION_LOOP_INSERTER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_BATCH_SERIALIZATION_LOOP_INSERTER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * A pass which inserts the batch serialization loops around instructions inside
 * of the pipeline stage to ensure multiple batches are executed during a
 * single pipeline stage.
 *
 * This pass is the second part of creating a pipeline with batch serialization,
 * with the first part, PipelineBatchSerializationBufferInserter, inserting the
 * buffers and slice/update operations to make sure that each iteration of the
 * batch serialization loop will execute on a different set of inputs.
 */
class PipelineBatchSerializationLoopInserter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "pipeline-batch-serialization-loop-inserter";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  Status InsertIntoPipeline(HloInstruction* pipeline_op);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_BATCH_SERIALIZATION_LOOP_INSERTER_H_
