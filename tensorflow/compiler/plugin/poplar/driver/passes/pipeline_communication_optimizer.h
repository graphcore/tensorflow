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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_COMMUNICATION_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_COMMUNICATION_OPTIMIZER_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * This pass optimizes and minimizes the amount of data being copied between
 * stages when there are multiple stages sharing data on the same IPU.
 * This can often occur when multiple pipeline stages which are not consecutive
 * in execution calculate gradients for the same variable and those gradients
 * have to be combined into a single gradient.
 * This pass tries to spot tensors which are copied by consecutive stages
 * (passed-through) with the start and end of the copy path being on the same
 * IPU but for different pipeline stages. This results in the inter-IPU copies
 * to be replaced by a FIFO.
 *
 * TODO(T15315) - support this pass for the interleaved schedule.
 */
class PipelineCommunicationOptimizer : public HloModulePass {
 public:
  explicit PipelineCommunicationOptimizer(bool remote_memory_supported);

  absl::string_view name() const override {
    return "pipeline-communication-optimizer";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Optimize a pipeline.
  StatusOr<bool> OptimizePipeline(HloInstruction* pipeline_op);

  bool remote_memory_supported_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_COMMUNICATION_OPTIMIZER_H_
