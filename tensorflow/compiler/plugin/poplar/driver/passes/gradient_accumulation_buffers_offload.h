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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_BUFFERS_OFFLOAD_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_BUFFERS_OFFLOAD_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * This pass tries to find any gradient accumulation buffers in pipeline
 * instructions and replace them with gradient accumulation buffers which are
 * stored in remote memory. It also adds loads instructions where those
 * accumulators are used and store instructions after the accumulators have been
 * updated.
 *
 * For example given:
 *
 * bwd_pipeline_stage {
 *  p0 = parameter(0)
 *  p1 = parameter(1) <- gradient accumulator
 *  ...
 *  grad = ...
 *  f = serialized_gradient_accumulation(p1, grad)
 *  ROOT t = tuple(f, ...)
 * }
 *
 * wu_comp {
 *   p0 = parameter(0)
 *   p1 = parameter(1) <- gradient accumulator
 *   p0' = apply_grad(p0, p1)
 *   ROOT t = tuple(p0)
 * }
 *
 * pipeline {
 *   p0 = parameter(0)
 *   ...
 *   ga = gradient-accumulator-create()
 *   stage = bwd_pipeline_stage(p0, ga, ...)
 *   stage_0 = gte(stage) index 0
 *   sink = gradient-accumulator-sink(stage_0)
 *   ...
 *   t = resource_update(p0, sink)
 *   gte0 = gte(t) index 0 <- updated value of p0
 *   ....
 *   ROOT out = tuple(gte0, ...)
 * }
 *
 * entry {
 *   p0 = parameter(0)
 *   ...
 *   p = pipeline (p0, ...)
 *   gte0 = gte(p) index 0
 *   ROOT t = tuple(gte0, ...)
 * }
 *
 * Turn it into:
 *
 * bwd_pipeline_stage {
 *  p0 = parameter(0)
 *  p1 = parameter(1) <- gradient accumulator
 *  p1_loaded = remote-parameter-load(p1)
 *  ...
 *  grad = ...
 *  f = serialized_gradient_accumulation(p1_loaded, grad)
 *  p1_stored = remote-parameter-store(p1, f)
 *  ROOT t = tuple(p1_stored, ...)
 * }
 *
 * wu_comp {
 *   p0 = parameter(0)
 *   p1 = parameter(1) <- gradient accumulator
 *   p1_loaded = remote-parameter-load(p1)
 *   p0' = apply_grad(p0, p1_loaded)
 *   ROOT t = tuple(p0)
 * }
 *
 * pipeline {
 *   p0 = parameter(0)
 *   ...
 *   ga = gradient-accumulator-create(), is_remote=true <- in remote memory
 *   stage = bwd_pipeline_stage(p0, ga, ...)
 *   stage_0 = gte(stage) index 0
 *   sink = gradient-accumulator-sink(stage_0)
 *   ...
 *   t = resource_update(p0, sink)
 *   gte0 = gte(t) index 0 <- updated value of p0
 *   ....
 *   ROOT out = tuple(gte0, ...)
 * }
 *
 * entry {
 *   p0 = parameter(0)
 *   ...
 *   p = pipeline (p0, ...)
 *   gte0 = gte(p) index 0
 *   ROOT t = tuple(gte0, ...)
 * }
 *
 */
class GradientAccumulationBuffersOffload : public HloModulePass {
 public:
  GradientAccumulationBuffersOffload(bool remote_memory_supported,
                                     int64 minimum_remote_tensor_size);

  absl::string_view name() const override {
    return "gradient-accumulation-buffers-offload";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> ShouldOffloadInPipeline(HloInstruction* const pipeline_op);
  StatusOr<bool> OffloadInPipeline(HloInstruction* const pipeline_op);

  const bool remote_memory_supported_;
  const int64 minimum_remote_tensor_size_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_GRADIENT_ACCUMULATION_BUFFERS_OFFLOAD_H_
