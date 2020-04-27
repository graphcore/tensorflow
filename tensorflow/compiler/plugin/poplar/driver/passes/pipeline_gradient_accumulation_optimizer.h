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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_GRADIENT_ACCUMULATION_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_GRADIENT_ACCUMULATION_OPTIMIZER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * A pass which detects outputs of backward pipeline stages (producers) being
 * used as inputs to gradient accumulation in subsequent backward pipeline
 * stages (consumers) all located on the same shard.
 *
 * This can be optimized by adding the gradients to the gradient accumulation
 * buffer inside of the producer stages and hence remove the memory used by
 * FIFOs.
 *
 * For example:
 * bwd_pipeline_stage_0 {
 *   // local layer gradient calculation
 *   ...
 *   grad = ...
 *   p5 = parameter(5)
 *   accumulator = parameter(6)
 *   // combine the gradients
 *   add_grads = add(grad, p5)
 *   // add the gradients to the gradient accumulation buffer
 *   accumlated = gradient-accumulator-add(accumulator, add_grads)
 *   ...
 *   ROOT tuple(..., accumulated, ...)
 * }
 *
 * bwd_pipeline_stage_2 {
 *   // local layer gradient calculation
 *   ...
 *   grad = ...
 *   ...
 *   ROOT tuple(..., grad, ...)
 * }
 *
 * bps2 = (...) bwd_pipeline_stage_2(...) stage_id = 2
 * bps2_grad = gte(bps2)
 * // bps2_grad is threaded through bwd_pipeline_stage_1
 * bps1 = (...) bwd_pipeline_stage_1(..., bps2_grad, ...) stage_id = 1
 * bps2_grad_threaded = gte(bps1)
 * gradient_buffer = gradient-accumulator-creator()
 * bps0 = (...) bwd_pipeline_stage_0(...,
 *                                   gradient_buffer,
 *                                   bps2_grad_threaded) stage_id = 0
 * accumulated_grad = gte(bps0)
 * accumulated_sink = gradient-accumulator-sink(accumulated_grad)
 *
 * Here the output of stage bwd_pipeline_stage_2 is threaded through the
 * bwd_pipeline_stage_1 and passed as an input to bwd_pipeline_stage_0 where it
 * is added to the gradient accumulator.
 *
 * This pass converts this into:
 * bwd_pipeline_stage_0 {
 *   // local layer gradient calculation
 *   ...
 *   grad = ...
 *   accumulator = parameter(5)
 *   // add the gradients to the gradient accumulation buffer
 *   accumlated = gradient-accumulator-add(accumulator, grad)
 *   ...
 *   ROOT tuple(..., accumulated, ...)
 * }
 *
 * bwd_pipeline_stage_2 {
 *   // local layer gradient calculation
 *   ...
 *   grad = ...
 *   accumulator = parameter(5)
 *   // add the gradients to the gradient accumulation buffer
 *   accumlated = gradient-accumulator-add(accumulator, grad)
 *   ...
 *   ROOT tuple(..., accumlated, ...)
 * }
 *
 * gradient_buffer = gradient-accumulator-creator()
 * bps2 = (...) bwd_pipeline_stage_2(..., gradient_buffer) stage_id = 2
 * bps2_accumulated_grad = gte(bps2)
 * // bps2_grad is no longer threaded through bwd_pipeline_stage_1
 * bps1 = (...) bwd_pipeline_stage_1(...) stage_id = 1
 * bps0 = (...) bwd_pipeline_stage_0(..., gradient_buffer) stage_id = 0
 * bps0_accumulated_grad = gte(bps0)
 * accumulated_sink = gradient-accumulator-sink(bps0_accumulated_grad,
 *                                              bps2_accumulated_grad)
 *
 * Where the gradients are added to the gradient accumulation within the stages
 * where the gradients were produced because `add` is a commutative/associative
 * operation.
 */
class PipelineGradientAccumulationOptimizer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "pipeline-gradient-accumulation-optimizer";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Optimize a pipeline.
  StatusOr<bool> OptimizePipeline(HloInstruction* pipeline_op);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_GRADIENT_ACCUMULATION_OPTIMIZER_H_
