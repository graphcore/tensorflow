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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_TUPLE_REMOVER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_TUPLE_REMOVER_H_

#include <string>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * Pass which breaks up any nested tuples inputs/outputs from pipeline stages
 * into non-tuple shapes.
 *
 * Given two stages:
 * stage_0 {
 * ...
 * ROOT stage_0_out = ((f32[128,128], f32[3,128]), f32[1,2]) some-op(...)
 * }
 *
 * stage_1 {
 *  p0 = (f32[128,128], f32[3,128]) parameter(0)
 *  x = some-op(p0)
 *  ....
 * }
 *
 * Convert it to:
 * stage_0 {
 * ...
 * stage_0_out = ((f32[128,128], f32[3,128]), f32[1,2]) some-op(...)
 * gte_0 = (f32[128,128], f32[3,128]) gte(stage_0_out), index=0
 * gte_0_0 = f32[128,128] gte(gte_0), index=0
 * gte_0_1 = f32[3,128] gte(gte_0), index=1
 * gte_1 = f32[1,2] gte(stage_0_out), index=1
 * ROOT new_root = (f32[128,128], f32[3,128], f32[1,2]) tuple(gte_0_0, gte_0_1,
 *                                                            gte_1)
 * }
 *
 * stage_1 {
 *  p0 = f32[128,128] parameter(0)
 *  p1 = f32[3,128] parameter(1)
 *  t = (f32[128,128], f32[3,128]) tuple(p0, p1)
 *  x = some-op(t)
 *  ....
 * }
 */
class PipelineTupleRemover : public HloModulePass {
 public:
  absl::string_view name() const override { return "pipeline-tuple-remover"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> FlattenPipeline(HloInstruction* pipeline_op);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_PIPELINE_TUPLE_REMOVER_H_
