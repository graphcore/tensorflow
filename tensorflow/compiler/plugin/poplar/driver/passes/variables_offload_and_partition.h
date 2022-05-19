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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_VARIABLES_OFFLOAD_AND_PARTITION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_VARIABLES_OFFLOAD_AND_PARTITION_H_

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {
struct CompilerAnnotations;

/**
 * This pass tries to find any resource variables which are only used by the
 * resource update and move them into remote memory.
 *
 * For example given:
 *
 * wu_comp {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   p2 = parameter(2)
 *   p0', p2' = apply_grads(p0, p1, p2)
 *   ROOT t = tuple(p0, p2)
 * }
 *
 * pipeline {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   ...
 *   pipeline_stages(p0, p1, ...)
 *   bwd_pipeline_stages(...., p0, p1, ....)
 *   ...
 *   p2 = parameter(2) <- parameter only used in the resource update
 *   t = resource_update(..., p0, p1, p2, ...)
 *   gte0 = gte(t) index 0 <- updated value of p0
 *   gte1 = gte(t) index 1 <- updated value of p2
 *   ....
 *   ROOT out = tuple(gte0, p1, gte1, , ...)
 * }
 *
 * entry {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   p2 = parameter(2)
 *   ...
 *   p = pipeline (p0, p1, p2 ...)
 *   gte0 = gte(p) index 0
 *   gte1 = gte(p) index 1
 *   gte2 = gte(p) index 2
 *   ROOT t = tuple(gte0, gte1, gte2, ...)
 * }
 *
 * Here p2 is only used by the resource update computation inside of the
 * pipeline. We therefore mark it as a remote paramater and add load/store
 * instructions inside of the resource update, resulting in:
 *
 * wu_comp {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   p2 = parameter(2)
 *   p2_loaded = remote-parameter-load(p2)
 *   p0', p2' = apply_grads(p0, p1, p2_loaded)
 *   p2_stored = remote-paramter-store(p2, p2')
 *   ROOT t = tuple(p0, p2_stored)
 * }
 *
 * pipeline {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   ...
 *   pipeline_stages(p0, p1, ...)
 *   bwd_pipeline_stages(...., p0, p1, ....)
 *   ...
 *   p2 = parameter(2) <- parameter now
 *   t = resource_update(..., p0, p1, p2, ...)
 *   gte0 = gte(t) index 0 <- updated value of p0
 *   gte1 = gte(t) index 1 <- updated value of p2
 *   ....
 *   ROOT out = tuple(gte0, p1, gte1, ...)
 * }
 *
 * entry {
 *   p0 = parameter(0)
 *   p1 = parameter(1)
 *   p2 = parameter(2)
 *   ...
 *   p = pipeline (p0, p1, p2 ...)
 *   gte0 = gte(p) index 0
 *   gte1 = gte(p) index 1
 *   gte2 = gte(p) index 2
 *   ROOT t = tuple(gte0, gte1, gte2, ...)
 * }
 */
class VariablesOffloadAndPartition : public HloModulePass {
 public:
  VariablesOffloadAndPartition(CompilerAnnotations& annotations,
                               bool remote_memory_supported,
                               int64_t minimum_remote_tensor_size,
                               int64_t partition_replication_factor);
  absl::string_view name() const override {
    return "variables-offload-and-partition";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Optimize an instruction which contains a resource update.
  StatusOr<bool> Optimize(HloInstruction* call_op);
  StatusOr<ThreeState> ShouldOffloadInPipeline(
      HloInstruction* const pipeline_op);
  StatusOr<bool> ShouldPartitionInPipeline(HloInstruction* const pipeline_op);

  CompilerAnnotations& annotations_;
  const bool remote_memory_supported_;
  const int64_t minimum_remote_tensor_size_;
  const int64_t partition_replication_factor_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_VARIABLES_OFFLOAD_AND_PARTITION_H_
