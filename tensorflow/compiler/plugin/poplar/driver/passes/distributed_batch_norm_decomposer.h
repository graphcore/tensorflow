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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DISTRIBUTED_BATCH_NORM_DECOMPOSER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DISTRIBUTED_BATCH_NORM_DECOMPOSER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * A pass which finds distributed batch norm training instructions which will be
 * recomputed and decomposes them into:
 *
 * stats = batch-norm-statistics(activations)
 * mean = get-tuple-element(stats), index=0
 * variance = get-tuple-element(stats), index=1
 * mean = recomputation-checkpoint(mean)
 * variance = recomputation-checkpoint(variance)
 * activations_normalized = batch-norm-inference(activations, mean, variance)
 *
 * This means that the batch norm statistic will not be recomputed, but stored
 * instead which avoids the overhead of all reduce operations inside of the
 * stats calculation.
 */
class DistributedBatchNormDecomposer : public HloModulePass {
 public:
  explicit DistributedBatchNormDecomposer(bool allow_recomputation,
                                          int64 replica_group_size);

  absl::string_view name() const override {
    return "distributed-batch-norm-decomposer";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  const bool allow_recomputation_;
  const int64 replica_group_size_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DISTRIBUTED_BATCH_NORM_DECOMPOSER_H_
