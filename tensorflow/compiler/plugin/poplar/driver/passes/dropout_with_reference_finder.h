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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DROPOUT_WITH_REFERENCE_FINDER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DROPOUT_WITH_REFERENCE_FINDER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

struct CompilerAnnotations;

/**
 * Pass which tries to find a "forward" HloDropout operation (where
 * CanCreateReferenceTensor() == true) and a "backward" HloDropout (where
 * CanCreateReferenceTensor() == false) which share a seed and make sure they
 * use the same "reference tensor".
 *
 * The "forward" operation is the "IpuDropout" operation at TF and TF2XLA level
 * - this operation inserts the seed operation at the XLA level directly into
 * the Hlo graph and hence its origin is accounted for.
 *
 * The "backward" operations it the "IpuDropoutWithSeed" operation at TF and
 * TF2XLA level - this operation consumes a seed which is an output of a
 * "forward" operation.
 *
 * When using IPUs, thanks to the Hardware RNG, the dropout mask can be
 * regenerated deterministically between forward and backward operations which
 * can save memory.
 * To do so, a "seed" and a "reference tensor" are required. The "seed" is
 * passed from the "forward" to the "backward" operation automatically in the TF
 * auto grad - the seed tensor is used to set the Hardware RNG state to a fixed
 * configuration (note that the "seed" should change between iterations and be
 * different between replicas).
 * The "reference tensor" is then used to deterministically map the "forward"
 * and the "backward" tensor to the same IPU tiles so that the same dropout mask
 * can be applied between "forward" and the "backward" operation - however this
 * tensor does not need to be actually stored in memory - only its layout is
 * required at compilation time.
 *
 * By joining the "forward" and the "backward" operations, unique connections
 * can be made between these operations to ensure that a reference tensor can be
 * used between them in order to improve the performance.
 *
 * If a dropout operation cannot be joined, it will still use a "reference
 * tensor" however this tensor will be mapped linearly to ensure determinstic
 * behaviour.
 *
 * Note that there are some constraints applied to joining the the "forward" and
 * the "backward" operation:
 * - They both need to be present in the same Hlo graph
 * - The "seed" cannot be an output of the entry computation
 * This is because it cannot be guaranteed that there is no "backward" dropout
 * in a different Hlo graph and currently there are no means to share a
 * "reference tensor" between compilations.
 *
 * This pass should be executed before any recomputation passes at XLA level as
 * those can insert extra dropout operations
 */
class DropoutWithReferenceFinder : public HloModulePass {
 public:
  explicit DropoutWithReferenceFinder(CompilerAnnotations& annotations);

  absl::string_view name() const override {
    return "dropout-with-reference-finder";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_DROPOUT_WITH_REFERENCE_FINDER_H_
