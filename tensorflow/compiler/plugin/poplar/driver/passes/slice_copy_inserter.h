/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_COPY_INSERTER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_COPY_INSERTER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class CallGraph;
class HloModule;

namespace poplarplugin {

class CompilerAnnotations;
class HloPoplarDataflowAnalysis;

/**
 * This pass inserts `copy` instructions after `slice` instructions if that
 * would result in less live memory.
 *
 * As a trivial example, consider a big tensor that's only used in a small
 * slice. Because the slice provides a view to the original tensor, the original
 * tensor needs to be kept live as long as the slice output is used. Inserting a
 * copy after the slice will allow the original tensor memory to be released.
 */
class SliceCopyInserter : public HloModulePass {
 public:
  explicit SliceCopyInserter(CompilerAnnotations& annotations);

  ~SliceCopyInserter() = default;

  absl::string_view name() const override { return "slice-copy-inserter"; }

  StatusOr<bool> Run(HloModule* module) override;

  StatusOr<bool> Run(HloModule* module, const CallGraph& call_graph,
                     const HloPoplarDataflowAnalysis& dataflow);

 private:
  const CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SLICE_COPY_INSERTER_H_
