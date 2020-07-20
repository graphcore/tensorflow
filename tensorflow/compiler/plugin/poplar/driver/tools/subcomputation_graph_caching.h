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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SUBCOMPUTATION_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SUBCOMPUTATION_GRAPH_CACHING_H_

#include <memory>
#include <poputil/GraphFunction.hpp>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace subcomputation_graph_caching {

class SubcomputationGraphCache {
 public:
  // Get or compile the DeferredVisitor for a computation.
  StatusOr<std::shared_ptr<DeferredVisitor>> GetOrCompileSubcomputation(
      CompilerResources& res, TensorOrRemoteBufferVectors& inputs,
      const HloComputation* computation);

 private:
  std::unordered_map<const HloComputation*, std::shared_ptr<DeferredVisitor>,
                     HloComputationHash, HloComputationEquals>
      table_;
};

}  // namespace subcomputation_graph_caching
}  // namespace poplarplugin
}  // namespace xla

#endif
