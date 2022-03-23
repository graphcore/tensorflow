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
#include <string>
#include <unordered_map>
#include <vector>

#include <poputil/GraphFunction.hpp>

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

using RemoteBufferHandleVector = std::vector<absl::optional<std::string>>;
using RemoteBufferHandleVectors = std::vector<RemoteBufferHandleVector>;

struct SubcomputationGraphCacheKey {
  const HloComputation* computation;
  bool keep_input_layouts;
  bool partitioned_elementwise_cluster;
  RemoteBufferHandleVectors remote_buffer_handles;
};

struct SubcomputationGraphCacheKeyHash {
  size_t operator()(const SubcomputationGraphCacheKey& key) const;
};

struct SubcomputationGraphCacheKeyEquals {
  bool operator()(const SubcomputationGraphCacheKey& a,
                  const SubcomputationGraphCacheKey& b) const;
};

class SubcomputationGraphCache {
 public:
  // Get or compile the DeferredVisitor for a computation.
  StatusOr<std::shared_ptr<DeferredVisitor>> GetOrCompileSubcomputation(
      CompilerResources& res, TensorOrRemoteBufferVectors& inputs,
      const HloComputation* computation, bool keep_input_layouts = false,
      bool partitioned_elementwise_cluster = false);

  // Get or compile the DeferredVisitor for a computation.
  StatusOr<std::shared_ptr<DeferredVisitor>> GetOrCompileSubcomputation(
      CompilerResources& res, DeferredArgRBVectors& inputs,
      const HloComputation* computation, bool keep_input_layouts = false,
      bool partitioned_elementwise_cluster = false);

 private:
  int64 next_rearrangement_id_ = 1;
  std::unordered_map<
      const SubcomputationGraphCacheKey, std::shared_ptr<DeferredVisitor>,
      SubcomputationGraphCacheKeyHash, SubcomputationGraphCacheKeyEquals>
      table_;
};

}  // namespace subcomputation_graph_caching
}  // namespace poplarplugin
}  // namespace xla

#endif
