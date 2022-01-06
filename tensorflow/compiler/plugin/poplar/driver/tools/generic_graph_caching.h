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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_GENERIC_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_GENERIC_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/xla/status.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include <map>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerResources;

using PoplarFunction = std::function<void(std::vector<poplar::Tensor>&,
                                          poplar::program::Sequence&)>;

namespace generic_graph_caching {

class GenericGraphCache {
 public:
  // Execute the func and cache it for the given instruction if it has not been
  // executed before.
  // `allocating_indices` indicates which of the args for inst have an
  // allocation target.
  // `layout_dependencies` indicates which of the args for inst have a layout
  // dependency. Note that the dependent allocation cannot be an Allocating
  // index or another layout dependency.
  Status ExecuteCached(
      const HloInstruction* inst, poplar::Graph& graph,
      CompilerResources& resources, poplar::program::Sequence& seq,
      PoplarFunction func, poputil::graphfn::Signature signature,
      std::vector<poplar::Tensor>& args,
      const absl::flat_hash_set<int64>& allocating_indices = {},
      const absl::flat_hash_map<int64, int64>& layout_dependencies = {},
      bool always_allocate = false);

 private:
  // Helper structs for the unordered map.
  struct HloInstructionHash {
    size_t operator()(const HloInstruction* inst) const;
  };
  struct HloInstructionEquals {
    bool operator()(const HloInstruction* a, const HloInstruction* b) const;
  };
  std::unordered_map<const HloInstruction*, poputil::graphfn::VoidFunction,
                     HloInstructionHash, HloInstructionEquals>
      table_;
};

}  // namespace generic_graph_caching
}  // namespace poplarplugin
}  // namespace xla

#endif
