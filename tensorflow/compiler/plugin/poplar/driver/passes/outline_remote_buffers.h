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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_OUTLINE_REMOTE_BUFFERS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_OUTLINE_REMOTE_BUFFERS_H_

#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/isomorphic_functions_map.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {
// A helper class to extract information about remote buffer inputs/outputs and
// how they should alias.
class RemoteBufferInputsOutputsInfos {
 public:
  explicit RemoteBufferInputsOutputsInfos(HloInstruction* inst);
  RemoteBufferInputsOutputsInfos() = delete;

  // Get the replication factor for a parameter load instruction - the index is
  // *before* permutation.
  uint64 GetLoadReplicationFactor(int64_t index) const;

  // Number of inputs/outputs which are loaded/stored.
  int64_t GetNumModifiedLoadStores() const;

  // Number of inputs which are loaded only.
  int64_t GetNumUnmodifiedLoads() const;

  // Number of total load inputs (both modified and unmodified).
  int64_t GetNumLoadInputs() const;

  // Get the replication factors.
  const absl::flat_hash_map<int64_t, uint64>& GetReplicationFactors() const;

  const std::vector<int64_t>& GetInputsOldToNewPermutation() const;
  const std::vector<int64_t>& GetInputsNewToOldPermutation() const;
  const std::vector<int64_t>& GetOutputsOldToNewPermutation() const;
  const std::vector<int64_t>& GetOutputsNewToOldPermutation() const;

  bool operator==(const RemoteBufferInputsOutputsInfos& other) const;
  bool operator!=(const RemoteBufferInputsOutputsInfos& other) const;

 private:
  // Store both kinds of permutations for quick lookup.
  std::vector<int64_t> inputs_old_to_new_permutation_;
  std::vector<int64_t> inputs_new_to_old_permutation_;
  std::vector<int64_t> outputs_old_to_new_permutation_;
  std::vector<int64_t> outputs_new_to_old_permutation_;
  absl::flat_hash_map<int64_t, uint64> input_to_replication_factor_;
  int64_t num_modified_load_stores_;
  int64_t num_unmodified_loads_;
};

/**
 * This pass outlines load and store instructions into functions to allow for
 * the load/store code to be reused between different calls.
 */
class OutlineRemoteBuffers : public HloModulePass {
 public:
  absl::string_view name() const override { return "outline-remote-buffers"; }

  StatusOr<bool> Run(HloModule* module) override;

  // Returns sets of functions which are all isomorphic and have the same
  // 'RemoteBufferInputsOutputsInfos' configuration.
  static SingleShardIsomorphicFunctions GetFunctionsForOutlining(
      HloModule* module);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_OUTLINE_REMOTE_BUFFERS_H_
