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
#include "tensorflow/compiler/plugin/poplar/driver/tools/execution_counter_util.h"

#include <algorithm>
#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {

StatusOr<poplar::Tensor> GetExecutionCounter(CompilerResources& resources,
                                             const HloInstruction* inst) {
  int64 shard = 0;
  if (inst->has_sharding()) {
    auto optional_shard = inst->sharding_unique_device();
    if (!optional_shard) {
      return FailedPrecondition("Expected single device sharding.");
    }
    shard = *optional_shard;
  }
  if (resources.execution_counter_scopes.empty()) {
    return FailedPrecondition("Cannot access an execution counter.");
  }
  return resources.execution_counter_scopes.top()->GetCounter(shard);
}

Status CopyExecutionCountersFromScope(CompilerResources& resources,
                                      ExecutionCounters& counters,
                                      poplar::program::Sequence& sequence) {
  // There must already be a scope present.
  if (resources.execution_counter_scopes.empty()) {
    return FailedPrecondition("Cannot set the execution counters from stack.");
  }
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence copies_seq,
      counters.SetInitialValuesFrom(resources.execution_counter_scopes.top()));
  sequence.add(copies_seq);
  return Status::OK();
}

namespace {
// Helper functions to make sure there is at least one shard.
uint64 GetNumCounters(CompilerResources& resources) {
  return std::max(resources.shard_compute_graphs.size(), 1UL);
}
poplar::Graph& GetGraphForShard(CompilerResources& resources, size_t shard) {
  return resources.shard_compute_graphs.size()
             ? resources.shard_compute_graphs.at(shard)
             : *resources.main_graph;
}
}  // namespace

ExecutionCounters::ExecutionCounters(
    CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id)
    : counters_(GetNumCounters(resources)),
      live_counters_(GetNumCounters(resources)),
      resources_(resources),
      dnai_(debug_name_and_id) {}

ExecutionCounters ExecutionCounters::Clone(
    const poplar::DebugNameAndId& debug_name_and_id) {
  ExecutionCounters cloned(resources_, debug_name_and_id);
  // Clone all the live counters.
  for (size_t shard = 0; shard != counters_.size(); ++shard) {
    if (live_counters_[shard]) {
      poplar::Graph& graph = GetGraphForShard(resources_, shard);
      cloned.counters_[shard] = graph.clone(
          counters_[shard],
          {debug_name_and_id, absl::StrCat("ExecutionCounter/", shard)});
      cloned.live_counters_[shard] = true;
    }
  }

  return cloned;
}

StatusOr<poplar::Tensor> ExecutionCounters::GetCounter(int64 shard) {
  CHECK_LT(shard, counters_.size());
  if (!live_counters_[shard]) {
    // Requesting a counter which was not live, create it.
    poplar::Graph& graph = GetGraphForShard(resources_, shard);
    counters_[shard] = graph.addVariable(
        poplar::INT, {}, {dnai_, absl::StrCat("ExecutionCounter/", shard)});
    graph.setTileMapping(counters_[shard], 0);
    live_counters_[shard] = true;
  }
  return counters_[shard];
}

StatusOr<poplar::program::Sequence> ExecutionCounters::SetInitialValuesFrom(
    ExecutionCounters* source) {
  CHECK_EQ(source->counters_.size(), counters_.size());
  poplar::program::Sequence seq({}, dnai_);
  for (size_t shard = 0; shard != counters_.size(); ++shard) {
    if (live_counters_[shard]) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor source_counter,
                          source->GetCounter(shard));
      // Copy the value.
      seq.add(poplar::program::Copy(source_counter, counters_[shard], false,
                                    {dnai_}));
    }
  }
  initialized_ = true;
  return seq;
}

poplar::program::Sequence ExecutionCounters::SetInitialValuesToZero() {
  poplar::program::Sequence seq({}, dnai_);
  for (size_t shard = 0; shard != counters_.size(); ++shard) {
    if (live_counters_[shard]) {
      poplar::Graph& graph = GetGraphForShard(resources_, shard);
      popops::zero(graph, counters_[shard], seq,
                   {dnai_, absl::StrCat("ZeroExecutionCounter/", shard)});
    }
  }
  initialized_ = true;
  return seq;
}

StatusOr<poplar::program::Sequence> ExecutionCounters::UpdateCounters(
    ExecutionCounters* destination) {
  CHECK_EQ(destination->counters_.size(), counters_.size());
  poplar::program::Sequence seq({}, dnai_);
  for (size_t shard = 0; shard != counters_.size(); ++shard) {
    if (live_counters_[shard]) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor destination_counter,
                          destination->GetCounter(shard));
      // Copy the value.
      seq.add(poplar::program::Copy(counters_[shard], destination_counter,
                                    false, {dnai_}));
    }
  }
  return seq;
}

poplar::program::Sequence ExecutionCounters::IncrementLiveCounters() const {
  poplar::program::Sequence seq({}, dnai_);
  for (size_t shard = 0; shard != counters_.size(); ++shard) {
    if (live_counters_[shard]) {
      poplar::Graph& graph = GetGraphForShard(resources_, shard);
      popops::addInPlace(
          graph, counters_[shard], 1, seq,
          {dnai_, absl::StrCat("IncrementExecutionCounter/", shard)});
    }
  }
  return seq;
}

const std::vector<bool>& ExecutionCounters::GetLiveCounters() const {
  return live_counters_;
}

bool ExecutionCounters::Initialized() const { return initialized_; }

}  // namespace poplarplugin
}  // namespace xla
