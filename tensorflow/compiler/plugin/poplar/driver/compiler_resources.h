/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_

#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include <poplar/OptionFlags.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/CTCLoss.hpp>
#include <popops/DynamicSlice.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/GraphFunction.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/prng_seed_state.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/mapping_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/progress_bar.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/subcomputation_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
class HloInstruction;
class CallGraph;

namespace poplarplugin {
class PartitionedElementwiseClusterVisitor;

struct HloResources {
  HloResources(
      HloModule* module, CompilerInformation information,
      IpuOptions::FloatingPointBehaviour global_floating_point_behaviour,
      uint32 replication_factor, uint32 local_replication_factor,
      uint32 partition_replication_factor, bool merge_infeed_io_copies,
      bool always_rearrange_copies_on_host,
      IpuSchedulingAlgorithm scheduler_selection, bool recomputation_enabled,
      int64_t experimental_distributed_batch_norm_replica_group_size,
      bool remote_memory_supported, bool enable_experimental_prng_stability,
      bool enable_fast_math, int64_t num_io_tiles,
      double io_tile_available_memory_proportion, bool enable_progress_bar)
      : annotations(module),
        information(std::move(information)),
        global_floating_point_behaviour(
            std::move(global_floating_point_behaviour)),
        replication_factor(replication_factor),
        local_replication_factor(local_replication_factor),
        partition_replication_factor(partition_replication_factor),
        merge_infeed_io_copies(merge_infeed_io_copies),
        always_rearrange_copies_on_host(always_rearrange_copies_on_host),
        scheduler_selection(std::move(scheduler_selection)),
        recomputation_enabled(recomputation_enabled),
        experimental_distributed_batch_norm_replica_group_size(
            experimental_distributed_batch_norm_replica_group_size),
        remote_memory_supported(remote_memory_supported),
        enable_experimental_prng_stability(enable_experimental_prng_stability),
        enable_fast_math(enable_fast_math),
        num_io_tiles(num_io_tiles),
        io_tile_available_memory_proportion(
            io_tile_available_memory_proportion) {
    if (enable_progress_bar) {
      progress_bar = absl::make_unique<ProgressBar>(module);
    } else {
      progress_bar = absl::make_unique<NoProgressBar>();
    }
  }

  std::vector<unsigned> shard_to_ipu_id;

  CompilerAnnotations annotations;

  const CompilerInformation information;

  const IpuOptions::FloatingPointBehaviour global_floating_point_behaviour;
  /* The global number of replicas that we are compiling for. */
  uint32 replication_factor;
  /* The local number of replicas owned by this process. This is the number of
   * replicas that we are responsible for at run-time in this process. This is
   * only different from the `replication_factor` when using multi-replica
   * distribution with the Poplar "runtime replica subset" feature. */
  uint32 local_replication_factor;
  /* The number of replicas that we perform replica partitioning across. */
  uint32 partition_replication_factor;

  bool merge_infeed_io_copies;

  bool always_rearrange_copies_on_host;

  IpuSchedulingAlgorithm scheduler_selection;

  bool recomputation_enabled;

  int64_t experimental_distributed_batch_norm_replica_group_size;

  bool remote_memory_supported;

  bool enable_experimental_prng_stability;

  bool enable_fast_math;

  int64_t num_io_tiles;

  double io_tile_available_memory_proportion;

  int64_t num_uninitialised = 0;  // to help guarentee they are all unique

  absl::flat_hash_map<const HloInstruction*, std::uint64_t>
      hlo_instruction_to_debug_id_mapping;

  // The implementation of the progress bar.
  std::unique_ptr<ProgressBarBase> progress_bar;
};

// This structure contains additional information required to lower the graph
// from an XLA graph to a poplar graph.
struct CompilerResources : public HloResources {
  std::unique_ptr<DriverGraph> main_graph;

  // If IO tiles are not in use, these are nullopt (use main_graph instead).
  absl::optional<DriverGraph> compute_graph;
  absl::optional<DriverGraph> io_graph;

  // If IO tiles are not in use, only shard_compute_graphs is populated.
  std::vector<DriverGraph> shard_compute_graphs;
  std::vector<DriverGraph> shard_io_graphs;

  absl::flat_hash_map<const HloInstruction*, const popops::SlicePlan*>
      slice_plan_mappings;

  std::list<popops::SlicePlan> slice_plans;
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      slice_plan_allocators;

  absl::flat_hash_map<const HloInstruction*, const popnn::ctc::Plan> ctc_plans;

  poplin::PlanningCache planning_cache;

  const poplar::OptionFlags default_conv_options;

  const poplar::OptionFlags default_matmul_options;

  const poplar::OptionFlags default_pooling_options;

  const poplar::OptionFlags default_slice_options;

  bool clear_matmul_pass_type = false;

  bool disable_graph_outlining = false;

  TensorMaps tensor_maps;

  LinearMapperState linear_mapping_state;

  generic_graph_caching::GenericGraphCache graph_cache;

  subcomputation_graph_caching::SubcomputationGraphCache subcomputation_cache;

  std::unique_ptr<DriverProgramSequence> preamble_sequence;

  std::stack<std::vector<DriverProgramSequence>>
      gradient_accumulation_zeroing_remote_buffers;

  std::stack<std::vector<DriverTensor>> gradient_accumulation_zeroing_tensors;

  std::stack<std::vector<DriverProgramSequence>>
      pipelining_write_undef_sequences;

  std::stack<DeferredAllocations> deferred_allocation_scopes;

  std::stack<ExecutionCounters*> execution_counter_scopes;

  bool use_stable_norm_statistics;

  poplar::OptionFlags gcl_options;

  int64_t triangular_solve_expander_block_size;

  int64_t cholesky_block_size;

  std::unique_ptr<CallGraph> module_call_graph;

  absl::flat_hash_map<std::string, std::unique_ptr<RemoteBufferHolder>>
      remote_buffers;

  bool enable_experimental_remote_buffer_embedding;

  absl::flat_hash_set<std::string> custom_codelets_in_graph;

  absl::flat_hash_map<std::string,
                      std::pair<poplar::program::Sequence, poplar::Tensor>>
      infeed_cache;

  absl::flat_hash_map<std::string,
                      std::pair<poplar::program::Sequence, poplar::Tensor>>
      outfeed_cache;

  // TODO(T28772): remove this mapping and the extra copy.
  absl::flat_hash_map<std::string, DriverTensor> remote_buffer_layouts;

  PartitionedElementwiseClusterVisitor* current_cluster_visitor;

  // Two flags for SR. One to keep track of whether its currently enabled
  // and another to keep track of whether we enabled it at all and so
  // whether we use SR at all.
  bool stochastic_rounding_enabled;
  bool stochastic_rounding_used;

  PrngSeedState prng_seed_state;
  const bool enable_prng_seed_consistency_checks = false;

  CompilerResources(
      HloModule* module, const CompilerInformation& information,
      poplar::OptionFlags conv_options, poplar::OptionFlags matmul_options,
      poplar::OptionFlags pooling_options, poplar::OptionFlags slice_options,
      bool clear_matmul_pass_type, bool disable_graph_outlining,
      bool merge_infeed_io_copies, uint32 replication_factor,
      uint32 local_replication_factor, uint32 partition_replication_factor,
      const IpuOptions::FloatingPointBehaviour& floating_point_behaviour,
      bool always_rearrange_copies_on_host,
      IpuSchedulingAlgorithm scheduler_selection, bool recomputation_enabled,
      bool use_stable_norm_statistics,
      int64_t experimental_distributed_batch_norm_replica_group_size,
      bool remote_memory_supported, poplar::OptionFlags gcl_options,
      int64_t triangular_solve_expander_block_size, int64_t cholesky_block_size,
      bool enable_experimental_remote_buffer_embedding,
      bool enable_experimental_prng_stability, bool enable_fast_math,
      int64_t num_io_tiles, double io_tile_available_memory_proportion,
      bool enable_progress_bar)
      : HloResources(
            module, information, floating_point_behaviour, replication_factor,
            local_replication_factor, partition_replication_factor,
            merge_infeed_io_copies, always_rearrange_copies_on_host,
            std::move(scheduler_selection), recomputation_enabled,
            experimental_distributed_batch_norm_replica_group_size,
            remote_memory_supported, enable_experimental_prng_stability,
            enable_fast_math, num_io_tiles, io_tile_available_memory_proportion,
            enable_progress_bar),
        default_conv_options(std::move(conv_options)),
        default_matmul_options(std::move(matmul_options)),
        default_pooling_options(std::move(pooling_options)),
        default_slice_options(std::move(slice_options)),
        clear_matmul_pass_type(clear_matmul_pass_type),
        disable_graph_outlining(disable_graph_outlining),
        use_stable_norm_statistics(use_stable_norm_statistics),
        gcl_options(gcl_options),
        triangular_solve_expander_block_size(
            triangular_solve_expander_block_size),
        cholesky_block_size(cholesky_block_size),
        enable_experimental_remote_buffer_embedding(
            enable_experimental_remote_buffer_embedding),
        current_cluster_visitor(nullptr),
        stochastic_rounding_enabled(floating_point_behaviour.esr()),
        stochastic_rounding_used(stochastic_rounding_enabled) {
    CHECK_EQ(enable_prng_seed_consistency_checks, false)
        << "PRNG seed consistency checks are only intended for use when "
           "testing.";
  }

  static std::unique_ptr<CompilerResources> CreateTestDefault(
      HloModule* module,
      const CompilerInformation& information = CompilerInformation()) {
    return CreateTestDefault(module,
                             /*enable_prng_seed_consistency_checks*/ false,
                             IpuOptions::FloatingPointBehaviour(), information);
  }

  static std::unique_ptr<CompilerResources> CreateTestDefault(
      HloModule* module, bool enable_prng_seed_consistency_checks,
      const IpuOptions::FloatingPointBehaviour& floating_point_behaviour,
      const CompilerInformation& information = CompilerInformation()) {
    return absl::WrapUnique(
        new CompilerResources(module, enable_prng_seed_consistency_checks,
                              floating_point_behaviour, information));
  }

  Status CreateMainGraph(const poplar::Target& target,
                         absl::optional<uint32> replication_factor = {}) {
    try {
      uint32 repl_factor =
          replication_factor ? *replication_factor : this->replication_factor;
      main_graph = absl::make_unique<DriverGraph>(
          target, poplar::replication_factor(repl_factor));
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Create Graph]", e);
    }
    return Status::OK();
  }

  void CreatePreambleSequence() {
    preamble_sequence =
        absl::make_unique<DriverProgramSequence>(*main_graph, "Preamble");
  }

  Status CreateMainGraphAndPreamble(
      const poplar::Target& target,
      absl::optional<uint32> replication_factor = {}) {
    TF_RETURN_IF_ERROR(CreateMainGraph(target, replication_factor));
    CreatePreambleSequence();
    return Status::OK();
  }

 private:
  CompilerResources(
      HloModule* module, bool enable_prng_seed_consistency_checks,
      const IpuOptions::FloatingPointBehaviour& floating_point_behaviour,
      const CompilerInformation& information)
      : CompilerResources(
            module, information,
            /*conv_options=*/{},
            /*matmul_options=*/{},
            /*pooling_options=*/{},
            /*slice_options=*/{},
            /*clear_matmul_pass_type=*/false,
            /*disable_graph_outlining=*/false,
            /*merge_infeed_io_copies=*/false,
            /*replication_factor=*/1,
            /*local_replication_factor=*/1,
            /*partition_replication_factor=*/1, floating_point_behaviour,
            /*always_rearrange_copies_on_host=*/false,
            IpuSchedulingAlgorithm::CHOOSE_BEST,
            /*recomputation_enabled=*/false,
            /*use_stable_norm_statistics=*/false,
            /*experimental_distributed_batch_norm_replica_group_size=*/1,
            /*remote_memory_supported=*/false,
            /*gcl_options=*/{},
            /*triangular_solve_expander_block_size=*/0,
            /*cholesky_block_size=*/9,
            /*enable_experimental_remote_buffer_embedding=*/false,
            /*enable_experimental_prng_stability=*/false,
            /*enable_fast_math=*/false,
            /*num_io_tiles=*/0,
            /*io_tile_available_memory_proportion=*/0.9,
            /*enable_progress_bar=*/false) {}
};

}  // namespace poplarplugin
}  // namespace xla

#endif
