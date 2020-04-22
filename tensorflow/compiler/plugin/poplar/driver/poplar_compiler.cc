/* Copyright 2017 Graphcore Ltd
 */

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

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_compiler.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/add_block_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/all_to_all_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/apply_recompute_suggestion.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/computation_flattener.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_nan.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_slice_folding.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/dependency_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_broadcast_converter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/f16_constant_folding.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gather_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_fuser.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_verifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/hlo_computation_name_uniquify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_barrier_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_schedule_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/lift_recompute_suggestion.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/lower_frontend_attributes.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/matmul_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_slice_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_canonicalize.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_scale_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_gather_expander.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_scatter_expander.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_communication_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_feed_hoisting.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_variables_offload.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_verifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_blocked_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/replication_factor_to_constant.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_schedule_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/root_token_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/suggest_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_condition_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/wide_const_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/liveness_look_ahead_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/convolution_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/initialize.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <popfloat/experimental/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include <poplar/CSRFunctions.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/replication_factor.hpp>
#include <poprand/RandomGen.hpp>

namespace se = ::stream_executor;

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {
std::once_flag help_flag_printed;

int64 SizeFunction(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
}

StatusOr<std::string> GetPathToGraphProgFile(std::string filename) {
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of('/') + 1);
    path = path + "../compiler/plugin/poplar/" + filename;
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  // This is for unit tests
  {
    char buf[256];
    if (!getcwd(buf, 255)) {
      return xla::InternalErrorStrCat("Failed to retrieve current working ",
                                      "directory when looking for '", filename,
                                      "' codelets");
    }
    std::string path(buf);
    path = path + "/tensorflow/compiler/plugin/poplar/" + filename;
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  return xla::InternalErrorStrCat("Failed to find '", filename, "' codelets");
}

bool GetConstantSubOutput(const HloInstruction* root, const Shape& layout,
                          std::vector<Literal>& sub_result) {
  if (root->opcode() == HloOpcode::kConstant) {
    auto literal = root->literal().Relayout(layout);
    sub_result.emplace_back(std::move(literal));
    return true;
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (unsigned int i = 0; i < root->operand_count(); i++) {
      auto& sub_shape = layout.tuple_shapes(i);
      if (!GetConstantSubOutput(root->operand(i), sub_shape, sub_result)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// This function returns true if all the root outputs are constants and all the
// constants are stored in result in a flat tuple order for each output
bool GetConstantOutput(const HloInstruction* root, const Shape& layout,
                       std::vector<std::vector<Literal>>& result) {
  if (root->opcode() == HloOpcode::kConstant) {
    auto literal = root->literal().Relayout(layout);
    std::vector<Literal> sub_result;
    sub_result.emplace_back(std::move(literal));
    result.emplace_back(std::move(sub_result));
    return true;
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (unsigned int i = 0; i < root->operand_count(); i++) {
      auto& sub_shape = layout.tuple_shapes(i);
      std::vector<Literal> sub_result;
      if (!GetConstantSubOutput(root->operand(i), sub_shape, sub_result)) {
        return false;
      }
      result.emplace_back(std::move(sub_result));
    }
    return true;
  }
  return false;
}

bool AnyComputationHasSideEffects(const HloModule* module) {
  for (const auto& comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    if (comp->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

bool ShardingEnabled(const HloModule* module) {
  for (const auto* comp : module->MakeNonfusionComputations()) {
    for (const auto* inst : comp->instructions()) {
      if (inst->has_sharding()) {
        auto sharding = inst->sharding();
        if (IsSupportedSharding(sharding)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool HasPipeliningWithDefaultSharding(const HloModule* module) {
  // Check if there are any pipelines.
  auto pipeline_ops_or = GetPipelines(module);
  if (!pipeline_ops_or.ok()) {
    LOG(FATAL) << pipeline_ops_or.status();
  }
  std::vector<HloInstruction*> pipelines = pipeline_ops_or.ValueOrDie();
  if (pipelines.empty()) {
    return false;
  }

  // Go through all the pipelines.
  for (HloInstruction* pipeline_op : pipelines) {
    auto stages_or = GetPipelineStages(pipeline_op->to_apply());
    if (!stages_or.ok()) {
      LOG(FATAL) << stages_or.status();
    }
    // Make sure the order of forward stages is strictly increasing by one.
    PipelineStages stages = stages_or.ValueOrDie();
    int64 next_stage_id = 0;
    for (HloInstruction* stage : stages.forward) {
      if (next_stage_id++ != stage->sharding().GetUniqueDevice()) {
        return false;
      }
    }
  }

  return true;
}

int64 MaximalShard(const HloModule* module) {
  int64 maximal_shard = 0;
  for (const auto* comp : module->MakeNonfusionComputations()) {
    for (const auto* inst : comp->instructions()) {
      if (inst->has_sharding()) {
        auto sharding = inst->sharding();
        if (IsSupportedSharding(sharding)) {
          const auto& vec = GetShardingDeviceIdVector(sharding);
          for (auto s : vec) {
            maximal_shard = std::max(maximal_shard, s);
          }
        }
      }
    }
  }
  return maximal_shard;
}

int64 NumIPUsInShards(const HloModule* module) {
  int64 num_explicit_shards = MaximalShard(module) + 1;
  // Round it up to the next highest power of 2.
  if (num_explicit_shards <= 1LL) {
    return 1LL;
  }
  int64 rounded = 2;
  num_explicit_shards--;
  while (num_explicit_shards >>= 1LL) {
    rounded <<= 1LL;
  }
  return rounded;
}

bool AreAllOutputsParameters(const HloModule* module,
                             std::vector<uint64>& output_paramater_numbers) {
  const HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();

  // Get all the outputs
  std::vector<const HloInstruction*> outputs;
  if (root->opcode() == HloOpcode::kTuple) {
    outputs = {root->operands().begin(), root->operands().end()};
  } else if (root->opcode() == HloOpcode::kParameter) {
    outputs = {root};
  } else {
    return false;
  }

  // Check if all the outputs are parameters so that we can simply remap input
  // instead of executing the engine
  for (auto op : outputs) {
    if (op->opcode() != HloOpcode::kParameter) {
      return false;
    } else {
      output_paramater_numbers.push_back(op->parameter_number());
    }
  }

  // Check that all the parameters are in a standard layout format.
  const ComputationLayout layout = module->entry_computation_layout();
  for (uint64 param_number : output_paramater_numbers) {
    if (param_number < static_cast<uint64>(layout.parameter_count())) {
      auto parameter_shape = layout.parameter_layout(param_number).shape();
      const bool parameter_has_standard_layout = absl::c_all_of(
          FlattenedXlaShape(parameter_shape), [](const Shape& shape) {
            return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
          });
      if (!parameter_has_standard_layout) {
        return false;
      }
    }
  }

  // Check that the computation output shape is the same as the root
  return ShapeUtil::Equal(
      root->shape(),
      root->GetModule()->entry_computation_layout().result_shape());
}

// Checkk that module is of type - scalar, elementwise instructions only.
bool AreAllScalarElementwiseGraph(const HloModule* module) {
  const auto& entry_comp = module->entry_computation();

  for (auto* inst : entry_comp->instructions()) {
    switch (inst->opcode()) {
      case HloOpcode::kConstant:
      case HloOpcode::kParameter:
        if (!ShapeUtil::IsScalar(inst->shape())) {
          return false;
        }
        break;
      case HloOpcode::kTuple:
        if (entry_comp->root_instruction() != inst) {
          return false;
        }
        break;
      default:
        if (!(ShapeUtil::IsScalar(inst->shape()) && inst->IsElementwise())) {
          return false;
        }
        break;
    }
  }

  return true;
}

HloPrintOptions GetPrintOptions() {
  HloPrintOptions opts;
  opts.set_print_operand_shape(false)
      .set_print_percent(false)
      .set_include_layout_in_shapes(false);
  return opts;
}

StatusOr<poplar::program::Program> InitializeSeed(poplar::Graph& graph,
                                                  int replication_factor) {
  const std::string seed_prefix = "__seed";

  auto seed =
      graph.addVariable(poplar::UNSIGNED_INT, {2}, seed_prefix + "/tensor");
  graph.setTileMapping(seed, 0);

  poplar::program::Sequence seq;
  if (!UseSyntheticData()) {
    // Copy the seed from the data stream and set it.
    auto data_stream = graph.addHostToDeviceFIFO(
        GetRandomNumberSeedStream(), seed.elementType(), seed.numElements());
    seq.add(poplar::program::Copy(data_stream, seed));
  } else if (UseSyntheticData() && UseSyntheticDataInitializer()) {
    // Initialize the seed on the device.
    auto& initializer = DataInitializer::GetSyntheticDataInitializer();
    TF_ASSIGN_OR_RETURN(auto literal,
                        initializer.GetData(ShapeUtil::MakeShape(U32, {2})));
    TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, seed, literal));
  }
  poprand::setSeed(graph, seed, 0, seq, seed_prefix + "/set");

  return seq;
}

bool InitializeCycleCounter(poplar::Graph& graph,
                            poplar::program::Sequence& seq) {
  int tile = PoplarXlaFlags::Get().log_cycle_count;
  if (tile < 0 ||
      graph.getTarget().getTargetType() != poplar::TargetType::IPU) {
    return false;
  } else {
    std::string cycleCounterId = PoplarExecutor::GetCycleCounterStream();
    poplar::Tensor cycleCounter =
        poplar::cycleCount(graph, seq, tile, cycleCounterId + "/tensor");
    poplar::DataStream fifo = graph.addDeviceToHostFIFO(
        cycleCounterId, cycleCounter.elementType(), cycleCounter.numElements());
    seq.add(poplar::program::Copy(cycleCounter, fifo));
    return true;
  }
}

void setFpBehaviour(poplar::Graph& graph,
                    const IpuOptions::FloatingPointBehaviour& fp_control,
                    poplar::program::Sequence& seq) {
  if (graph.getTarget().getTargetType() == poplar::TargetType::IPU) {
    poplar::FloatingPointBehaviour fp_behaviour(
        fp_control.inv(), fp_control.div0(), fp_control.oflo(),
        fp_control.esr(), fp_control.nanoo());
    poplar::setFloatingPointBehaviour(graph, seq, fp_behaviour,
                                      "setFpBehaviour");
  } else {
    LOG(WARNING) << "Setting IPU floating point behaviour is not supported "
                    "on IPU_MODEL";
  }
}

void PrintHelpString() { LOG(INFO) << PoplarXlaFlags::GetFlagUsageString(); }

Status CreatePoplarGraphs(CompilerResources& resources, const HloModule* module,
                          PoplarExecutor* poplar_executor) {
  try {
    const poplar::Target& poplar_target =
        poplar_executor->GetOrCreatePoplarTarget();
    resources.main_graph = absl::make_unique<poplar::Graph>(
        poplar_target,
        poplar::replication_factor(resources.replication_factor));
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Create Graph] ", e);
  }

  if (resources.replication_factor > 1) {
    LOG(INFO)
        << "Automatically replicating the TensorFlow model by a factor of "
        << resources.replication_factor << ".";
  }

  auto& main_graph = GetMasterGraph(resources);
  const poplar::Target& target = main_graph.getTarget();
  if (ShardingEnabled(module)) {
    auto num_ipus = target.getNumIPUs();
    auto tiles_per_ipu = target.getTilesPerIPU();

    IpuSelectionOrder order = poplar_executor->GetSelectionOrder();
    if (order == IpuSelectionOrder::AUTO) {
      order = HasPipeliningWithDefaultSharding(module)
                  ? IpuSelectionOrder::SNAKE
                  : IpuSelectionOrder::ZIGZAG;
    }

    VLOG(1) << "Using " << IpuSelectionOrder_Name(order)
            << " selection order when mapping shards to IPUs.";
    for (unsigned virtual_graph_idx = 0; virtual_graph_idx < num_ipus;
         ++virtual_graph_idx) {
      unsigned ipu;
      // Given IPUs:
      //     ||                    ||
      //  _______               _______
      // |       |             |       |
      // |   2   |=============|   3   |
      // |_______|             |_______|
      //     ||                    ||
      //     ||                    ||
      //  _______               _______
      // |       |             |       |
      // |   0   |=============|   1   |
      // |_______|             |_______|
      switch (order) {
        case IpuSelectionOrder::SNAKE: {
          // With snake allocation order, we want to use IPUs in the following
          // order {0, 1, 3, 2}. This allows consecutive virtual graphs to
          // always be neighbours.
          unsigned mod = virtual_graph_idx % 4;
          ipu = virtual_graph_idx - mod + (mod < 2 ? mod : (5 - mod));
          break;
        }
        case IpuSelectionOrder::HOOF: {
          // With hoof allocation order, we want to use IPUs in the following
          // order {0, 2, 3, 1}. This allows the first and last virtual graph to
          // be on the same C2 card (and hence directly connected).
          unsigned half_num_ipus = num_ipus / 2;
          ipu = virtual_graph_idx < half_num_ipus
                    ? virtual_graph_idx * 2
                    : num_ipus - 1 - (virtual_graph_idx % half_num_ipus) * 2;
          break;
        }
        case IpuSelectionOrder::ZIGZAG:
        default: {
          // With zig-zag allocation order, we want to use IPUs in the following
          // order {0, 1, 2, 3}. Default ordering.
          ipu = virtual_graph_idx;
          break;
        }
      }
      resources.shard_graphs.emplace_back(main_graph.createVirtualGraph(
          ipu * tiles_per_ipu, (ipu + 1) * tiles_per_ipu));
      resources.shard_to_ipu_id.push_back(ipu);
    }
    VLOG(1) << "Created " << num_ipus << " IPU shards";
    VLOG(1) << "Shards have been mapped to the following IPUs:";
    int64 next_shard_id = 0;
    for (unsigned hw_id : resources.shard_to_ipu_id) {
      VLOG(1) << "  * Shard " << next_shard_id++ << " mapped to IPU " << hw_id;
    }
  }
  TF_ASSIGN_OR_RETURN(const std::string codelets_path,
                      GetPathToGraphProgFile("tf.gp"));
  main_graph.addCodelets(codelets_path);
  poplin::addCodelets(main_graph);
  popnn::addCodelets(main_graph);
  popops::addCodelets(main_graph);
  poprand::addCodelets(main_graph);
  popfloat::experimental::addCodelets(main_graph);

  return Status::OK();
}

StatusOr<std::vector<IpuSchedulerAlgorithm>> GetSchedulerList(
    CompilerResources& res) {
  std::vector<IpuSchedulerAlgorithm> schedulers;
  bool all = res.scheduler_selection.empty();
  if (all || res.scheduler_selection == "Clustering") {
    schedulers.push_back(CreateClusteringMemoryScheduler(res.information));
  }
  if (all || res.scheduler_selection == "PostOrder") {
    schedulers.push_back(
        MemorySchedulerAlgorithmToIPU(PostOrderMemoryScheduler));
  }
  if (res.scheduler_selection == "LookAhead") {
    schedulers.push_back(
        CreateLivenessLookAheadMemoryScheduler(res.information));
  }
  if (res.scheduler_selection == "ShortestPath") {
    schedulers.push_back(CreateShortestPathScheduler(res.information));
  }

  if (schedulers.size() == 0) {
    return xla::InvalidArgument(
        "Invalid scheduler specified. Options are 'LookAhead', "
        "'PostOrder' and 'Clustering'");
  }
  return schedulers;
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> PoplarCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    se::DeviceMemoryAllocator* device_allocator) {
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  if (stream_exec == nullptr) {
    return tensorflow::errors::Unknown(
        "NULL stream pointer in poplar compiler");
  }

  if (PoplarXlaFlags::Get().help) {
    std::call_once(help_flag_printed, &PrintHelpString);
  }

  VLOG(1) << "Begin compilation: " << module->name() << " for ordinal  "
          << stream_exec->device_ordinal();

  PoplarExecutor* poplar_executor(
      static_cast<PoplarExecutor*>(stream_exec->implementation()));

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;
  if (module->config().hlo_profiling_enabled()) {
    const auto& name = module->entry_computation()->name();
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    profile_index_map = absl::make_unique<HloProfileIndexMap>(*module);
    profile_printer =
        CreateHloProfilePrinterData(*profile_index_map, cost_analysis, name);
  }

  const ModuleFilenames filenames =
      poplar_executor->GetModuleFilenames(*module);
  if (poplar_executor->HaveExecutableCache()) {
    if (poplar_executor->HaveCachedExecutable(filenames)) {
      TF_ASSIGN_OR_RETURN(PoplarExecutable * poplar_executable,
                          PoplarExecutable::Deserialize(
                              std::move(module), std::move(profile_printer),
                              std::move(profile_index_map), filenames));

      if (poplar_executor->EnableSerialization()) {
        TF_RETURN_IF_ERROR(
            poplar_executor->CreateSerializedExecutableDirIfMissing());
        try {
          std::ifstream file(filenames.CachedExecutableFilename(),
                             std::ios::binary);
          auto poplar_binary = poplar::Executable::deserialize(file);

          TF_RETURN_IF_ERROR(PoplarExecutable::Export(
              filenames, poplar_binary, *poplar_executable,
              poplar_executor->GetReportExecutionFlags(),
              poplar_executor->GetOrCreatePoplarTarget()));
        } catch (const std::exception& e) {
          return PoplarExceptionToTensorflowStatus("[Deserialize] ", e);
        }
      }
      std::unique_ptr<Executable> executable;
      executable.reset(poplar_executable);

      VLOG(1) << "Loaded " << executable->module().name() << " from "
              << filenames.CachedEngineFilename();

      return std::move(executable);
    } else {
      VLOG(1) << "Couldn't find " << filenames.CachedEngineFilename()
              << " in executable cache";
    }
  }

  if (!poplar_executor->HasPoplarTarget()) {
    return xla::FailedPrecondition(
        "No device target has been configured. Did you configure the IPU "
        "devices by "
        "running `tensorflow.python.ipu.configure_ipu_system(ipu_options)`?");
  }
  std::lock_guard<std::mutex> g(static_mu_);

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // Work out the IPU division for this IPU.
  // Given device with `num_ipus` IPU chips, we get the number of shards
  // `num_shards` and the replication factor is `num_ipus`/`num_shards` (and
  // we also make sure `num_ipus` % `num_shards` == 0).
  const auto num_ipus = poplar_executor->GetOrCreatePoplarTarget().getNumIPUs();
  const auto num_shards = NumIPUsInShards(module.get());
  const auto replication_factor = num_ipus / num_shards;

  // Check that it's divisible.
  if (num_ipus % num_shards) {
    return xla::InternalErrorStrCat(
        "Trying to compile a graph for an IPU device with ", num_ipus,
        " IPUs and ", num_shards,
        " shards. The number of shards needs to "
        " divide the number of IPUs.");
  }

  CompilerResources resources(
      poplar_executor->GetConvolutionOptions(),
      poplar_executor->GetMatMulOptions(), poplar_executor->GetPoolingOptions(),
      poplar_executor->UseVerifiedTransfers(),
      poplar_executor->ClearMatmulPassType(),
      poplar_executor->DisableGraphConvCaching(),
      poplar_executor->DisableGraphOutlining(),
      poplar_executor->MergeInfeedCopies(), replication_factor,
      poplar_executor->GetMaxAllReduceBufferSize(),
      poplar_executor->GetMaxReduceScatterBufferSize(),
      poplar_executor->GetMaxInterIpuCopyBufferSize(),
      poplar_executor->GetMaxSendRecvClusterSize(),
      poplar_executor->GetMaxSchedulerLookaheadDepth(),
      poplar_executor->GetMaxSchedulerSearchSpaceSize(), module.get(),
      poplar_executor->FloatingPointBehaviour(),
      poplar_executor->AlwaysRearrangeCopiesOnTheHost(),
      poplar_executor->GetSchedulerSelection(),
      poplar_executor->RecomputationEnabled(),
      poplar_executor->UseStableNormStatistics(),
      poplar_executor->SupportsRemoteBuffers());

  if (replication_factor > 1) {
    VLOG(1) << "Created " << replication_factor << " replica IPU graph.";
  }

  {
    HloPassPipeline pipeline("IPU");
    if (!poplar_executor->RetainControlDependencies()) {
      pipeline.AddPass<DependencyReplacer>(false);
    }
    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<HloGetDimensionSizeRewriter>();
    pipeline.AddPass<CustomOpReplacer>();
    pipeline.AddPass<ParsePoplarBackendConfig>();
    pipeline.AddPass<PipelineFixer>();
    pipeline.AddPass<ReplicationFactorToConstant>(resources.replication_factor);
    pipeline.AddPass<GradientAccumulationFuser>(resources.annotations);
    pipeline.AddPass<HloComputationNameUniquify>();
    pipeline.AddPass<CholeskyExpander>();
    pipeline.AddPass<TriangularSolveExpander>();
    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<NotSupportedGatherExpander>();
    pipeline.AddPass<NotSupportedScatterExpander>();
    pipeline.AddPass<DynamicIndexSplitter>();
    pipeline.AddPass<HloPassFix<ConstantSliceFolding>>();
    pipeline.AddPass<HloPassFix<FuseOpsEarly>>(resources.annotations);
    pipeline.AddPass<HloCSE>(false);
    pipeline.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>();
    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<RootTokenReplacer>();
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<MapInliner>();
    pipeline.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>();
    pipeline.AddPass<ZeroSizedHloElimination>();
    pipeline.AddPass<ComputationFlattener>();
    pipeline.AddPass<TupleSimplifier>(true);
    // pass.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<F16ConstantFolding>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<HloCSE>(true);
    pipeline.AddPass<WideConstFinder>();
    pipeline.AddPass<CommutativeInstructionReorderOperands>();
    {
      auto& pass =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("repeated-fusing");
      pass.AddPass<CastsElimination>(resources.annotations);
      pass.AddPass<HloCSE>(true);
      pass.AddPass<HloDCE>();
      pass.AddPass<WhileLoopConstantSinking>();
      pass.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>();
      pass.AddPass<ReshapeMover>();
      pass.AddPass<SortSimplifier>();
      pass.AddPass<HloDCE>();
      pass.AddPass<WhileLoopConditionSimplify>();
      pass.AddPass<PipelineOptimizer>();
      pass.AddPass<HloPassFix<WhileLoopToRepeatSimplify>>();
      if (poplar_executor->EnableGatherSimplifier()) {
        pass.AddPass<GatherSimplifier>();
      }
      pass.AddPass<ScatterSimplifier>();
      pass.AddPass<MultiUpdateCanonicalize>();
      pass.AddPass<MultiUpdateCombiner>(resources.annotations);
      if (poplar_executor->EnableMultiSliceCombiner()) {
        pass.AddPass<MultiSliceCombiner>(resources.annotations);
      }
    }
    pipeline.AddPass<AllToAllFinder>(resources.annotations,
                                     resources.replication_factor);
    {
      auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "multi-update-optimizer");
      pass.AddPass<MultiUpdateScaleApply>(resources.annotations);
      pass.AddPass<MultiUpdateApply>(resources.annotations);
      pass.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>();
      pass.AddPass<HloCSE>(true);
      pass.AddPass<HloDCE>();
    }
    if (poplar_executor->EnableMatmulCombiner()) {
      pipeline.AddPass<MatmulCombiner>(resources.annotations);
    }
    pipeline.AddPass<SliceOptimizer>(resources.annotations);
    pipeline.AddPass<HloPassFix<FuseOpsLate>>(resources.annotations);
    pipeline.AddPass<ElementwiseBroadcastConverter>();
    pipeline.AddPass<FuseWideConst>(resources.annotations);
    pipeline.AddPass<RecomputeInstructions>(
        poplar_executor->RecomputationEnabled());
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<PipelineResourceUpdateFixer>();
    pipeline.AddPass<PipelineResourceVariablesOffload>(
        resources.annotations, resources.remote_memory_supported);
    // Passes below this point need to respect control dependencies.
    if (poplar_executor->RecomputationEnabled()) {
      pipeline.AddPass<SuggestRecompute>();
      pipeline.AddPass<AddBlockRecompute>();
      {
        auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
            "resolve-recompute-suggestions");

        pass.AddPass<HloPassFix<RemoveBlockedRecomputeSuggestions>>();
        pass.AddPass<HloPassFix<LiftRecomputeSuggestion>>();
        pass.AddPass<ApplyRecomputeSuggestion>();
      }
    }
    pipeline.AddPass<HloPassFix<RemoveBlockedRecomputeSuggestions>>();
    pipeline.AddPass<HloPassFix<RemoveRecomputeSuggestions>>();
    pipeline.AddPass<DependencyReplacer>(true);
    pipeline.AddPass<HostComputeBarrierInserter>();
    pipeline.AddPass<ShardingPass>();
    pipeline.AddPass<HostComputeScheduleOptimizer>();
    pipeline.AddPass<PipelineFeedHoisting>();
    pipeline.AddPass<PipelineFIFOInserter>();
    pipeline.AddPass<PipelineCommunicationOptimizer>();
    {
      auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "post-pipeline-communication-optimizer");
      pass.AddPass<HloDCE>();
      pass.AddPass<PipelineOptimizer>();
    }
    pipeline.AddPass<InterIpuCopyInserter>();
    pipeline.AddPass<GradientAccumulationOptimizer>();
    // Passes below this point need to respect the inplace information.
    pipeline.AddPass<InplaceFinder>();
    pipeline.AddPass<ExpressionOutliner>();
    pipeline.AddPass<PipelineCopyInserter>();
    pipeline.AddPass<PipelineRecomputation>(
        poplar_executor->RecomputationEnabled(),
        poplar_executor->StatefulRecomputationEnabled());
    pipeline.AddPass<HloDCE>();
    // Beyond this point non of the passes in the pipeline are allowed to modify
    // the instructions in the HloModule.

    // TODO(T10195) re-enable.
    // if (!PoplarXlaFlags::Get().allow_nans) {
    //   pipeline.AddPass<ConstantNaN>();
    // }

    pipeline.AddPass<ModuleFlatten>(resources.annotations);
    pipeline.AddPass<PipelineVerifier>(poplar_executor->RecomputationEnabled());
    pipeline.AddPass<GradientAccumulationVerifier>(
        resources.replication_factor);
    pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
    if (resources.information.max_all_reduce_buffer_size > 0 ||
        resources.information.max_inter_ipu_copies_buffer_size > 0 ||
        resources.information.max_send_recv_cluster_size > 0) {
      pipeline.AddPass<IpuScheduler>(
          SizeFunction, CreateClusteringMemoryScheduler(resources.information));
      pipeline.AddPass<CombineInstructions>();
      pipeline.AddPass<HloDescheduler>();
    }
    pipeline.AddPass<AllocationFinder>(resources.annotations);
    pipeline.AddPass<HloPassFix<ForwardAllocation>>(resources.annotations);

    TF_ASSIGN_OR_RETURN(auto schedulers, GetSchedulerList(resources));

    TF_ASSIGN_OR_RETURN(auto scheduler, BestIpuSchedule(schedulers));

    pipeline.AddPass<ResourceUpdateScheduleOptimizer>();
    pipeline.AddPass<IpuScheduler>(SizeFunction, scheduler);
    pipeline.AddPass<LowerFrontendAttributes>();

    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }

  HloComputation* entry = module->entry_computation();

  if (poplar_executor->IpuTraceEventsEnabled()) {
    poplar_executor->AddCompileBeginEventRecord(module->name());
  }

  // Set layout if there isn't one
  auto comp_layout =
      module->mutable_entry_computation_layout()->mutable_result_layout();
  if (!comp_layout->LayoutIsSet()) {
    auto shape = entry->root_instruction()->shape();
    TF_CHECK_OK(comp_layout->CopyLayoutFromShape(shape));
  }

  // Strip all layout information, as the Poplar lowering does not use
  // layout information
  StripAllInstructionLayouts(module.get());

  VLOG(1) << "Compiling main computation " << entry->name();
  if (VLOG_IS_ON(1)) {
    XLA_VLOG_LINES(1, module->ToString(GetPrintOptions()));
  }

  if (VLOG_IS_ON(2)) {
    const auto& annotations = resources.annotations;
    XLA_VLOG_LINES(2, annotations.input_output_aliasing_map.ToString());
  }

  // Create a call graph of the final compiled module which can be used by the
  // lowering.
  resources.module_call_graph = CallGraph::Build(module.get());

  std::unique_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;

  std::vector<std::vector<Literal>> constant_output;
  const bool is_constant_output = GetConstantOutput(
      entry->root_instruction(), comp_layout->shape(), constant_output);

  const bool any_computation_has_side_effects =
      AnyComputationHasSideEffects(module.get());
  const auto is_constant_graph =
      is_constant_output && !any_computation_has_side_effects;

  std::string map_json;
  std::vector<uint64> remaped_output;

  const bool all_outputs_are_parameters =
      AreAllOutputsParameters(module.get(), remaped_output);

  bool is_remap_graph =
      all_outputs_are_parameters && !any_computation_has_side_effects;

  const bool all_scalar_elementwise_graph =
      AreAllScalarElementwiseGraph(module.get());

  const bool is_scalar_elementwise_graph =
      all_scalar_elementwise_graph && !any_computation_has_side_effects;

  if (is_constant_graph) {
    VLOG(1) << "Skip engine compilation - output is constant.";
  } else if (is_remap_graph) {
    VLOG(1) << "Skip engine compilation - all outputs are inputs.";
  } else if (is_scalar_elementwise_graph) {
    VLOG(1) << "Skip engine compilation - scalar elementwise graph.";
  } else {
    EntryVisitor visitor(resources, entry);
    // Only create the graphs if we are compiling.
    TF_RETURN_IF_ERROR(
        CreatePoplarGraphs(resources, module.get(), poplar_executor));
    auto& main_graph = GetMasterGraph(resources);

    EmbeddingPlansPreplanning embeddings_preplanning;
    TF_RETURN_IF_ERROR(embeddings_preplanning.Plan(module.get(), resources));

    try {
      ConvolutionPreplanning convolution_preplanning;
      TF_RETURN_IF_ERROR(convolution_preplanning.Plan(module.get(), resources));
      MatMulPreplanning matmul_preplanning;
      TF_RETURN_IF_ERROR(matmul_preplanning.Plan(module.get(), resources));
      auto order = module->schedule().sequence(entry).instructions();

      TF_RETURN_IF_ERROR(resources.streams_indices.InitializeIndexTensors(
          resources, poplar_executor->VerifiedTransfers()));

      // The following line starts the lowering in poplar.
      TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, order));
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Build graph] ", e);
    }

    poplar::program::Sequence main_program;

    // Set up the random seed
    TF_ASSIGN_OR_RETURN(auto seed_setup,
                        InitializeSeed(main_graph, replication_factor));
    main_program.add(seed_setup);

    // Set up the floating point control register if required
    const auto& fp_control = poplar_executor->FloatingPointBehaviour();
    if (fp_control.flags_set()) {
      setFpBehaviour(main_graph, fp_control, main_program);
    }

    // Add zeroing for registered variables
    main_program.add(ZeroTensors(resources));

    // Add the main program sequence
    main_program.add(visitor.GetSequence());

    if (InitializeCycleCounter(main_graph, main_program)) {
      poplar_executor->SetHasCycleCounter();
    }

    // =======================================================================
    // DO NOT CHANGE THE ORDER OF THESE WITHOUT UPDATING PoplarProgramType IN
    // poplar_executor.h
    // =======================================================================
    progs.push_back(visitor.GetHostToDevice());
    progs.push_back(main_program);
    progs.push_back(visitor.GetDeviceToHost());

    if (!PoplarXlaFlags::Get().save_vertex_graph.empty()) {
      auto filename =
          tensorflow::io::JoinPath(PoplarXlaFlags::Get().save_vertex_graph,
                                   module->name() + ".vertex_graph");
      VLOG(1) << "Dumping vertex graph " << filename;
      std::ofstream stream(filename);
      main_graph.outputVertexGraph(stream, progs);
    }

    try {
      VLOG(1) << "Compile engine " << module->name();

      map_json = GetTensorMappingJson(module->name(), main_graph,
                                      resources.tensor_maps);

      auto& opts = poplar_executor->GetOptionsFlags();
      auto progress_logging = [](int progress, int total) {
        float progress_percent = std::floor(
            100.0f * static_cast<float>(progress) / static_cast<float>(total));
        VLOG(1) << "Poplar compilation " << progress_percent << "% complete";
      };

      poplar::Executable exec =
          poplar::compileGraph(main_graph, progs, opts, progress_logging);

      if (poplar_executor->HaveExecutableCache()) {
        if (!poplar_executor->HaveCachedExecutable(filenames)) {
          TF_RETURN_IF_ERROR(
              poplar_executor->CreateExecutableCacheDirIfMissing());
          TF_RETURN_IF_ERROR(PoplarExecutable::Serialize(
              filenames, exec, resources.annotations, replication_factor,
              poplar_executor->GetReportExecutionFlags(),
              resources.streams_indices.GetAssignedIds(),
              resources.streams_indices.CheckpointFeedsOrder()));
        }
      }
      if (poplar_executor->EnableSerialization()) {
        TF_RETURN_IF_ERROR(
            poplar_executor->CreateSerializedExecutableDirIfMissing());

        TF_RETURN_IF_ERROR(PoplarExecutable::Export(
            filenames, exec, resources, replication_factor,
            poplar_executor->GetReportExecutionFlags(),
            poplar_executor->GetOrCreatePoplarTarget()));
      }

      engine.reset(new poplar::Engine(std::move(exec), opts));

    } catch (const std::exception& e) {
      if (poplar_executor->CompilerReportingEnabled()) {
        // Catch all exceptions but only do the profile printing if it is of
        // graph_memory_allocation_error type.
        const poplar::graph_memory_allocation_error* p_e_ptr =
            dynamic_cast<const poplar::graph_memory_allocation_error*>(&e);
        if (p_e_ptr) {
          DumpIfPoplarOutOfMemoryAllocationException(poplar_executor,
                                                     module->name(), *p_e_ptr);
        }
      }
      return PoplarExceptionToTensorflowStatus("[Compile engine] ", e);
    }
  }

  if (poplar_executor->IpuTraceEventsEnabled()) {
    std::stringstream report_stream;

    if (poplar_executor->CompilerReportingEnabled() && engine != nullptr) {
      try {
        auto rep = engine->getGraphProfile();
        if (poplar_executor->CompilerReportingTextFormat()) {
          auto opts = poplar_executor->GetReportGraphFlags();
          SetFlagIfNotPresent(opts, "showVarStorage", "true");
          poplar::printGraphSummary(report_stream, rep, opts);
        } else if (poplar_executor->CompilerReportingCborFormat()) {
          poplar::serializeToCBOR(report_stream, rep);
        } else {
          poplar::serializeToJSON(report_stream, rep);
        }

        if (PoplarXlaFlags::Get().dump_text_reports_to_stdio) {
          auto opts = poplar_executor->GetReportGraphFlags();
          SetFlagIfNotPresent(opts, "showVarStorage", "true");
          poplar::printGraphSummary(std::cout, rep, opts);
        }
      } catch (const std::exception& e) {
        return PoplarExceptionToTensorflowStatus("[Compiler report] ", e);
      }
    }

    uint64 duration = tensorflow::Env::Default()->NowMicros() - start_micros;

    if (report_stream.tellp() > poplar_executor->MaxReportSize()) {
      LOG(INFO)
          << "Dropping a Poplar compilation report of size "
          << report_stream.tellp()
          << " which is larger than the configured maximum report size "
          << std::to_string(poplar_executor->MaxReportSize())
          << ". To change the maximum report size use the max_report_size"
          << " argument in ipu.utils.create_ipu_config.\n"
          << "Example:\n"
          << "cfg = ipu.utils.create_ipu_config(max_report_size=0x100000000) "
          << "Note that the max report size is in bytes.";
      report_stream.str(std::string());
    }

    TF_ASSIGN_OR_RETURN(auto inst_info,
                        GetInstructionCompilationInfo(module, resources));

    poplar_executor->AddCompileEndEventRecord(
        module->name(), report_stream.str(), map_json, inst_info, duration);
  }

  std::unique_ptr<Executable> executable;
  PoplarExecutable* poplar_executable = new PoplarExecutable(
      std::move(module), std::move(profile_printer),
      std::move(profile_index_map), std::move(engine),
      std::move(resources.annotations.input_output_aliasing_map),
      is_constant_graph, std::move(constant_output), is_remap_graph,
      is_scalar_elementwise_graph, std::move(remaped_output),
      replication_factor, std::move(resources.annotations.infeed_infos),
      std::move(resources.annotations.outfeed_infos),
      std::move(resources.annotations.stream_infos),
      std::move(resources.annotations.stream_meta_infos),
      std::move(resources.annotations.send_infos),
      std::move(resources.annotations.recv_infos),
      std::move(resources.annotations.host_embedding_lookup_infos),
      std::move(resources.annotations.host_embedding_update_infos),
      std::move(resources.annotations.remote_parameter_infos),
      resources.streams_indices.GetAssignedIds(),
      resources.streams_indices.CheckpointFeedsOrder());

  executable.reset(poplar_executable);

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  if (module_group->empty()) {
    return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return tensorflow::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on Poplar.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return tensorflow::errors::Unimplemented(
        "Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module,
                      RunHloPasses(std::move(hlo_modules[0]), stream_exec[0][0],
                                   device_allocator));
  TF_ASSIGN_OR_RETURN(
      auto executable,
      RunBackend(std::move(module), stream_exec[0][0], device_allocator));
  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup>,
                                   const AotCompilationOptions&) {
  return xla::InvalidArgument("AOT compilation not supported on Poplar");
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return kPoplarPlatformId;
}

HloCostAnalysis::ShapeSizeFunction PoplarCompiler::ShapeSizeBytesFunction()
    const {
  return PoplarExecutable::ShapeSizeBytes;
}

std::mutex PoplarCompiler::static_mu_;

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool RegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      xla::poplarplugin::kPoplarPlatformId, &CreateComputationPlacer);
  return true;
}

bool placer_registration = RegisterComputationPlacer();

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      xla::poplarplugin::kPoplarPlatformId,
      []() { return absl::make_unique<xla::poplarplugin::PoplarCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
