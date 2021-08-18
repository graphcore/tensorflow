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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_compiler.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <gcl/TileAllocation.hpp>
#include <limits>
#include <mutex>
#include <popfloat/experimental/codelets.hpp>
#include <poplar/CSRFunctions.hpp>
#include <poplar/CodeletFileType.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/exceptions.hpp>
#include <poplar/replication_factor.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>
#include <popsparse/codelets.hpp>
#include <poputil/exceptions.hpp>
#include <random>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/no_control_deps_checker.h"
#include "tensorflow/compiler/plugin/poplar/driver/invariant_passes/resource_update_checker.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/add_block_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/add_stochastic_rounding_options.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/all_to_all_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/apply_recompute_suggestion.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/computation_flattener.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_nan.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_slice_folding.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/conv_bwd_input_to_fwd_weights_transpose.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/dead_control_dependencies_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/dependency_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/distributed_batch_norm_decomposer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_broadcast_converter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/embeddings_gradient_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/f16_constant_folding.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fix_root_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/function_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/function_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_into_poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fusion_inliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gather_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_buffers_offload.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_fuser.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_verifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/hlo_computation_name_uniquify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_barrier_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_schedule_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_embedding_notification.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_tileset_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/io_tiles_placer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/lift_recompute_suggestion.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/lower_frontend_attributes.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/mark_replica_identical_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/matmul_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_conv_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_slice_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_canonicalize.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_update_scale_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_use_feeds_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_gather_expander.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_scatter_expander.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/outline_remote_buffers.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_buffer_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_loop_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_communication_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_control_dependency_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_feed_hoisting.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_gradient_accumulation_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation_stage_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_resource_update_input_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_stage_merger.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_tuple_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_verifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/post_serialize_gradient_accumulation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_checkpoint_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recomputation_input_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_casts.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_canonicalizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_merger.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_parameter_parallel_combiner.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_blocked_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/remove_recompute_suggestions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/repeat_loop_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/replicated_resource_update_elementwise_clustering.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/replication_factor_to_constant.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_fixer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_schedule_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/root_token_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scatter_simplifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/seed_hoisting.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/serialize_gradient_accumulation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/suggest_recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/variables_offload_and_partition.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_condition_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/wide_const_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable_cache.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/convolution_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/ctc_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/embedding_plans_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/map_hlo_instruction_to_debug_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_passes/matmul_preplanning.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/liveness_look_ahead_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/compilation_stats.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/error.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace se = ::stream_executor;

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

std::once_flag help_flag_printed;

int64 SizeFunction(const BufferValue& buffer) {
  if (buffer.shape().IsOpaque()) {
    return 0;
  }

  return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
}

// CompilationStats wrapper for generating PVTI tracepoints
// for HloPasses.
class PVTICompilerStats : public xla::CompilationStats {
 public:
  void StartPass(absl::string_view pass_name) override {
    TensorflowPoplarPluginTracepoint::BeginTrace(pass_name);
  }

  void EndPass(absl::string_view pass_name) override {
    TensorflowPoplarPluginTracepoint::EndTrace(pass_name);
  }

  // Pure virtual but not needed for our tracepoints.
  void CompilationReport() override {}
};

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

Status CheckUnsupportedPrecompileInstructions(const HloModule* module) {
  // Verify the operations in the module.
  for (const auto* comp : module->computations()) {
    for (const auto* inst : comp->instructions()) {
      if (inst->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      if (IsPoplarInstruction(PoplarOp::UserOp)(inst)) {
        if (!Cast<HloUserOpInstruction>(inst)->IsHashable()) {
          const std::string error_message = absl::StrCat(
              "IPU devices have been configured for pre-compilation, however "
              "the program contains custom user operation ",
              inst->ToString(),
              " which cannot be safely pre-compiled. If it is safe to "
              "pre-compile these Custom User operations, please set the "
              "'is_hashable' attribute to 'true' in the operations metadata "
              "function. Please see the documentation for further details.");
          return FailedPrecondition("%s", error_message);
        }
      }

      if (IsPoplarInstruction(PoplarOp::RecvFromHost)(inst) ||
          IsPoplarInstruction(PoplarOp::SendToHost)(inst)) {
        return FailedPrecondition(
            "IPU devices have been configured for pre-compilation, however "
            "the program contains `outside_compilation_scope` operations which "
            "cannot be safely pre-compiled.");
      }
    }
  }
  return Status::OK();
}

// Can only be called after CustomOpReplacer
bool IsCacheable(const HloModule* module) {
  for (const auto* comp : module->computations()) {
    for (const auto* inst : comp->instructions()) {
      if (IsPoplarInstruction(PoplarOp::UserOp)(inst) &&
          !(Cast<HloUserOpInstruction>(inst)->IsHashable())) {
        return false;
      }
    }
  }
  return true;
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

StatusOr<bool> ModuleExecutionCanStall(const HloModule* module,
                                       int32 num_io_tiles) {
  if (num_io_tiles) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  switch (pipeline_ops.size()) {
    case 0: {
      return false;
    }
    case 1: {
      TF_ASSIGN_OR_RETURN(const auto schedule,
                          GetPipelineSchedule(pipeline_ops[0]));
      switch (schedule) {
        case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
        case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved: {
          return true;
        }
        case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential: {
          return false;
        }
        default: { return FailedPrecondition("Unknown pipeline schedule."); }
      }
    }
    default: {
      return Unimplemented(
          "Multiple pipelines in the same HloModule are not supported");
    }
  }
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

StatusOr<poplar::program::Program> InitializeSeed(
    poplar::Graph& graph, int replication_factor,
    const poplar::DebugContext& debug_context = {"__seed"}) {
  PoplarOpDefDebugInfo debug_info(debug_context, "InitializeSeed");

  auto seed =
      graph.addVariable(poplar::UNSIGNED_INT, {2}, {debug_info, "seed"});
  graph.setTileMapping(seed, 0);

  poplar::program::Sequence seq({}, {debug_info});

  const auto use_synthetic_data =
      UseSyntheticDataFor(SyntheticDataCategory::Seed);
  if (!use_synthetic_data) {
    // Copy the seed from the data stream and set it.
    auto data_stream = graph.addHostToDeviceFIFO(
        GetRandomNumberSeedStream(), seed.elementType(), seed.numElements());
    seq.add(poplar::program::Copy(data_stream, seed, false, {debug_info}));
  } else if (use_synthetic_data && UseSyntheticDataInitializer()) {
    // Initialize the seed on the device.
    auto& initializer = DataInitializer::GetSyntheticDataInitializer();
    TF_ASSIGN_OR_RETURN(auto literal,
                        initializer.GetData(ShapeUtil::MakeShape(U32, {2})));
    TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, seed, literal));
  }
  poprand::setSeed(graph, seed, 0, seq, {debug_info, "set"});

  return seq;
}

bool InitializeCycleCounter(poplar::Graph& graph,
                            poplar::program::Sequence& seq,
                            const poplar::DebugContext& debug_context) {
  PoplarOpDefDebugInfo debug_info(debug_context, "InitializeCycleCounter");

  int tile = PoplarXlaFlags::Get().log_cycle_count;
  if (tile < 0 ||
      graph.getTarget().getTargetType() != poplar::TargetType::IPU) {
    return false;
  } else {
    std::string cycleCounterId = PoplarExecutor::GetCycleCounterStream();
    poplar::Tensor cycleCounter =
        poplar::cycleCount(graph, seq, tile, poplar::SyncType::INTERNAL,
                           {debug_info, cycleCounterId});
    poplar::DataStream fifo = graph.addDeviceToHostFIFO(
        cycleCounterId, cycleCounter.elementType(), cycleCounter.numElements());
    seq.add(poplar::program::Copy(cycleCounter, fifo, false, {debug_info}));
    return true;
  }
}

bool EnableProgressBar(const HloModule* module) {
  const std::string& show_progress_bar =
      PoplarXlaFlags::Get().show_progress_bar;

  if (show_progress_bar == "true") {
    return true;
  } else if (show_progress_bar == "false") {
    return false;
  } else if (show_progress_bar == "auto") {
    // Do not create the progress bar if this is not attached to a console.
    if (!isatty(fileno(stdout))) {
      return false;
    }

    // This doesn't check VLOG for all the files, but it's usually set for all
    // the files.
    if (VLOG_IS_ON(1)) {
      return false;
    }

    int64 num_expensive_ops = 0;
    for (const HloComputation* comp : module->computations()) {
      for (const HloInstruction* inst : comp->instructions()) {
        switch (inst->opcode()) {
          case HloOpcode::kAllReduce:
          case HloOpcode::kCholesky:
          case HloOpcode::kConvolution:
          case HloOpcode::kCustomCall:
          case HloOpcode::kDot:
          case HloOpcode::kInfeed:
          case HloOpcode::kOutfeed:
          case HloOpcode::kTriangularSolve: {
            num_expensive_ops++;
            break;
          }
          default: { break; }
        }
      }
    }
    return num_expensive_ops >= 5;
  } else {
    LOG(FATAL) << "Unknown value for 'show_progress_bar' flag. Needs to be one "
                  "of 'true', 'false' or 'auto' but got "
               << show_progress_bar;
    return true;
  }
}

void setFpBehaviour(poplar::Graph& graph,
                    const IpuOptions::FloatingPointBehaviour& fp_control,
                    poplar::program::Sequence& seq) {
  if (graph.getTarget().getTargetType() == poplar::TargetType::IPU) {
    const auto esr =
        fp_control.esr() == StochasticRoundingBehaviour::StochasticRounding_On;
    poplar::FloatingPointBehaviour fp_behaviour(
        fp_control.inv(), fp_control.div0(), fp_control.oflo(), esr,
        fp_control.nanoo());
    poplar::setFloatingPointBehaviour(graph, seq, fp_behaviour,
                                      "setFpBehaviour");
  } else {
    LOG(WARNING) << "Setting IPU floating point behaviour is not supported "
                    "on IPU_MODEL";
  }
}

void PrintHelpString() { LOG(INFO) << PoplarXlaFlags::GetFlagUsageString(); }

StatusOr<int> GetNumIoTiles(const PoplarExecutor* poplar_executor) {
  const int64 value = poplar_executor->GetNumIoTiles();
  if (value == 0) {
    return 0;
  }

  constexpr int kNumIoTilesMaxValue = 192;

  if (value > kNumIoTilesMaxValue) {
    return InvalidArgument(
        "%d is an invalid number of IO tiles. The number of IO tiles must be "
        "in the range [0, %d].",
        value, kNumIoTilesMaxValue);
  }

  // Round up the number of IO tiles to the next even number.
  if (value % 2) {
    LOG(INFO) << "Rounding the number of IO tiles up from " << value
              << " to the next even number " << value + 1 << ".";
    return value + 1;
  }

  return value;
}

std::vector<unsigned> DisjointTiles(const std::vector<unsigned>& tiles,
                                    unsigned num_tiles_per_ipu) {
  CHECK_LE(tiles.size(), num_tiles_per_ipu);
  CHECK(absl::c_is_sorted(tiles));

  std::vector<unsigned> other_tiles;
  other_tiles.reserve(num_tiles_per_ipu - tiles.size());

  unsigned i = 0;
  for (unsigned tile = 0; tile < num_tiles_per_ipu; ++tile) {
    if (i < tiles.size() && tiles[i] == tile) {
      ++i;
    } else {
      other_tiles.push_back(tile);
    }
  }

  CHECK_EQ(other_tiles.size() + tiles.size(), num_tiles_per_ipu);
  return other_tiles;
}

struct Tilesets {
  std::vector<unsigned> compute_tiles;
  std::vector<unsigned> io_tiles;
};

absl::optional<Tilesets> PartitionTiles(const poplar::Graph& main_graph,
                                        unsigned num_io_tiles,
                                        unsigned num_tiles_per_ipu) {
  if (num_io_tiles == 0) {
    return absl::nullopt;
  }

  CHECK_LT(num_io_tiles, num_tiles_per_ipu);

  const auto num_compute_tiles = num_tiles_per_ipu - num_io_tiles;

  const auto compute_tiles =
      gcl::perIPUTiles(main_graph, num_io_tiles, num_compute_tiles,
                       /*sorted=*/true, /*tilePairs=*/true);
  CHECK_EQ(compute_tiles.size(), num_compute_tiles);

  const auto io_tiles = DisjointTiles(compute_tiles, num_tiles_per_ipu);

  return Tilesets{compute_tiles, io_tiles};
}

Status CreatePoplarGraphs(CompilerResources& resources, const HloModule* module,
                          PoplarExecutor* poplar_executor) {
  try {
    const poplar::Target& poplar_target =
        poplar_executor->GetOrCreatePoplarTarget();
    resources.main_graph = absl::make_unique<poplar::Graph>(
        poplar_target,
        poplar::replication_factor(resources.replication_factor));
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Create Graph]", e);
  }

  if (resources.replication_factor > 1) {
    LOG(INFO)
        << "Automatically replicating the TensorFlow model by a factor of "
        << resources.replication_factor << ".";
  }

  auto& main_graph = GetMasterGraph(resources);
  const poplar::Target& target = main_graph.getTarget();
  const auto num_ipus = target.getNumIPUs();
  const auto tiles_per_ipu = target.getTilesPerIPU();

  TF_ASSIGN_OR_RETURN(const auto num_io_tiles, GetNumIoTiles(poplar_executor));
  const absl::optional<Tilesets> tilesets =
      PartitionTiles(main_graph, num_io_tiles, tiles_per_ipu);

  if (ShardingEnabled(module)) {
    IpuSelectionOrder order = poplar_executor->GetSelectionOrder();
    if (order == IpuSelectionOrder::AUTO) {
      order = HasPipeliningWithDefaultSharding(module)
                  ? IpuSelectionOrder::SNAKE
                  : IpuSelectionOrder::ZIGZAG;
    }

    absl::flat_hash_set<int64> shards_with_io_instructions;
    for (const HloComputation* comp : module->computations()) {
      for (const HloInstruction* inst : comp->instructions()) {
        TF_ASSIGN_OR_RETURN(const Tileset tileset, GetTileset(inst));
        if (tileset == TILESET_IO_TILES) {
          const std::vector<int64>& sharding =
              GetShardingDeviceIdVector(inst->sharding());
          for (const int64 shard : sharding) {
            if (!shards_with_io_instructions.contains(shard)) {
              shards_with_io_instructions.insert(shard);
            }
          }
        }
      }
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

      bool has_io_instructions =
          shards_with_io_instructions.contains(virtual_graph_idx);

      poplar::Graph ipu_graph = main_graph.createVirtualGraph(
          ipu * tiles_per_ipu, (ipu + 1) * tiles_per_ipu);

      if (tilesets.has_value()) {
        if (has_io_instructions) {
          LOG(INFO) << "Reserving " << num_io_tiles << " IO tile(s) on IPU "
                    << ipu << ".";
          resources.shard_compute_graphs.emplace_back(
              ipu_graph.createVirtualGraph(tilesets->compute_tiles));
          resources.shard_io_graphs.emplace_back(
              ipu_graph.createVirtualGraph(tilesets->io_tiles));
        } else {
          // Insert a placeholder I/O graph to preserve indexing by device id.
          resources.shard_io_graphs.emplace_back(
              ipu_graph.createVirtualGraph(0));
          resources.shard_compute_graphs.emplace_back(std::move(ipu_graph));
        }
      } else {
        resources.shard_compute_graphs.emplace_back(std::move(ipu_graph));
      }

      resources.shard_to_ipu_id.push_back(ipu);
    }
    VLOG(1) << "Created " << num_ipus << " IPU shards";
    VLOG(1) << "Shards have been mapped to the following IPUs:";
    int64 next_shard_id = 0;
    for (unsigned hw_id : resources.shard_to_ipu_id) {
      VLOG(1) << "  * Shard " << next_shard_id++ << " mapped to IPU " << hw_id;
    }
  }

  if (tilesets.has_value()) {
    resources.compute_graph.emplace(
        main_graph.createVirtualGraph(tilesets->compute_tiles));

    resources.io_graph.emplace(
        main_graph.createVirtualGraph(tilesets->io_tiles));
  }

  std::stringstream codelets_cpp_src{
#include "tensorflow/compiler/plugin/poplar/tf.cppembed"
  };
  std::stringstream codelets_asm_src{
#include "tensorflow/compiler/plugin/poplar/tf.Sembed"
  };

  std::stringstream compile_output;
  try {
    main_graph.addCodelets(codelets_cpp_src, "-DNDEBUG -O3", compile_output,
                           poplar::CodeletFileType::CppSource);
    main_graph.addCodelets(codelets_asm_src, "-DNDEBUG -O3", compile_output,
                           poplar::CodeletFileType::AsmSource);
  } catch (const poplar::graph_program_compilation_error&) {
    return xla::InternalError("Failed to compile Poplar TF codelets: %s",
                              compile_output.str());
  }
  poplin::addCodelets(main_graph);
  popnn::addCodelets(main_graph);
  popops::addCodelets(main_graph);
  poprand::addCodelets(main_graph);
  popsparse::addCodelets(main_graph);
  popfloat::experimental::addCodelets(main_graph);

  return Status::OK();
}

StatusOr<std::vector<NamedIpuSchedulerAlgorithm>> GetSchedulerList(
    CompilerResources& res) {
  std::vector<NamedIpuSchedulerAlgorithm> schedulers;

  const bool all =
      res.scheduler_selection == IpuSchedulingAlgorithm::CHOOSE_BEST;
  if (all || res.scheduler_selection == IpuSchedulingAlgorithm::POST_ORDER) {
    schedulers.push_back(
        {IpuSchedulingAlgorithm_Name(IpuSchedulingAlgorithm::POST_ORDER),
         MemorySchedulerAlgorithmToIPU(PostOrderMemoryScheduler)});
  }
  if (all || res.scheduler_selection == IpuSchedulingAlgorithm::CLUSTERING) {
    schedulers.push_back(
        {IpuSchedulingAlgorithm_Name(IpuSchedulingAlgorithm::CLUSTERING),
         CreateClusteringMemoryScheduler(res.information)});
  }
  if (all || res.scheduler_selection == IpuSchedulingAlgorithm::SHORTEST_PATH) {
    schedulers.push_back(
        {IpuSchedulingAlgorithm_Name(IpuSchedulingAlgorithm::SHORTEST_PATH),
         CreateShortestPathScheduler(res.information)});
  }

  // Not enabled with CHOOSE_BEST because of its time complexity.
  if (res.scheduler_selection == IpuSchedulingAlgorithm::LOOK_AHEAD) {
    schedulers.push_back(
        {IpuSchedulingAlgorithm_Name(IpuSchedulingAlgorithm::LOOK_AHEAD),
         CreateLivenessLookAheadMemoryScheduler(res.information)});
  }

  if (schedulers.empty()) {
    return xla::InvalidArgument(
        "Invalid scheduler specified. Options are 'LOOK_AHEAD',"
        " 'POST_ORDER', 'CLUSTERING' and 'SHORTEST_PATH'.");
  }

  return schedulers;
}

Status GetOperatingSystemInfo(utsname& details) {
  bool is_ok = uname(&details) == 0;
  if (!is_ok) {
    return xla::InternalErrorStrCat("Get system info error. ",
                                    std::strerror(errno));
  }

  return Status::OK();
}

StatusOr<std::string> GetFrameworkInfo(const std::string& module_name) {
  // tf info
  Json::Value tf_info;
  tf_info["Version"] = TF_VERSION_STRING;
  tf_info["Githash"] = tf_git_version();
  tf_info["Cluster"] = module_name;

  Json::Value root;
  root["TensorFlow"] = tf_info;

  // system info
  Json::Value system_info;
  utsname details;
  TF_RETURN_IF_ERROR(GetOperatingSystemInfo(details));
  system_info["Operating system"] = details.sysname;
  system_info["Operating system version"] = details.version;
  system_info["Machine"] = details.machine;

  system_info["Nominal CPU frequency [GHz]"] = absl::StrFormat(
      "%.2f", tensorflow::port::NominalCPUFrequency() / 1000000000.0);
  system_info["Number of total cpus"] = tensorflow::port::NumTotalCPUs();
  root["System Information"] = system_info;

  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";
  std::string tensorflow_info = Json::writeString(json_builder, root);

  return tensorflow_info;
}

void AddFrameworkFileToDirectory(const std::string& tensorflow_info,
                                 const std::string& directory) {
  CreateDirIfMissing(directory);
  Json::Value attrib_dummy;
  bool is_json_str = JsonParse(tensorflow_info, attrib_dummy);

  if (tensorflow_info.size() > 0 && is_json_str) {
    std::unique_ptr<tensorflow::WritableFile> wfile;
    std::string file_name =
        tensorflow::io::JoinPath(directory, "framework.json");
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(file_name, &wfile));
    TF_CHECK_OK(wfile->Append(tensorflow_info));
    TF_CHECK_OK(wfile->Close());
  }
}

void AddPipelineOptimizerPass(HloPassPipeline& pipeline) {
  auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
      "pipeline-optimizer-wrapper");
  pass.AddPass<PipelineOptimizer>();
  pass.AddPass<HloDCE>();
  pass.AddPass<HloCSE>(true);
}

/* RAII class used for locking the executable cache for a given file.
 * The idea is that when multiple processes are compiling the same executable
 * and have set the executable cache path to the same directory, they will
 * attempt to lock the same file. Only one of them will acquire the lock, and
 * then perform the compilation, serialize the result to disk, and finally
 * unlock the file. The other processes will then acquire the lock (one by one)
 * and find the serialized executable, skipping compilation.
 */
struct ExecutableCacheLock {
 public:
  static StatusOr<std::unique_ptr<ExecutableCacheLock>> CreateAndAcquire(
      const std::string& filepath) {
    const int fd = ::open(filepath.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd == -1) {
      return tensorflow::IOError("Failed to open " + filepath, errno);
    }

    VLOG(1) << "Acquiring lock for " << filepath;
    if (::flock(fd, LOCK_EX) == -1) {
      return tensorflow::IOError("Failed to lock " + filepath, errno);
    }

    VLOG(1) << "Acquired lock for " << filepath;

    // Using plain new to be able to keep constructor private.
    return std::unique_ptr<ExecutableCacheLock>(
        new ExecutableCacheLock(filepath, fd));
  }

  ~ExecutableCacheLock() {
    // Closing the file descriptor releases the lock.
    VLOG(1) << "Releasing lock for " << filepath_;
    if (::close(fd_) == -1) {
      LOG(WARNING) << "Failed to close file descriptor for " << filepath_
                   << ": " << std::strerror(errno);
    }

    // Attempt to delete the file to clean up after us. Deleting a file on
    // POSIX only means removing the name from the filesystem. If any other
    // processes still have the file open, the file will remain in existence
    // until the last file descriptor referring to it is closed.
    auto status = tensorflow::Env::Default()->DeleteFile(filepath_);

    // The different processes will race to delete the file, so we only care
    // if the deletion failed with anything else than NOT_FOUND.
    if (!status.ok() && status.code() != tensorflow::error::NOT_FOUND) {
      LOG(WARNING) << "Failed to delete " << filepath_ << ": "
                   << status.ToString();
    }
  }

 private:
  explicit ExecutableCacheLock(const std::string& filepath, int fd)
      : filepath_(filepath), fd_(fd) {}

  std::string filepath_;
  int fd_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutableCacheLock);
};

StatusOr<std::unique_ptr<PoplarExecutableCore>> CompileEngine(
    HloModule* module, PoplarExecutor* poplar_executor,
    uint64 executable_hash) {
  TENSORFLOW_TRACEPOINT();

  VLOG(1) << "Begin XLA compilation: " << module->name() << " " << std::hex
          << " (Hash: 0x" << HloHash(module).GetHash() << std::dec
          << ") for ordinal  " << poplar_executor->device_ordinal();

  poplar::OptionFlags opt_flags = poplar_executor->GetOptionsFlags();

  const bool in_precompile_mode =
      poplar_executor->ConnectionType() == IpuDeviceConnectionType::PRE_COMPILE;
  if (in_precompile_mode &&
      PoplarXlaFlags::Get().executable_cache_path.empty()) {
    return FailedPrecondition(
        "IPU devices have been configured for pre-compilation, however "
        "'executable_cache_path' in 'TF_POPLAR_FLAGS' has not been set. Please "
        "set the path to a directory where the compiled binaries should be "
        "stored using the 'TF_POPLAR_FLAGS' environment variable, for example "
        "'TF_POPLAR_FLAGS=--executable_cache_path=/path/to/storage'.");
  }

  // Check for autoReport.directory value in POPLAR_ENGINE_OPTIONS.
  // POPLAR_ENGINE_OPTIONS overrides values in opt_flags.

  // The output will be as follows
  // <autoReport.directory> or <cwd>
  //   debug.cbor
  //   tf_report_<xxx>
  //     frameworks.json
  //     profile.pop
  //   tf_report_<yyy>
  //     frameworks.json
  //     profile.pop

  absl::optional<std::string> auto_dir =
      GetPoplarEngineOption("autoReport.directory");

  if (auto_dir.has_value()) {
    // Set value in ipu options.
    opt_flags.set("autoReport.directory", *auto_dir);
  } else {
    // Check for autoReport.directory value in ipu options.
    auto auto_dir_itr = absl::c_find_if(
        opt_flags, [&](const poplar::OptionFlags::OptionFlag& flag) {
          return flag.first == "autoReport.directory";
        });
    if (auto_dir_itr != opt_flags.end()) {
      auto_dir = auto_dir_itr->second;
    } else {
      // default to the current working directory
      auto_dir = "./";
    }
  }

  std::unique_ptr<ExecutableCacheLock> executable_cache_lock;

  ModuleFilenames filenames =
      poplar_executor->GetModuleFilenames(executable_hash);

  if (poplar_executor->HaveExecutableCache()) {
    TF_RETURN_IF_ERROR(poplar_executor->CreateExecutableCacheDirIfMissing());
    TF_ASSIGN_OR_RETURN(executable_cache_lock,
                        ExecutableCacheLock::CreateAndAcquire(
                            filenames.CompilationLockFilename()));

    if (poplar_executor->HaveCachedExecutable(filenames)) {
      absl::optional<PoplarExecutableCore::RuntimeReplicaOptions>
          runtime_replica_options = absl::nullopt;
      if (poplar_executor->HasMultiReplicaDistributionOptions()) {
        runtime_replica_options = PoplarExecutableCore::RuntimeReplicaOptions{
            poplar_executor->GetMultiReplicaProcessIndex(),
            poplar_executor->GetMultiReplicaProcessCount()};
      }

      TF_ASSIGN_OR_RETURN(std::unique_ptr<PoplarExecutableCore> executable_core,
                          PoplarExecutableCore::Deserialize(
                              module, runtime_replica_options, filenames));

      if (poplar_executor->EnableSerialization()) {
        TF_RETURN_IF_ERROR(
            poplar_executor->CreateSerializedExecutableDirIfMissing());
        try {
          VLOG(1) << "Trying to deserialize cached file: "
                  << filenames.CachedExecutableFilename();
          std::ifstream file(filenames.CachedExecutableFilename(),
                             std::ios::binary);
          auto poplar_binary = poplar::Executable::deserialize(file);

          TF_RETURN_IF_ERROR(PoplarExecutableCore::Export(
              filenames, poplar_binary, *executable_core, {} /* device_opts */,
              opt_flags, poplar_executor->GetOrCreatePoplarTarget()));
        } catch (const std::exception& e) {
          const std::string origin =
              "[Deserialize][File: " + filenames.CachedExecutableFilename() +
              "]";
          return PoplarExceptionToTensorflowStatus(origin, e);
        }
      }

      VLOG(1) << "Loaded " << module->name() << " from "
              << filenames.CachedExecutableFilename();

      return executable_core;
    } else {
      VLOG(1) << "Couldn't find " << filenames.CachedExecutableFilename()
              << " in executable cache";
    }
  }

  if (!poplar_executor->HasPoplarTarget()) {
    return xla::FailedPrecondition(
        "No device target has been configured. Did you configure the IPU "
        "devices by running "
        "`tensorflow.python.ipu.utils.configure_ipu_system(ipu_options)`?");
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // Work out the IPU division for this IPU.
  // Given device with `num_ipus` IPU chips, we get the number of shards
  // `num_shards` and the replication factor is `num_ipus`/`num_shards` (and
  // we also make sure `num_ipus` % `num_shards` == 0).
  const poplar::Target& target = poplar_executor->GetOrCreatePoplarTarget();
  const auto num_ipus = target.getNumIPUs();
  const auto num_shards = NumIPUsInShards(module);
  const auto replication_factor = num_ipus / num_shards;
  TF_ASSIGN_OR_RETURN(const auto num_io_tiles, GetNumIoTiles(poplar_executor));

  // Check that it's divisible.
  if (num_ipus % num_shards) {
    return xla::InternalErrorStrCat(
        "Trying to compile a graph for an IPU device with ", num_ipus,
        " IPUs and ", num_shards,
        " shards. The number of shards needs to"
        " divide the number of IPUs.");
  }

  const auto num_local_ipus = poplar_executor->GetNumIpusInLocalProcess(target);
  const auto local_replication_factor = num_local_ipus / num_shards;

  if (num_local_ipus % num_shards) {
    return xla::InternalErrorStrCat(
        "With multi-replica distribution, the current local process has ",
        num_local_ipus, " IPUs, while the graph has ", num_shards, " shards.",
        " The number of shards needs to divide the number of local IPUs.");
  }

  CHECK_LE(local_replication_factor, replication_factor);

  // Currently we only support performing replica partitioning across the local
  // replicas in each process, as this allows access to all the parts of a
  // partitioned remote buffer locally. This means that copying to/from all the
  // parts of the partitioned remote buffer can be done without any additional
  // inter-process collective communication.
  const auto partition_replication_factor = local_replication_factor;

  VLOG(1) << "Local replication factor " << local_replication_factor
          << ", global replication factor " << replication_factor
          << ", partition replication factor " << partition_replication_factor;

  const auto information =
      CompilerInformation()
          .set_max_all_reduce_buffer_size(
              poplar_executor->GetMaxAllReduceBufferSize())
          .set_max_reduce_scatter_buffer_size(
              poplar_executor->GetMaxReduceScatterBufferSize())
          .set_max_inter_ipu_copies_buffer_size(
              poplar_executor->GetMaxInterIpuCopyBufferSize())
          .set_max_reduce_many_buffer_size(
              poplar_executor->GetMaxReduceManyBufferSize())
          .set_max_send_recv_cluster_size(
              poplar_executor->GetMaxSendRecvClusterSize())
          .set_max_scheduler_lookahead_depth(
              poplar_executor->GetMaxSchedulerLookaheadDepth())
          .set_max_scheduler_search_space_size(
              poplar_executor->GetMaxSchedulerSearchSpaceSize())
          .set_minimum_remote_tensor_size(
              poplar_executor->GetMinimumRemoteTensorSize());

  CompilerResources resources(
      module, information, poplar_executor->GetConvolutionOptions(),
      poplar_executor->GetMatMulOptions(), poplar_executor->GetPoolingOptions(),
      poplar_executor->GetSliceOptions(),
      poplar_executor->ClearMatmulPassType(),
      poplar_executor->DisableGraphOutlining(),
      poplar_executor->MergeInfeedCopies(), replication_factor,
      local_replication_factor, partition_replication_factor,
      poplar_executor->FloatingPointBehaviour(),
      poplar_executor->AlwaysRearrangeCopiesOnTheHost(),
      poplar_executor->GetSchedulerSelection(),
      poplar_executor->RecomputationEnabled(),
      poplar_executor->UseStableNormStatistics(),
      poplar_executor->ExperimentalDistributedBatchNormReplicaGroupSize(),
      poplar_executor->SupportsRemoteBuffers(), poplar_executor->GclOptions(),
      poplar_executor->GetTriangularSolveExpanderBlockSize(),
      poplar_executor->GetCholeskyBlockSize(),
      poplar_executor->EnableExperimentalRemoteBufferEmbedding(),
      poplar_executor->EnableFastMath(), num_io_tiles,
      poplar_executor->GetIoTileAvailableMemoryProportion(),
      EnableProgressBar(module));

  if (replication_factor > 1) {
    VLOG(1) << "Created " << replication_factor << " replica IPU graph.";
  }

  const int64 num_IPUs = target.getNumIPUs();
  const std::string target_type = poplar::toString(target.getTargetType());
  const std::string target_arch =
      target.getTargetType() == poplar::TargetType::IPU
          ? target.getTargetArchString()
          : "";
  const bool gateway_mode = target.getGatewayMode();
  const bool supports_remote_buffers = poplar_executor->SupportsRemoteBuffers();

  resources.progress_bar->Start();

  {
    std::unique_ptr<PVTICompilerStats> optimizer_compiler_stats =
        absl::make_unique<PVTICompilerStats>();
    HloPassPipeline optimizer_pipeline("OptimizerPipeline",
                                       optimizer_compiler_stats.get());

    if (PoplarXlaFlags::Get().enable_hlo_verifier) {
      optimizer_pipeline.AddInvariantChecker<HloVerifier>(
          /*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
    }

    {
      auto& pipeline = optimizer_pipeline.AddPass<HloPassPipeline>(
          "before control dependencies");
      // Make sure there are no control dependencies for the passes in this
      // pipeline.
      pipeline.AddInvariantChecker<NoControlDepsChecker>();

      pipeline.AddPass<FlattenCallGraph>();
      pipeline.AddPass<CustomOpReplacer>();
      pipeline.AddPass<ParsePoplarBackendConfig>();
      pipeline.AddPass<DynamicPadder>();
      pipeline.AddPass<PipelineFixer>();
      pipeline.AddPass<PipelineTupleRemover>();
      pipeline.AddPass<ReplicationFactorToConstant>(
          resources.replication_factor);
      pipeline.AddPass<GradientAccumulationFuser>(resources.annotations);
      pipeline.AddPass<HloComputationNameUniquify>();
      pipeline.AddPass<FlattenCallGraph>();
      pipeline.AddPass<NotSupportedGatherExpander>();
      pipeline.AddPass<NotSupportedScatterExpander>();
      pipeline.AddPass<QrExpander>();
      pipeline.AddPass<DynamicIndexSplitter>();
      pipeline.AddPass<HloPassFix<ConstantSliceFolding>>();
      pipeline.AddPass<HloPassFix<FuseOpsEarly>>(resources.annotations);
      pipeline.AddPass<MultiConvFixer>();
      pipeline.AddPass<HloCSE>(false);

      pipeline.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>(
          resources.enable_fast_math);
      {
        auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
            "pipeline-gradient-accumulation-optimizer-wrapper");
        pass.AddPass<PipelineGradientAccumulationOptimizer>();
        pass.AddPass<PipelineOptimizer>();
        pass.AddPass<HloDCE>();
        pass.AddPass<HloCSE>(true);
      }
      pipeline.AddPass<SortSimplifier>();
      pipeline.AddPass<RootTokenReplacer>();
      pipeline.AddPass<ReshapeMover>();
      pipeline.AddPass<MapInliner>();
      pipeline.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>(
          resources.enable_fast_math);
      pipeline.AddPass<ZeroSizedHloElimination>();
      pipeline.AddPass<FlattenCallGraph>();
      pipeline.AddPass<DistributedBatchNormDecomposer>(
          resources.recomputation_enabled,
          resources.experimental_distributed_batch_norm_replica_group_size);
      pipeline.AddPass<HloPassFix<SeedHoisting>>();
      pipeline.AddPass<PipelineRecomputation>(resources.recomputation_enabled);
      pipeline.AddPass<RecomputationCheckpointRemover>();
      pipeline.AddPass<FlattenCallGraph>();
      pipeline.AddPass<PipelineTupleRemover>();
      pipeline.AddPass<ComputationFlattener>();
      pipeline.AddPass<TupleSimplifier>(true);
      pipeline.AddPass<HloDCE>();
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
        pass.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>(
            resources.enable_fast_math);
        pass.AddPass<ReshapeMover>();
        pass.AddPass<SortSimplifier>();
        pass.AddPass<FunctionOptimizer>();
        pass.AddPass<HloDCE>();
        pass.AddPass<WhileLoopConditionSimplify>();
        pass.AddPass<PipelineOptimizer>();
        pass.AddPass<HloPassFix<WhileLoopToRepeatSimplify>>();
        if (poplar_executor->EnableGatherSimplifier()) {
          pass.AddPass<GatherSimplifier>();
        }
        pass.AddPass<ScatterSimplifier>();
        pass.AddPass<MultiUpdateCanonicalize>();
        pass.AddPass<EmbeddingsGradientOptimizer>();
        pass.AddPass<MultiUpdateCanonicalize>();
        pass.AddPass<MultiUpdateCombiner>(resources.annotations);
        if (poplar_executor->EnableMultiSliceCombiner()) {
          pass.AddPass<MultiSliceCombiner>(resources.annotations);
        }
      }
      pipeline.AddPass<PipelineResourceUpdateInputOptimizer>();
      AddPipelineOptimizerPass(pipeline);
      pipeline.AddPass<CommutativeInstructionReorderOperands>();
      pipeline.AddPass<AllToAllFinder>(resources.annotations,
                                       resources.replication_factor);
      {
        auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
            "multi-update-optimizer");
        pass.AddPass<MultiUpdateScaleApply>(resources.annotations);
        pass.AddPass<MultiUpdateApply>(resources.annotations);
        pass.AddPass<HloPassFix<PoplarAlgebraicSimplifier>>(
            resources.enable_fast_math);
        pass.AddPass<HloCSE>(true);
        pass.AddPass<HloDCE>();
      }
      if (poplar_executor->EnableMatmulCombiner()) {
        pipeline.AddPass<MatmulCombiner>(resources.annotations);
      }
      pipeline.AddPass<SerializeGradientAccumulation>();
      pipeline.AddPass<SliceOptimizer>(resources.annotations);
      pipeline.AddPass<FuseOpsLate>(resources.annotations);
      pipeline.AddPass<HloPassFix<FuseOpsIntoPoplarOps>>(resources.annotations);
      pipeline.AddPass<ExpressionOutliner>(/*maximum_num_elements=*/8);
      pipeline.AddPass<ElementwiseSimplifier>();
      pipeline.AddPass<ElementwiseBroadcastConverter>();
      pipeline.AddPass<FuseWideConst>(resources.annotations);
      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<HloCSE>(true);
      pipeline.AddPass<GradientAccumulationBuffersOffload>(
          resources.remote_memory_supported,
          resources.information.minimum_remote_tensor_size);
      pipeline.AddPass<PipelineStageMerger>();
      pipeline.AddPass<PipelineCommunicationOptimizer>(
          resources.remote_memory_supported);
      AddPipelineOptimizerPass(pipeline);
      {
        auto& batch_serialization_pass = pipeline.AddPass<HloPassPipeline>(
            "pipeline-batch-serialization-fixer-wrapper");
        batch_serialization_pass
            .AddPass<PipelineBatchSerializationBufferInserter>(
                resources.remote_memory_supported);
        AddPipelineOptimizerPass(batch_serialization_pass);
        batch_serialization_pass
            .AddPass<PipelineBatchSerializationLoopInserter>();
      }
      pipeline.AddPass<ResourceUpdateFixer>();
      pipeline.AddPass<VariablesOffloadAndPartition>(
          resources.annotations, resources.remote_memory_supported,
          resources.information.minimum_remote_tensor_size,
          resources.partition_replication_factor);
      pipeline.AddPass<PipelineFeedHoisting>();
      pipeline.AddPass<PipelineFIFOInserter>(resources.remote_memory_supported);
      pipeline.AddPass<ReplicatedResourceUpdateElementwiseClustering>(
          resources.partition_replication_factor, resources.replication_factor);
      {
        auto inline_fusion = [](const HloInstruction* inst) {
          return IsReplicatedParameterLoadFusion(inst) ||
                 IsReplicatedParameterStoreFusion(inst);
        };
        pipeline.AddPass<FusionInliner>(inline_fusion);
      }
      pipeline.AddPass<RemoteBufferCanonicalizer>(resources.annotations);
      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<HloCSE>(true);
      pipeline.AddPass<OutlineRemoteBuffers>();
      pipeline.AddPass<ResourceUpdateCopyInserter>();
      pipeline.AddPass<ResourceUpdateFixer>();
    }

    // Passes below this point need to respect control dependencies.
    {
      auto& pipeline = optimizer_pipeline.AddPass<HloPassPipeline>(
          "with control dependencies");
      pipeline.AddInvariantChecker<ResourceUpdateChecker>();

      pipeline.AddPass<HostEmbeddingNotification>();
      pipeline.AddPass<RecomputationInputRemover>();
      pipeline.AddPass<RecomputeInstructions>(resources.recomputation_enabled);

      if (resources.recomputation_enabled) {
        if (UsesRecomputationSuggestions(module)) {
          LOG(INFO) << "Detected SuggestRecompute operation - this will be "
                       "removed in release 2.2";

          pipeline.AddPass<SuggestRecompute>();
          pipeline.AddPass<AddBlockRecompute>();
          {
            auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
                "resolve-recompute-suggestions");

            pass.AddPass<HloPassFix<RemoveBlockedRecomputeSuggestions>>();
            pass.AddPass<HloPassFix<LiftRecomputeSuggestion>>();
            pass.AddPass<ApplyRecomputeSuggestion>();
          }
          pipeline.AddPass<HloPassFix<RemoveBlockedRecomputeSuggestions>>();
          pipeline.AddPass<HloPassFix<RemoveRecomputeSuggestions>>();
        } else {
          pipeline.AddPass<RecomputeCasts>();
        }
      }

      pipeline.AddPass<HostComputeBarrierInserter>();
      pipeline.AddPass<FlattenCallGraph>();
      pipeline.AddPass<ShardingPass>();
      pipeline.AddPass<HostComputeScheduleOptimizer>();
      pipeline.AddPass<InterIpuCopyInserter>();
      pipeline.AddPass<PipelineControlDependencyInserter>();
      pipeline.AddPass<IoTilesPlacer>(
          poplar_executor->ShouldPlaceOpsOnIoTiles(), num_io_tiles,
          target.getBytesPerTile(),
          poplar_executor->GetIoTileAvailableMemoryProportion());
      pipeline.AddPass<InterTilesetCopyInserter>();
      pipeline.AddPass<TupleSimplifier>(true);
      pipeline.AddPass<FixRootInstructionsPass>();
      pipeline.AddPass<DeadControlDependenciesElimination>();
      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<PostSerializeGradientAccumulation>();
      pipeline.AddPass<CopyInserter>();
      pipeline.AddPass<FunctionCombiner>();
    }

    // Passes below this point need to respect the inplace information.
    {
      auto& pipeline = optimizer_pipeline.AddPass<HloPassPipeline>(
          "with inplace information");
      pipeline.AddInvariantChecker<ResourceUpdateChecker>();

      pipeline.AddPass<InplaceFinder>();
      pipeline.AddPass<ExpressionOutliner>();
      pipeline.AddPass<PipelineCopyInserter>();
      pipeline.AddPass<RepeatLoopCopyInserter>();
      pipeline.AddPass<ModuleFlatten>(resources.annotations);
      pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
      pipeline.AddPass<ConvBwdInputToFwdWeightsTranspose>();
      pipeline.AddPass<PipelineRecomputationStageInserter>(
          resources.recomputation_enabled, resources.remote_memory_supported);
      if (resources.recomputation_enabled) {
        pipeline.AddPass<FlattenCallGraph>();
      }
      pipeline.AddPass<DeadControlDependenciesElimination>();
      pipeline.AddPass<HloDCE>();
      // Beyond this point non of the passes in the pipeline are allowed to
      // modify the instructions in the HloModule.

      // TODO(T10195) re-enable.
      // if (!PoplarXlaFlags::Get().allow_nans) {
      //   pipeline.AddPass<ConstantNaN>();
      // }

      pipeline.AddPass<PipelineVerifier>(resources.recomputation_enabled);
      pipeline.AddPass<GradientAccumulationVerifier>(
          resources.replication_factor);
      if (resources.information.max_all_reduce_buffer_size > 0 ||
          resources.information.max_inter_ipu_copies_buffer_size > 0 ||
          resources.information.max_send_recv_cluster_size > 0 ||
          resources.information.max_reduce_many_buffer_size > 0) {
        pipeline.AddPass<IpuScheduler>(
            SizeFunction,
            CreateClusteringMemoryScheduler(resources.information));
        pipeline.AddPass<CombineInstructions>();
        pipeline.AddPass<HloDescheduler>();
      }
      pipeline.AddPass<RemoteBufferMerger>(
          resources.annotations, poplar_executor->RemoteBufferMergingMode());
      pipeline.AddPass<RemoteParameterParallelCombiner>();
      pipeline.AddPass<AllocationFinder>(
          resources.annotations, resources.always_rearrange_copies_on_host);
      pipeline.AddPass<HloPassFix<ForwardAllocation>>(resources.annotations);

      TF_ASSIGN_OR_RETURN(auto schedulers, GetSchedulerList(resources));

      TF_ASSIGN_OR_RETURN(auto scheduler, BestIpuSchedule(schedulers));

      pipeline.AddPass<ResourceUpdateScheduleOptimizer>();
      pipeline.AddPass<IpuScheduler>(SizeFunction, scheduler);
      pipeline.AddPass<ModuleFlatten>(resources.annotations);
      pipeline.AddPass<LowerFrontendAttributes>();
      pipeline.AddPass<MarkReplicaIdenticalInstructions>();
      pipeline.AddPass<AddStochasticRoundingOptions>(
          resources.global_floating_point_behaviour.esr());
      pipeline.AddPass<MultiUseFeedsFinder>();
    }

    TF_RETURN_IF_ERROR(optimizer_pipeline.Run(module).status());
  }

  // Indicates whether the binary generated for this module can stall without
  // more data arriving.
  TF_ASSIGN_OR_RETURN(const bool is_module_which_can_stall,
                      ModuleExecutionCanStall(module, num_io_tiles));

  VLOG(1) << "End XLA compilation: " << module->name() << " (Hash: 0x"
          << std::hex << HloHash(module).GetHash() << ")";

  HloComputation* entry = module->entry_computation();

  // Set layout if there isn't one
  auto comp_layout =
      module->mutable_entry_computation_layout()->mutable_result_layout();
  if (!comp_layout->LayoutIsSet()) {
    auto shape = entry->root_instruction()->shape();
    TF_CHECK_OK(comp_layout->CopyLayoutFromShape(shape));
  }

  std::vector<std::vector<Literal>> constant_output;
  const bool is_constant_output = GetConstantOutput(
      entry->root_instruction(), comp_layout->shape(), constant_output);

  const bool any_computation_has_side_effects =
      AnyComputationHasSideEffects(module);
  const auto is_constant_graph =
      is_constant_output && !any_computation_has_side_effects;

  std::vector<uint64> remaped_output;

  const bool all_outputs_are_parameters =
      AreAllOutputsParameters(module, remaped_output);

  bool is_remap_graph =
      all_outputs_are_parameters && !any_computation_has_side_effects;

  const bool all_scalar_elementwise_graph =
      AreAllScalarElementwiseGraph(module);

  const bool is_scalar_elementwise_graph =
      all_scalar_elementwise_graph && !any_computation_has_side_effects;

  bool compile = true;
  if (is_constant_graph) {
    VLOG(1) << "Skip engine compilation - output is constant.";
    compile = false;
  } else if (is_remap_graph) {
    VLOG(1) << "Skip engine compilation - all outputs are inputs.";
    compile = false;
  } else if (is_scalar_elementwise_graph) {
    VLOG(1) << "Skip engine compilation - scalar elementwise graph.";
    compile = false;
  }

  const bool enable_trace_events = poplar_executor->IpuTraceEventsEnabled();
  if (enable_trace_events && compile) {
    poplar_executor->AddCompileBeginEventRecord(module->name());
  }

  // Strip all layout information, as the Poplar lowering does not use
  // layout information
  StripAllInstructionLayouts(module);

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
  resources.module_call_graph = CallGraph::Build(module);

  std::unique_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;
  bool logging_cycle_count;

  if (compile) {
    if (in_precompile_mode) {
      TF_RETURN_IF_ERROR(CheckUnsupportedPrecompileInstructions(module));
    }
    const bool is_cacheable = IsCacheable(module);

    // Generate a framework.json if autoReport.all or directory is configured.
    TF_ASSIGN_OR_RETURN(std::string tensorflow_info,
                        GetFrameworkInfo(module->name()));
    if (GetPoplarEngineOption("autoReport.all").has_value()) {
      // The current behaviour is to output the frameworks.json into the sub
      // director for each module.
      auto subdir = poplar_executor->GetModuleReportDirectory(module->name());
      AddFrameworkFileToDirectory(tensorflow_info,
                                  tensorflow::io::JoinPath(*auto_dir, subdir));
    }
    std::string tf_report_dir = poplar_executor->ReportDirectory();
    if (tf_report_dir.size() > 0) {
      AddFrameworkFileToDirectory(
          tensorflow_info,
          tensorflow::io::JoinPath(tf_report_dir, module->name()));
    }

    // Only create the graphs if we are compiling.
    TF_RETURN_IF_ERROR(CreatePoplarGraphs(resources, module, poplar_executor));
    auto& main_graph = GetMasterGraph(resources);

    EntryVisitor visitor(resources, entry);
    try {
      resources.progress_bar->MoveToNextStage();
      // Run a compile only Poplar specific pipeline - these passes do not
      // modify the module in a functional way.
      VLOG(1) << "Begin Poplar Pipeline.";
      std::unique_ptr<PVTICompilerStats> poplar_pipline_compiler_stats =
          absl::make_unique<PVTICompilerStats>();
      HloPassPipeline pipeline("PoplarPipeline",
                               poplar_pipline_compiler_stats.get());
      pipeline.AddPass<EmbeddingPlansPreplanning>(resources);
      pipeline.AddPass<ConvolutionPreplanning>(resources);
      pipeline.AddPass<MatMulPreplanning>(resources);
      pipeline.AddPass<CTCPreplanning>(resources);
      pipeline.AddPass<MapHloInstructionToDebugIdPass>(
          resources.hlo_instruction_to_debug_id_mapping);

      TF_RETURN_IF_ERROR(pipeline.Run(module).status());
      VLOG(1) << "End Poplar Pipeline.";

      resources.progress_bar->MoveToNextStage();

      auto order = module->schedule().sequence(entry).instructions();

      // The following line starts the lowering in poplar.
      VLOG(1) << "Begin Poplar graph construction.";
      TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, order));
      VLOG(1) << "End Poplar graph construction.";
      resources.progress_bar->MoveToNextStage();
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Build graph]", e);
    }

    poplar::program::Sequence main_program({}, {"MainProgram"});

    // Decide whether to synchronise all the replica's starting points.
    if (PoplarXlaFlags::Get().sync_replica_start && replication_factor > 1) {
      main_program.add(poplar::program::Sync(poplar::SyncType::GLOBAL));
    }

    // Set up the random seed.
    TF_ASSIGN_OR_RETURN(auto seed_setup,
                        InitializeSeed(main_graph, replication_factor));
    main_program.add(seed_setup);

    // Set up the floating point control register if required
    if (poplar_executor->FloatingPointBehaviourFlagsSet()) {
      const auto& fp_control = poplar_executor->FloatingPointBehaviour();
      setFpBehaviour(main_graph, fp_control, main_program);
    }

    // Add the preamble sequence.
    main_program.add(resources.preamble_sequence);

    // Add the main program sequence.
    main_program.add(visitor.GetSequenceAndInitializeCounters());

    logging_cycle_count = InitializeCycleCounter(main_graph, main_program,
                                                 "InitializeCycleCounter");

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
      VLOG(1) << "Begin compiling Poplar engine " << module->name();

      auto progress_logging = [&resources](int progress, int total) {
        resources.progress_bar->Update(progress, total);
        // Log into VLOG too.
        float progress_percent = std::floor(
            100.0f * static_cast<float>(progress) / static_cast<float>(total));
        VLOG(1) << "Poplar compilation " << progress_percent << "% complete";
      };

      if (poplar_executor->HasMultiReplicaDistributionOptions()) {
        SetRuntimeReplicaOptions(
            &opt_flags, poplar_executor->GetMultiReplicaProcessIndex(),
            poplar_executor->GetMultiReplicaProcessCount(), replication_factor);
        opt_flags.set("target.syncReplicasIndependently", "true");
      }

      // The poplar exeutableDebugName will cause the profile reports to be
      // output to a subdirectory of autoReport.directory if it is set to a
      // non-empty string.
      // If set to an empty string the profile reports will be written to the
      // autoReport.directory.
      std::string executable_debug_name = "";
      if (poplar_executor->GetAutoAssignReportSubdirectories()) {
        executable_debug_name =
            poplar_executor->GetModuleReportDirectory(module->name());
      }

      poplar::Executable exec =
          poplar::compileGraph(main_graph, progs, opt_flags, progress_logging,
                               executable_debug_name);

      if (is_cacheable) {
        // If we have the lock, serialize the result to the executable cache.
        if (executable_cache_lock) {
          // Serialize some additional options that Poplar does not serialize
          // on its own.
          poplar::OptionFlags options_to_serialize =
              poplar_executor->GetReportExecutionFlags();

          auto& annotations = resources.annotations;

          TF_RETURN_IF_ERROR(PoplarExecutableCore::Serialize(
              filenames, exec, options_to_serialize,
              PoplarExecutableInfo{
                  num_IPUs,
                  target_type,
                  target_arch,
                  gateway_mode,
                  supports_remote_buffers,
                  is_module_which_can_stall,
                  TF_MAJOR_VERSION,
                  TF_MINOR_VERSION,
                  tf_git_version(),
                  replication_factor,
                  annotations.infeed_infos,
                  annotations.outfeed_infos,
                  annotations.send_infos,
                  annotations.recv_infos,
                  annotations.host_embedding_lookup_infos,
                  annotations.host_embedding_update_infos,
                  annotations.host_embedding_notify_infos,
                  annotations.remote_parameter_infos,
                  annotations.remote_parameter_host_rearrangements,
                  annotations.entry_input_infos,
                  annotations.feed_input_infos,
                  annotations.entry_output_infos,
                  annotations.feed_output_infos,
                  logging_cycle_count}));

          if (in_precompile_mode) {
            LOG(INFO) << "A pre-compiled Poplar program has been saved to "
                      << filenames.CachedExecutableFilename();
          }
        }

        if (poplar_executor->EnableSerialization()) {
          TF_RETURN_IF_ERROR(
              poplar_executor->CreateSerializedExecutableDirIfMissing());

          TF_RETURN_IF_ERROR(PoplarExecutableCore::Export(
              filenames, exec, resources, replication_factor,
              {} /* device_opts */, opt_flags,
              poplar_executor->GetOrCreatePoplarTarget()));
        }
      } else {
        if (executable_cache_lock) {
          LOG(INFO)
              << "Executable caching has been enabled, however the "
                 "program being compiled contains custom user operations "
                 "which cannot be safely serialized. If it is safe to "
                 "pre-compile these custom user operations, please set "
                 "the 'is_hashable' attribute to 'true' in the "
                 "operations metadata function. Please see the documentation "
                 "for further details.";
        }
      }

      engine = absl::make_unique<poplar::Engine>(std::move(exec), opt_flags);
      VLOG(1) << "End compiling Poplar engine.";

    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Compile engine]", e);
    }

    if (enable_trace_events && compile) {
      uint64 duration = tensorflow::Env::Default()->NowMicros() - start_micros;

      TF_ASSIGN_OR_RETURN(auto inst_info,
                          GetInstructionCompilationInfo(module, resources));

      std::string map_json = GetTensorMappingJson(module->name(), main_graph,
                                                  resources.tensor_maps);

      poplar_executor->AddCompileEndEventRecord(module->name(), map_json,
                                                inst_info, duration);
    }
  }

  resources.progress_bar->Finish();

  std::unique_ptr<PoplarExecutableCore> executable_core =
      absl::make_unique<PoplarExecutableCore>(
          std::move(engine),
          std::move(resources.annotations.input_output_aliasing_map),
          is_constant_graph, std::move(constant_output), is_remap_graph,
          is_scalar_elementwise_graph,
          /*loaded_from_cache=*/false, std::move(remaped_output),
          std::move(resources.annotations.stream_infos),
          std::move(resources.annotations.stream_meta_infos),
          PoplarExecutableInfo{
              num_IPUs,
              target_type,
              target_arch,
              gateway_mode,
              supports_remote_buffers,
              is_module_which_can_stall,
              TF_MAJOR_VERSION,
              TF_MINOR_VERSION,
              tf_git_version(),
              replication_factor,
              std::move(resources.annotations.infeed_infos),
              std::move(resources.annotations.outfeed_infos),
              std::move(resources.annotations.send_infos),
              std::move(resources.annotations.recv_infos),
              std::move(resources.annotations.host_embedding_lookup_infos),
              std::move(resources.annotations.host_embedding_update_infos),
              std::move(resources.annotations.host_embedding_notify_infos),
              std::move(resources.annotations.remote_parameter_infos),
              std::move(
                  resources.annotations.remote_parameter_host_rearrangements),
              std::move(resources.annotations.entry_input_infos),
              std::move(resources.annotations.feed_input_infos),
              std::move(resources.annotations.entry_output_infos),
              std::move(resources.annotations.feed_output_infos),
              logging_cycle_count});

  return executable_core;
}
}  // namespace

StatusOr<std::unique_ptr<HloModule>> PoplarCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    se::DeviceMemoryAllocator* device_allocator) {
  TENSORFLOW_TRACEPOINT();
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  TENSORFLOW_TRACEPOINT();
  if (stream_exec == nullptr) {
    return tensorflow::errors::Unknown(
        "NULL stream pointer in Poplar compiler");
  }

  if (PoplarXlaFlags::Get().help) {
    std::call_once(help_flag_printed, &PrintHelpString);
  }

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

  std::lock_guard<std::mutex> g(static_mu_);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      PoplarExecutableCache::GetInstance().GetOrCompileExecutable(
          std::move(module), std::move(profile_printer),
          std::move(profile_index_map), poplar_executor, CompileEngine));

  return executable;
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  TENSORFLOW_TRACEPOINT();
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
