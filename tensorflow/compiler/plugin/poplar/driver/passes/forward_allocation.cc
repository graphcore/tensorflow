/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"

#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remap_deduce.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {
namespace poplarplugin {
bool ForwardAllocationGraphComparator::operator()(
    const HloInstruction* const& lhs, const HloInstruction* const& rhs) const {
  // Make sure parameters are last.
  const bool lhs_parameter = lhs->opcode() == HloOpcode::kParameter;
  const bool rhs_parameter = rhs->opcode() == HloOpcode::kParameter;
  if (lhs_parameter == rhs_parameter) {
    return HloPtrComparator()(lhs, rhs);
  }

  return rhs_parameter;
}

template <typename Predicate>
static ForwardAllocationGraph::MetaGraphSet reduce(
    const ForwardAllocationGraph::MetaGraphSet& values, Predicate pred) {
  // For some reason absl const iterator doesn't define begin and end - we use a
  // copy instead.
  ForwardAllocationGraph::MetaGraphSet result;
  absl::c_copy_if(values, std::inserter(result, std::begin(result)), pred);
  return result;
}

template <typename Predicate>
static absl::optional<HloInstruction*> reduce_to_one(
    const ForwardAllocationGraph::MetaGraphSet& values, Predicate pred) {
  auto result = reduce(values, pred);
  return result.size() == 1
             ? absl::optional<HloInstruction*>(*std::begin(result))
             : absl::nullopt;
}

// Returns a vector of instructions which we want to use as a target. Note that
// the order of the targets is in decreasing priority order, where we want to
// target bias adds first, then layer norms and then elementwise ops.
template <typename Predicate>
static std::vector<HloInstruction*> find_all_targets(
    const ForwardAllocationGraph::MetaGraphSet& values, Predicate pred) {
  auto insts = reduce(values, pred);

  auto biases =
      reduce(insts, [](HloInstruction* inst) { return IsPopOpsBiasAdd(inst); });
  auto norms = reduce(insts, [](HloInstruction* inst) {
    return IsNormInferenceOrTraining(inst);
  });

  // Add the instructions in order.
  std::vector<HloInstruction*> result;
  result.insert(std::end(result), std::begin(biases), std::end(biases));
  result.insert(std::end(result), std::begin(norms), std::end(norms));
  for (auto inst : insts) {
    if (biases.count(inst) == 0 && norms.count(inst) == 0) {
      result.push_back(inst);
    }
  }

  return result;
}

static bool output_and_all_operands_same_type(const HloInstruction* inst) {
  const PrimitiveType& type = inst->shape().element_type();
  for (auto* operand : inst->operands()) {
    if (type != operand->shape().element_type()) {
      return false;
    }
  }
  return true;
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops
static bool IsPrefixPathOk(const std::vector<HloInstruction*>& path,
                           const HloInstruction* source,
                           const HloInstruction* target) {
  const auto is_node_ok_on_path =
      [path, source, target](HloInstruction* inst, const unsigned path_idx,
                             const unsigned path_size) {
        const HloInstruction* op_source =
            (path_idx == 0) ? source : path[path_idx - 1];
        // Elementwise ops are ok.
        if (IsPopOpsElementwise(inst)) {
          // Unless both inst and the target are a binary elementwise operation
          // - this will force a shorter path as inst is also a valid target.
          // target is only valid if operands are different - it doesn't make
          // sense to map one operand the same as itself.
          if (IsPopOpsElementwiseBinaryOperandsDifferent(inst) &&
              IsPopOpsElementwiseBinary(target)) {
            return false;
          }
          if (inst->opcode() == HloOpcode::kConvert) {
            return true;
          }
          // Make sure the shapes match up.
          return output_and_all_operands_same_type(inst) &&
                 op_source->shape() == inst->shape();
        }
        if (IsPopOpsFusion(inst, "zero_pad")) {
          return output_and_all_operands_same_type(inst);
        }
        if (IsPopOpsFusion(inst, "implicit")) {
          return output_and_all_operands_same_type(inst) &&
                 op_source->shape() == inst->shape();
        }
        if (IsAnySliceApply(inst)) {
          // Only handle operand 0
          return inst->operand_index(op_source) == 0 &&
                 output_and_all_operands_same_type(inst);
        }
        switch (inst->opcode()) {
          case HloOpcode::kCopy:
          case HloOpcode::kConcatenate:
          case HloOpcode::kReshape:
          case HloOpcode::kTranspose:
            return output_and_all_operands_same_type(inst);
          case HloOpcode::kPad: {
            // Only handle operand 0
            return inst->operand_index(op_source) == 0 &&
                   output_and_all_operands_same_type(inst);
          }
          case HloOpcode::kSlice: {
            return IsUniformSingleDimSlice(inst);
          }
          default:
            break;
        }
        return false;
      };
  return ForwardAllocationGraph::IsPathOk(path, is_node_ok_on_path);
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops.
// We allow the suffix path to have a GTE at the end of the path.
// For valid paths, either returns the GTE index for the last node or 0.
static absl::optional<int64_t> IsSuffixPathOk(
    const std::vector<HloInstruction*>& path) {
  const auto is_node_ok_on_path = [](HloInstruction* inst,
                                     const unsigned path_idx,
                                     const unsigned path_size) {
    // Element-wise and bias ops are ok.
    if (IsPopOpsElementwise(inst) || IsPopOpsBiasAdd(inst)) {
      if (inst->opcode() == HloOpcode::kConvert) {
        return true;
      } else {
        return output_and_all_operands_same_type(inst);
      }
    }
    switch (inst->opcode()) {
      case HloOpcode::kGetTupleElement:
        // We only allow GTEs at the end of the path
        return path_idx == (path_size - 1);
      case HloOpcode::kCopy:
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        return output_and_all_operands_same_type(inst);
      default:
        break;
    }
    return false;
  };
  bool path_ok = ForwardAllocationGraph::IsPathOk(path, is_node_ok_on_path);
  if (!path_ok) {
    return absl::nullopt;
  }
  // Get the GTE index at the end of the path if there was one.
  return (path.size() >= 1 &&
          path.back()->opcode() == HloOpcode::kGetTupleElement)
             ? path.back()->tuple_index()
             : 0LL;
}

// An operation is layout sensitive if the allocation of one of its inputs
// requires us to be able to access a tensor and the corresponding
// HloInstruction which created another input.
static bool IsLayoutSensitiveTarget(const HloInstruction* target) {
  return IsPopOpsBiasAdd(target);
}

// An operation is layout dependent if the allocation of one of its inputs
// depends on the layout of another input tensor - note that unlike layout
// sensitive target, we do not need the access to the instruction which created
// the tensor on which we depend on.
// The input tensors should also be different - an input tensor's layout cannot
// depend on itself
static bool IsLayoutDependentTarget(const HloInstruction* target) {
  if (IsPopOpsElementwiseBinaryOperandsDifferent(target)) {
    return true;
  }

  if (IsPopOpsFusion(target, "implicit_binary") &&
      target->operand_count() == 2) {
    return true;
  }

  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      return true;
    case HloOpcode::kCustomCall: {
      if (IsPoplibsHloCustomOp(target)) {
        auto poplar_inst = Cast<HloPoplarInstruction>(target);
        auto layout_dependencies = poplar_inst->LayoutDependencies();
        return layout_dependencies.size();
      }
      break;
    }
    default:
      break;
  }
  return false;
}

// TODO - this should probably be in a more central location
static absl::optional<int64_t> GetLayoutSensitiveOperandIndex(
    const HloInstruction* target, const HloInstruction* operand,
    const HloInstruction* layout_producer) {
  const auto op_idx = target->operand_index(operand);
  if (IsPopOpsBiasAdd(target)) {
    CHECK_LT(op_idx, 2);
    return op_idx;
  }
  return absl::nullopt;
}

static absl::optional<std::pair<int64_t, int64_t>>
GetLayoutDependentOperandIndices(const HloInstruction* target,
                                 const HloInstruction* operand) {
  const auto op_idx = target->operand_index(operand);

  // Some PopOps elementwise binary ops have more than two inputs (for example
  // scaled inplace with a scalar) - we make sure that we only target the first
  // two operands.
  if (IsPopOpsElementwiseBinary(target) && op_idx < 2) {
    return std::make_pair(op_idx, (op_idx + 1) % 2);
  }

  // For implicit broadcast instruction, we only consider allocation for the
  // operand which is being broadcasted (and the other operand is not).
  if (IsPopOpsFusion(target, "implicit_binary") && target->shape().rank() > 1) {
    const int64_t other_op_idx = (op_idx + 1) % 2;
    const HloInstruction* other_operand = target->operand(other_op_idx);
    if (operand->shape().dimensions() != target->shape().dimensions() &&
        other_operand->shape().dimensions() == target->shape().dimensions()) {
      return std::make_pair(op_idx, other_op_idx);
    } else {
      return absl::nullopt;
    }
  }

  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      // Only a layout dependent target on operands index 1 and 2.
      if (op_idx == 1 || op_idx == 2) {
        return std::make_pair(op_idx, 0);
      }
      return absl::nullopt;
    case HloOpcode::kCustomCall: {
      if (IsPoplibsHloCustomOp(target)) {
        auto poplar_inst = Cast<HloPoplarInstruction>(target);
        auto layout_dependencies = poplar_inst->LayoutDependencies();
        auto itr = layout_dependencies.find(op_idx);
        if (itr != layout_dependencies.end()) {
          return *itr;
        }
        return absl::nullopt;
      }
      break;
    }
    default:
      break;
  }
  return absl::nullopt;
}

// Depth First Tree traversal from source to non-tuple outputs, traversing
// through GetTupleElement.
void ForwardAllocation::FlattenInputs(
    HloInstruction* inst,
    ForwardAllocationGraph::MetaGraphSet& deferred_inputs) {
  const Shape& shape = inst->shape();
  if (shape.IsTuple()) {
    // We can only defer allocation of tuples iff all the users of the inst are
    // unique GTEs with compatible sharding.
    absl::flat_hash_set<int64_t> tuple_indexes;
    for (HloInstruction* user : inst->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        auto tuple_index = user->tuple_index();
        if (tuple_indexes.contains(tuple_index)) {
          // We can't defer allocation here - we require GTEs to be unique.
          return;
        }
        if (user->has_sharding() || inst->has_sharding()) {
          // Make sure they both have sharding.
          if (!(user->has_sharding() && inst->has_sharding())) {
            return;
          }
          // We require compatible sharding - otherwise a copy would have to
          // take place which requires the tensor to be allocated.
          const auto& sharding = inst->sharding();
          const auto& tuple_sub_sharding =
              sharding.IsTuple()
                  ? sharding.GetSubSharding(inst->shape(), {tuple_index})
                  : sharding;
          if (tuple_sub_sharding != user->sharding()) {
            // We can't defer allocation here due to incompatible sharding.
            return;
          }
        }
        tuple_indexes.insert(tuple_index);
      } else {
        // We can't defer allocation here - we can only look through GTEs.
        return;
      }
    }
    for (HloInstruction* user : inst->users()) {
      // We have guaranteed that we are only looking through GTEs.
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
      // We can only look through if it's inplace.
      if (IsLoweredInplace(user)) {
        FlattenInputs(user, deferred_inputs);
      }
    }
  } else {
    deferred_inputs.insert(inst);
  }
}

// Inputs to the graph are non-tuple tensors which originate from constants,
// infeeds, parameters or recvs. To find such tensors we traverse through
// GetTupleElement instructions, keeping track of this path. For example, given
// following HLO computation:
// clang-format off
//%comp (arg0: (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2])) -> f32[1,4,4,2] {
// %arg0 = (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) parameter(0)
// %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=0
// %gte1 = f32[1,1,2,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=1
// %convolution.36.29 = f32[1,4,4,2] convolution(%gte0, %gte1), window={size=1x1}, dim_labels=b01f_01io->b01f
// %gte2 = (f32[1,2], f32[1,2]) get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=2
// %gte2.0 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=0
// %gte2.0_r = f32[2] reshape(%gte2.0)
// %gte2.1 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=1
// %gte2.1_r = f32[2] reshape(%gte2.1)
// %gte3 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=3
// %gte4 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=4
// ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %gte2.0_r, %gte2.1_r, %gte3, %gte4), epsilon=0.001, feature_index=3
//}
// clang-format on
// In this graph %arg0 is the input, but we traverse the graph and find that
// %gte0, %gte1, %gte2.0, %gte2.1, %gte3, %gte4 are the non tuple inputs and we
// find the forward allocations for those.
StatusOr<ForwardAllocationGraph::MetaGraphSet> ForwardAllocation::FindInputs(
    HloComputation* comp, CallGraph* call_graph) {
  ForwardAllocationGraph::MetaGraphSet deferred_inputs;

  auto call_graph_node = call_graph->GetNode(comp);
  auto& callsites = call_graph_node.caller_callsites();
  std::vector<int64_t> parameters_to_add;
  // In pipeliening, do not add a parameter as an input location, unless it was
  // a parameter/gradient accumulation buffer in the outer scope.
  if (callsites.size() != 1 ||
      !IsAnyPipelineStageOpOrResourceUpdate(callsites[0].instruction())) {
    parameters_to_add.resize(comp->num_parameters());
    absl::c_iota(parameters_to_add, 0);
  } else {
    HloInstruction* caller = callsites[0].instruction();
    for (int64_t i = 0; i != comp->num_parameters(); ++i) {
      const HloInstruction* operand = caller->operand(i);
      if (operand->opcode() == HloOpcode::kParameter ||
          IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(operand)) {
        parameters_to_add.push_back(i);
      }
    }
  }

  for (int64_t param_number : parameters_to_add) {
    FlattenInputs(comp->parameter_instruction(param_number), deferred_inputs);
  }

  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    bool is_input = false;
    switch (inst->opcode()) {
      case HloOpcode::kConstant:
      case HloOpcode::kInfeed:
      case HloOpcode::kReduce: {
        is_input = true;
        break;
      }
      case HloOpcode::kFusion: {
        is_input = IsReductionFusion(inst);
        break;
      }
      case HloOpcode::kCustomCall: {
        is_input = Cast<HloPoplarInstruction>(inst)->AllocatingOutput();
        break;
      }
      default: { break; }
    }
    if (is_input) {
      FlattenInputs(inst, deferred_inputs);
    }
  }
  return deferred_inputs;
}  // namespace

bool ForwardAllocation::CreateForwardAllocationTarget(
    HloReachabilityMap* reachability_map, HloInstruction* source,
    HloInstruction* target, const int64_t input_index,
    HloInstruction* layout_producer, const int64_t layout_output_index,
    const std::vector<HloInstruction*>& other_targets,
    const std::vector<HloInstruction*>& forward_path,
    const std::vector<HloInstruction*>& backward_path) {
  // Make sure that the layout producer can be executed before the
  // source - i.e. source is not reachable form the layout producer.
  if (reachability_map->IsReachable(source, layout_producer)) {
    return false;
  }
  TF_CHECK_OK(layout_producer->AddControlDependencyTo(source));
  reachability_map->UpdateReachabilityThroughInstruction(source);

  // Make sure that the target can be executed before all the other
  // independent targets with the new control dependency.
  // Keep track of any dependencies we add in case we have to undo them.
  std::vector<HloInstruction*> added_dependants;
  absl::flat_hash_set<HloInstruction*> backward_path_insts = {
      backward_path.begin(), backward_path.end()};
  bool dependencies_ok = true;
  for (auto new_dependent : other_targets) {
    if (new_dependent == target) {
      continue;
    }

    // If the other target is in the backward path then we don't need to worry
    // about it because the backward/prefix path has been validated.
    if (backward_path_insts.contains(new_dependent)) {
      continue;
    }

    if (reachability_map->IsReachable(new_dependent, target)) {
      dependencies_ok = false;
      break;
    }

    if (!reachability_map->IsReachable(target, new_dependent)) {
      // Try to add a control dependecy, if it fails we can't proceed.
      if (target->AddControlDependencyTo(new_dependent).ok()) {
        reachability_map->UpdateReachabilityThroughInstruction(new_dependent);
        added_dependants.push_back(new_dependent);
      } else {
        dependencies_ok = false;
        break;
      }
    }
  }

  if (!dependencies_ok) {
    // Remove all the added dependencies
    TF_CHECK_OK(layout_producer->RemoveControlDependencyTo(source));
    reachability_map->UpdateReachabilityThroughInstruction(source);
    for (auto inst : added_dependants) {
      TF_CHECK_OK(target->RemoveControlDependencyTo(inst));
      reachability_map->UpdateReachabilityThroughInstruction(inst);
    }
    return false;
  }

  std::vector<const HloInstruction*> c_forward_path(forward_path.begin(),
                                                    forward_path.end());
  std::vector<const HloInstruction*> c_backward_path(backward_path.begin(),
                                                     backward_path.end());
  TensorLocation src{source, 0};
  TensorTarget tensor_target{target,          input_index,
                             layout_producer, layout_output_index,
                             c_forward_path,  c_backward_path};
  tensor_allocation_map[src] = std::move(tensor_target);
  return true;
}

StatusOr<bool> ForwardAllocation::FindLayoutSensativeTargets(
    HloComputation* comp, std::set<const HloInstruction*>& ops_with_layout,
    CallGraph* call_graph) {
  bool found_target = false;

  // Check that there is work for us to do.
  if (!absl::c_any_of(comp->instructions(), IsLayoutSensitiveTarget)) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto deferred_allocation_inputs,
                      FindInputs(comp, call_graph));

  const auto is_input = [&deferred_allocation_inputs,
                         this](HloInstruction* inst) {
    auto itr = deferred_allocation_inputs.find(inst);
    if (itr != deferred_allocation_inputs.end()) {
      return tensor_allocation_map.find(TensorLocation{inst, 0}) ==
             tensor_allocation_map.end();
    }
    return false;
  };

  const auto is_layout_producer = [&ops_with_layout](HloInstruction* inst) {
    return ops_with_layout.count(inst);
  };

  const auto get_operands = [](HloInstruction* inst) {
    return inst->operands();
  };

  const auto g = ForwardAllocationGraph(comp->root_instruction(), get_operands);
  const auto layout_producing_ops = g.FindVertices(is_layout_producer);

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(comp);

  // Get everything that depends upon an op with a special layout
  const auto get_consumers = [&is_layout_producer, &g](HloInstruction* inst) {
    return g.FindConsumers(inst, [&is_layout_producer](HloInstruction* inst) {
      return !is_layout_producer(inst);
    });
  };
  const ForwardAllocationGraph layout_op_consumers(layout_producing_ops,
                                                   get_consumers);

  const auto alloc_dependencies = layout_op_consumers.Transpose();
  const auto source_ops = g.FindVertices(is_input);

  // Get everything that depends on a source op
  const auto get_source_consumers = [&is_layout_producer, &layout_producing_ops,
                                     &alloc_dependencies,
                                     &g](HloInstruction* inst) {
    return g.FindConsumers(inst,
                           [&is_layout_producer, &layout_producing_ops,
                            &alloc_dependencies](HloInstruction* inst) {
                             return !is_layout_producer(inst) &&
                                    !alloc_dependencies.contains(inst) &&
                                    layout_producing_ops.count(inst) == 0;
                           },
                           true);
  };
  const ForwardAllocationGraph source_consumers(source_ops,
                                                get_source_consumers);

  for (const auto& edges : source_consumers) {
    const auto& source = edges.first;
    if (!edges.second.empty()) {
      // Target is the op consuming the allocated tensor which is layout
      // sensitive.
      const auto is_valid_target = [&](HloInstruction* a) {
        return alloc_dependencies.contains(a) && IsLayoutSensitiveTarget(a);
      };
      std::vector<HloInstruction*> targets =
          find_all_targets(edges.second, is_valid_target);

      const auto shortest_paths_from_source = g.ShortestPathsFrom(source);

      for (HloInstruction* target : targets) {
        // Find layout producers for the target.
        // layout_producer is the op which produces the tensor whose layout is
        // important - it cannot have any allocation dependencies.
        const auto& itr = alloc_dependencies.find(target);
        // Skip if the target has not allocation dependencies or if the target
        // has no layout producer.
        if (itr == alloc_dependencies.end() || itr->second.empty()) {
          continue;
        }
        const auto is_not_alloc_dependency = [&](HloInstruction* a) {
          return !alloc_dependencies.contains(a);
        };
        // TODO we only allow a single layout producer at the moment.
        const auto optional_layout_producer =
            reduce_to_one(itr->second, is_not_alloc_dependency);
        if (!optional_layout_producer) {
          continue;
        }
        auto* layout_producer = *optional_layout_producer;

        // Try and find the shortest paths from/to target.
        auto optional_prefix = shortest_paths_from_source.To(target);
        auto optional_suffix = g.ShortestPath(layout_producer, target);
        if (!(optional_prefix && optional_suffix)) {
          continue;
        }
        auto prefix = *optional_prefix;
        auto suffix = *optional_suffix;
        // Only some operands are layout sensitive.
        auto optional_op_idx = GetLayoutSensitiveOperandIndex(
            target, prefix.rbegin()[1], layout_producer);
        if (optional_op_idx) {
          const auto op_idx = *optional_op_idx;
          // The paths don't contain the source or target instructions
          prefix.erase(prefix.begin());
          prefix.pop_back();
          suffix.erase(suffix.begin());
          suffix.pop_back();
          const auto prefix_path_ok = IsPrefixPathOk(prefix, source, target);
          const auto suffix_path_ok = IsSuffixPathOk(suffix);
          if (prefix_path_ok && suffix_path_ok) {
            if (source_consumers[source].count(layout_producer) == 0) {
              auto layout_output_idx = *suffix_path_ok;
              const bool created_target = CreateForwardAllocationTarget(
                  reachability_map.get(), source, target, op_idx,
                  layout_producer, layout_output_idx, targets, suffix, prefix);
              found_target |= created_target;
              if (created_target) {
                break;
              }
            }
          }
        }
      }
    }
  }
  return found_target;
}

StatusOr<bool> ForwardAllocation::FindLayoutDependentTargets(
    HloComputation* comp, CallGraph* call_graph) {
  bool found_target = false;

  // Check that there is work for us to do.
  if (!absl::c_any_of(comp->instructions(), IsLayoutDependentTarget)) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto deferred_allocation_inputs,
                      FindInputs(comp, call_graph));

  const auto is_input = [&deferred_allocation_inputs,
                         this](HloInstruction* inst) {
    auto itr = deferred_allocation_inputs.find(inst);
    if (itr != deferred_allocation_inputs.end()) {
      return tensor_allocation_map.find(TensorLocation{inst, 0}) ==
             tensor_allocation_map.end();
    }
    return false;
  };

  const auto get_operands = [](HloInstruction* inst) {
    return inst->operands();
  };

  const ForwardAllocationGraph g(comp->root_instruction(), get_operands);

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(comp);

  const auto source_ops = g.FindVertices(is_input);

  // Get everything that depends on a source op
  const auto get_source_consumers = [&g](HloInstruction* inst) {
    return g.FindConsumers(inst, [](HloInstruction*) { return true; }, true);
  };
  const ForwardAllocationGraph source_consumers(source_ops,
                                                get_source_consumers);

  for (const auto& edges : source_consumers) {
    const auto& source = edges.first;
    if (!edges.second.empty()) {
      // Target is the op consuming the allocated tensor which is layout
      // dependent.
      const auto is_valid_target = [&](HloInstruction* a) {
        return IsLayoutDependentTarget(a);
      };
      std::vector<HloInstruction*> targets =
          find_all_targets(edges.second, is_valid_target);

      const auto shortest_paths = g.ShortestPathsFrom(source);
      for (auto target : targets) {
        // Try and find the shortest paths to target.
        auto optional_prefix = shortest_paths.To(target);
        if (!optional_prefix) {
          continue;
        }
        auto prefix = *optional_prefix;
        // Only some operands are layout dependent.
        auto optional_op_idices =
            GetLayoutDependentOperandIndices(target, prefix.rbegin()[1]);
        if (!optional_op_idices) {
          continue;
        }
        int64_t op_idx, layout_operand_idx;
        std::tie(op_idx, layout_operand_idx) = *optional_op_idices;
        // The path don't contain the source or target instructions
        prefix.erase(prefix.begin());
        prefix.pop_back();
        auto layout_producer = target->mutable_operand(layout_operand_idx);
        // Check that the prefix path is one that we can traverse.
        const auto prefix_path_ok = IsPrefixPathOk(prefix, source, target);
        if (!prefix_path_ok) {
          continue;
        }

        const bool created_target = CreateForwardAllocationTarget(
            reachability_map.get(), source, target, op_idx, layout_producer, 0,
            targets, {}, prefix);
        found_target |= created_target;
        if (created_target) {
          break;
        }
      }
    }
  }
  return found_target;
}

ForwardAllocation::ForwardAllocation(CompilerAnnotations& annotations)
    : tensor_allocation_map(annotations.tensor_allocation_map) {}

StatusOr<bool> ForwardAllocation::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  bool found_target = false;

  // Stores all the ops which we have identified to have layouts.
  std::set<const HloInstruction*> ops_with_layout;
  // Add all the tensor allocation targets.
  for (auto& ta : tensor_allocation_map) {
    ops_with_layout.insert(ta.second.tgt);
  }

  for (const auto& computation : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(computation)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool found_sensative_target_in_computation,
                        FindLayoutSensativeTargets(computation, ops_with_layout,
                                                   call_graph.get()));
    found_target |= found_sensative_target_in_computation;
    TF_ASSIGN_OR_RETURN(
        bool found_dependent_target_in_computation,
        FindLayoutDependentTargets(computation, call_graph.get()));
    found_target |= found_dependent_target_in_computation;
  }

  if (found_target) {
    VLOG(2) << "After the ForwardAllocation:";
    DumpTensorAllocationMap(tensor_allocation_map, 2);
  }

  return found_target;
}

}  // namespace poplarplugin
}  // namespace xla
