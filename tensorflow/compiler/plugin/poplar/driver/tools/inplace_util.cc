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

#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"

#include <queue>
#include <stack>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace {

// Only add a dependency iff `to` was not already reachable from `from`.
void AddDependency(HloInstruction* from, HloInstruction* to,
                   HloReachabilityMap* reachability_map,
                   std::vector<HloInstruction*>& added_dependencies) {
  // If there already wasn't a control dependency then insert it
  if (!reachability_map->IsReachable(from, to)) {
    TF_CHECK_OK(from->AddControlDependencyTo(to));
    reachability_map->UpdateReachabilityThroughInstruction(to);
    added_dependencies.push_back(from);
  }
}

void RemoveDependencies(std::vector<HloInstruction*> froms, HloInstruction* to,
                        HloReachabilityMap* reachability_map) {
  for (auto* from : froms) {
    TF_CHECK_OK(from->RemoveControlDependencyTo(to));
  }
  reachability_map->UpdateReachabilityThroughInstruction(to);
}

// Returns true if user is of specific inplace type and uses inst at a inplace
// operand index.
bool IsUsedAsInplace(const HloInstruction* user, const HloInstruction* inst,
                     const HloInstructionType inplace_type) {
  auto user_description = HloInstructionDescription(user);
  if (user_description.GetType() != inplace_type) {
    return false;
  }
  auto use_indicies = user->OperandIndices(inst);
  std::vector<int64> inplace_indexes =
      user_description.GetInplaceOperandIndexes();
  std::vector<int64> intersection;

  absl::c_sort(use_indicies);
  absl::c_sort(inplace_indexes);
  absl::c_set_intersection(use_indicies, inplace_indexes,
                           std::back_inserter(intersection));
  return intersection.size();
}

bool IsUniqueOperand(HloInstruction* inplace, HloInstruction* inplace_parent) {
  return inplace->OperandIndices(inplace_parent).size() == 1;
}

bool IsNotDependencyOfPeers(HloInstruction* inplace,
                            HloInstruction* inplace_parent,
                            HloReachabilityMap* reachability_map,
                            std::vector<HloInstruction*>& added_dependencies) {
  for (auto* peer : inplace_parent->users()) {
    if (peer == inplace) {
      continue;
    }
    if (reachability_map->IsReachable(inplace, peer)) {
      return false;
    } else {
      AddDependency(peer, inplace, reachability_map, added_dependencies);
    }
  }
  return true;
}

// A function which is used to decide whether the instruction is inplace
// get tuple element type given our backend implementation of these ops in
// Poplar.
bool ConvertToInplaceGetTupleElement(HloInstruction* inst) {
  // An instruction is inplace get tuple element (GTE) if:
  // 1. It has an inplace GTE type, and
  // 2. All other users of operand 0 (peers) are GTEs and there is no other
  // GTE with the same GTE index.
  auto inplace_desc = HloInstructionDescription(inst);

  // Verify it is inplace read/write (cond 1).
  if (inplace_desc.GetType() != HloInstructionType::kInplaceGetTupleElement) {
    return false;
  }

  for (auto* peer : inst->operand(0)->users()) {
    if (peer == inst) {
      continue;
    }
    if (peer->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    if (peer->tuple_index() == inst->tuple_index()) {
      return false;
    }
  }
  return true;
}

// A function which is used to decide whether the instruction is inplace
// read-write type given our backend implementation of these ops in Poplar and
// the current reachability graph.
bool ConvertToInplaceReadWrite(HloInstruction* inst,
                               HloReachabilityMap* reachability_map) {
  // An instruction is inplace read/write if:
  // 1. It has an inplace read/write type, and
  // 2. For each inplace operand instruction, instruction is not a dependency
  // of peer (users of the same operands) and
  // 3. It is not using the output of the the ROOT instruction.
  auto inplace_desc = HloInstructionDescription(inst);

  // Verify it is inplace read/write (cond 1).
  if (inplace_desc.GetType() != HloInstructionType::kInplaceReadWrite) {
    return false;
  }

  // Keep track of all control dependencies we add.
  std::vector<HloInstruction*> added_dependencies;

  bool is_inplace = true;
  // Go through all the inplace operands.
  for (auto op_idx : inplace_desc.GetInplaceOperandIndexes()) {
    HloInstruction* op = inst->mutable_operand(op_idx);
    // We expect all the inplace operands to be only used once as an operand.
    if (!IsUniqueOperand(inst, op)) {
      is_inplace = false;
      break;
    }
    // Verify that inplace is not a dependency of any of the peers (cond 2).
    if (!IsNotDependencyOfPeers(inst, op, reachability_map,
                                added_dependencies)) {
      is_inplace = false;
      break;
    }
    // The ROOT of a computation has an implicit user (the caller of the
    // computation). Therefore, root instructions with other users cannot be
    // inplace.
    if (op->parent()->root_instruction() == op) {
      is_inplace = false;
    }
  }

  if (!is_inplace) {
    // If we can't make this op inplace, remove all the dependencies which we
    // have added.
    RemoveDependencies(added_dependencies, inst, reachability_map);
  }
  return is_inplace;
}

// A function which is used to decide whether the instruction is inplace
// read-only type given our backend implementation of these ops in Poplar and
// the current reachability graph.
bool ConvertToInplaceReadOnly(HloInstruction* inst,
                              HloReachabilityMap* reachability_map,
                              InplaceWorkList& worklist) {
  // For read only instructions, not only do we need to consider whether `inst`
  // is inplace read/only, but we also need to consider the indirect source of
  // inst and all the indirect consumers of it.
  // For example in the following graph:
  //   x
  //   |\___________________________________________________________________
  //   |                                            |                       |
  // a = reshape(x)                             b = broadcast(x)   c = negate(x)
  //   |__________________                __________|___________
  //   |                  |              |                      |
  // d = reshape(a)  e = log(a)       f = slice(b)         g = slice(b)
  //   |                  |                  |                      |
  // h = not-inplace(f) i = not-inplace(g) j = not-inplace(f) k = not-inplace(g)
  //
  // Where c and e have been marked as inplace (read-write) ops.
  // Ops a, b, d, f, g are potential inplace read-only ops.
  // Ops h, i, j, k are ops which are not inplace.
  // Because x is the source to all the inplace read-only ops, when trying to
  // classify any of them we need to classify them all.
  // To do so we identify the following:
  // * Inplace read-only cluster - all the inplace read-only ops which
  //   (indirectly) consume the same tensors.
  // * Cluster sources - all the tensors which are used inside the read-only
  //   cluster and are not inplace read-only.
  // * Inplace read/write users - inplace consumers of the cluster sources which
  //   modify the tensor.
  // * Not inplace users - consumers of the cluster output which do not modify
  //   the tensor.
  // In the above graph we get the following:
  // * Inplace read-only cluster: a, b, d, f, g
  // * Cluster sources: x
  // * Inplace read/write users - c, e
  // * Not inplace users - h, i, j, k
  // We apply the following constrain to decide whether all the ops in the
  // inplace read-only cluster are inplace.
  // * If there are no inplace read/write users, then all the ops in the
  //    cluster are inplace.
  // * Else if there is one inplace read/write user, and it only has one use
  //   of an inplace read-only op and all the not inplace users can be executed
  //   before that inplace op to avoid overwriting of the tensor, then all the
  //   ops in the cluster are inplace.
  // * Otherwise we mark all the inplace read-only ops which consume the cluster
  //   sources as not inplace cluster sources and try again.
  // In the above graph we have more than one inplace read/write user, we
  // therefore mark a and b as not inplace which will result in a copy.
  // This results in two smaller clusters being created:
  // a = reshape(x)                             b = broadcast(x)
  //   |__________________                __________|___________
  //   |                  |              |                      |
  // d = reshape(a)  e = log(a)       f = slice(b)         g = slice(b)
  //   |                  |                  |                      |
  // h = not-inplace(f) i = not-inplace(g) j = not-inplace(f) k = not-inplace(g)
  // Cluster 1:
  // * Inplace read-only cluster: d
  // * Cluster sources: a
  // * Inplace read/write users - e
  // * Not inplace users - h, i
  // Cluster 2:
  // * Inplace read-only cluster: f, g
  // * Cluster sources: b
  // * Inplace read/write users - None
  // * Not inplace users - j, k
  // In Cluster 1, we mark d as inplace read-only iff h can be executed before
  // e (which is inplace read/write).
  // In Cluster 2, we mark f and g as inplace read-only as there are no inplace
  // read/write users.
  auto inplace_desc = HloInstructionDescription(inst);

  if (inplace_desc.GetType() != HloInstructionType::kInplaceReadOnly) {
    return false;
  }

  while (!worklist.contains(inst)) {
    // Build the cluster from inst.
    // Stores all the read only instructions found.
    absl::flat_hash_set<HloInstruction*> read_only_cluster;

    // Stores all the inputs to the cluster which are independent of each other.
    absl::flat_hash_set<HloInstruction*> cluster_sources;

    // All the uses of any cluster nodes in inplace read/write op.
    absl::flat_hash_map<HloInstruction*, uint64> inplace_read_write_users;

    // All the non inplace uses of any cluster nodes.
    HloInstructionSet not_inplace_users;

    std::queue<HloInstruction*> to_visit;
    absl::flat_hash_set<HloInstruction*> visited;
    to_visit.push(inst);

    while (!to_visit.empty()) {
      HloInstruction* node = to_visit.front();
      to_visit.pop();
      // Do not consider ops which we have already visited.
      if (visited.contains(node)) {
        continue;
      }
      visited.insert(node);

      auto node_description = HloInstructionDescription(node);

      // First extend the cluster by traversing from the current node to its
      // users. Note that at this point the node can be any HloInstructionType.
      // Go through all the users which use this node in a inplace operand
      // position.
      for (auto* user : node->users()) {
        auto user_description = HloInstructionDescription(user);
        if (IsUsedAsInplace(user, node, HloInstructionType::kInplaceReadOnly)) {
          // If a kInplaceReadOnly user is using the current node as an inplace
          // input, then we want to add it to the cluster.
          to_visit.push(user);
        } else if (IsUsedAsInplace(user, node,
                                   HloInstructionType::kInplaceReadWrite)) {
          if (IsLoweredInplace(user)) {
            // If a kInplaceReadWrite user is using the current node as an
            // inplace input, then add it to inplace_read_write_users.
            inplace_read_write_users[user]++;
          } else {
            // If the inplace read/write instruction is marked as not inplace,
            // then we mark it as not inplace.
            not_inplace_users.insert(user);
          }
        } else {
          // Otherwise add the node as non inplace user.
          not_inplace_users.insert(user);
        }
      }

      // Now extend the cluster by traversing the operands.
      // We only extend the cluster if the current node is kInplaceReadOnly
      // type and it is not in the worklist, otherwise we classify the current
      // node as a cluster source.
      if (node_description.GetType() == HloInstructionType::kInplaceReadOnly &&
          !worklist.contains(node)) {
        // Go through all the inplace operands.
        for (auto op_idx : node_description.GetInplaceOperandIndexes()) {
          auto* operand = node->mutable_operand(op_idx);
          to_visit.push(operand);
        }
        read_only_cluster.insert(node);
      } else {
        cluster_sources.insert(node);
      }
    }
    // All ops in the cluster can be inplace if there are no inplace read/write
    // users.
    bool cluster_ok_to_inplace = inplace_read_write_users.empty();
    // The ops in the cluster can also be inplace if there is one inplace use of
    // any tensors in the cluster and all the not inplace users can be executed
    // before the inplace op.
    if (inplace_read_write_users.size() == 1 &&
        std::begin(inplace_read_write_users)->second == 1) {
      auto* inplace_op = std::begin(inplace_read_write_users)->first;
      std::vector<HloInstruction*> added_dependencies;
      bool can_execute_before_all_other_users = true;
      // Check that we can execute all the users before the inplace op.
      for (auto* user : not_inplace_users) {
        if (!reachability_map->IsReachable(inplace_op, user)) {
          AddDependency(user, inplace_op, reachability_map, added_dependencies);
        } else {
          can_execute_before_all_other_users = false;
          break;
        }
      }

      if (can_execute_before_all_other_users) {
        cluster_ok_to_inplace = true;
      } else {
        // Remove all the dependencies which were added.
        RemoveDependencies(added_dependencies, inplace_op, reachability_map);
      }
    }

    if (cluster_ok_to_inplace) {
      for (auto* op : read_only_cluster) {
        worklist[op] = true;
      }
    } else {
      // Mark all the inplace read-only ops which consume the cluster sources as
      // not inplace and try again.
      for (auto* cluster_source : cluster_sources) {
        for (auto* user : cluster_source->users()) {
          if (read_only_cluster.contains(user)) {
            worklist[user] = false;
          }
        }
      }
    }
  }
  return worklist.at(inst);
}

}  // namespace

HloInstructionDescription::HloInstructionDescription(
    const HloInstruction* inst) {
  switch (inst->opcode()) {
    // Inplace read/write ops.
    // Unary Elementwise ops - inplace on operand 0.
    case HloOpcode::kAbs:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLogistic:
    case HloOpcode::kLog1p:
    case HloOpcode::kLog:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
    // Binary Elementwise ops - inplace on operand 0.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    // These ops are implemented as inplace ops on operand 0 as well.
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kScatter: {
      // All of the above ops are inplace on operand 0.
      type_ = HloInstructionType::kInplaceReadWrite;
      inplace_operands_ = {0};
      break;
    }

    // Inplace on all operands.
    case HloOpcode::kAllReduce:
    case HloOpcode::kMap:
    case HloOpcode::kSort:
    case HloOpcode::kTuple: {
      OperandIndexes indexes(inst->operand_count());
      absl::c_iota(indexes, 0);
      type_ = HloInstructionType::kInplaceReadWrite;
      inplace_operands_ = indexes;
      break;
    }

    case HloOpcode::kFusion: {
      if (IsPopOpsFusion(inst)) {
        auto fusion_config = inst->backend_config<PoplarBackendConfig>()
                                 .ValueOrDie()
                                 .fusion_config();
        auto inplace_descriptions = fusion_config.inplace_descriptions();
        for (const auto& inplace_description : inplace_descriptions) {
          inplace_operands_.push_back(
              HloPoplarUseDescription::FromProto(inplace_description)
                  .operand_number());
        }
        absl::c_sort(inplace_operands_);
        if (inplace_operands_.size()) {
          type_ = HloInstructionType::kInplaceReadWrite;
        } else {
          type_ = HloInstructionType::kNotInplace;
        }
      } else {
        // A non poplibs fusion is inplace on all operands.
        OperandIndexes indexes(inst->operand_count());
        absl::c_iota(indexes, 0);
        type_ = HloInstructionType::kInplaceReadWrite;
        inplace_operands_ = indexes;
      }
      break;
    }

    case HloOpcode::kCall: {
      if (IsRepeatLoop(inst)) {
        OperandIndexes indexes;
        const int64 num_operands = inst->operand_count();
        const HloComputation* comp = inst->to_apply();
        const HloInstruction* root = comp->root_instruction();

        // The loop is considered to be inplace on all operands unless all it's
        // users are GTEs
        const bool all_users_gtes = absl::c_all_of(
            inst->users(), [](const HloInstruction* user) -> bool {
              return user->opcode() == HloOpcode::kGetTupleElement;
            });
        // The root instruction needs to be an inplace tuple - this makes sure
        // that an particular input is only used in a single place.
        // The loop also must have been broken up into individual inputs.

        // Check which inputs are actually modified.
        if (GetRepeatLoopAllowFinerAliasAnalysis(inst) &&
            IsLoweredInplace(root) && root->opcode() == HloOpcode::kTuple &&
            num_operands == root->operand_count() && all_users_gtes) {
          // Vector indiciating whether a given input/output index has a gte
          // output.
          std::vector<bool> has_gte(num_operands, false);
          for (const HloInstruction* user : inst->users()) {
            CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
            has_gte[user->tuple_index()] = true;
          }

          for (int64 idx = 0; idx != num_operands; ++idx) {
            // An operand is not inplace if there is no gte for it and it's used
            // directly in the root instruction at the same index.
            if (has_gte[idx] ||
                root->operand(idx) != comp->parameter_instruction(idx)) {
              indexes.push_back(idx);
            }
          }
        } else {
          // Inplace on all its inputs.
          indexes.resize(num_operands);
          absl::c_iota(indexes, 0);
        }

        inplace_operands_ = indexes;
        type_ = HloInstructionType::kInplaceReadWrite;
      } else if (IsPipelineOp(inst) || IsResourceUpdate(inst)) {
        // Pipeline and ResourceUpdate operations are inplace on all
        // their inputs.
        OperandIndexes indexes(inst->operand_count());
        absl::c_iota(indexes, 0);
        type_ = HloInstructionType::kInplaceReadWrite;
        inplace_operands_ = indexes;
      } else if (IsAnyPipelineStageOp(inst)) {
        // Pipeline stages are only inplace on operands which are not
        // parameters/execution counters.

        // Backward pipeline stages don't mark gradient accumulators as inplace
        // inputs.
        const bool is_bwd = IsPipelineStageBackward(inst);

        HloComputation* comp = inst->to_apply();
        for (int64 op_idx = 0; op_idx != inst->operand_count(); ++op_idx) {
          const HloInstruction* operand = inst->operand(op_idx);
          if (!IsPipelineStageReadOnlyInput(operand) &&
              !(is_bwd && IsPoplarInstruction(
                              PoplarOp::GradientAccumulatorCreate)(operand))) {
            // If the stage modifies the input inplace, add it as an inplace
            // operand.
            if (IsOutputModifiedInplace(comp->parameter_instruction(op_idx))) {
              inplace_operands_.push_back(op_idx);
            }
          }
        }
        type_ = HloInstructionType::kInplaceReadWrite;
      } else if (IsFunction(inst)) {
        // Functions are inplace on remote buffer inputs.
        // Assume that the first "num_modified_remote_buffers" inputs are remote
        // buffers which are modified and they are also the first
        // "num_modified_remote_buffers" outputs.
        // Assume that the next "num_unmodified_remote_buffers" inputs are
        // remote buffers which are only loaded.
        const int64 num_modified_remote_buffers =
            GetFunctionNumberModifiedRemoteBufferInputs(inst);
        const int64 num_unmodified_remote_buffers =
            GetFunctionNumberUnmodifiedRemoteBufferInputs(inst);
        // TODO(T10387): consider unmodified remote buffers as read only.
        if (num_modified_remote_buffers + num_unmodified_remote_buffers) {
          OperandIndexes indexes(num_modified_remote_buffers +
                                 num_unmodified_remote_buffers);
          absl::c_iota(indexes, 0);
          type_ = HloInstructionType::kInplaceReadWrite;
          inplace_operands_ = indexes;
        } else {
          type_ = HloInstructionType::kNotInplace;
        }
      } else {
        // Calls are not inplace.
        type_ = HloInstructionType::kNotInplace;
      }
      break;
    }

    case HloOpcode::kWhile: {
      // Inplace on it's input tuple.
      CHECK_EQ(inst->operand_count(), 1);
      type_ = HloInstructionType::kInplaceReadWrite;
      inplace_operands_ = {0};
      break;
    }

    case HloOpcode::kCustomCall: {
      if (IsPoplibsHloCustomOp(inst)) {
        auto poplar_inst = Cast<HloPoplarInstruction>(inst);

        const auto use_descriptions = poplar_inst->GetUseDescriptions();

        if (use_descriptions.size()) {
          for (const HloPoplarUseDescription& description : use_descriptions) {
            inplace_operands_.push_back(description.operand_number());
          }
          absl::c_sort(inplace_operands_);
          type_ = HloInstructionType::kInplaceReadWrite;
        } else {
          type_ = HloInstructionType::kNotInplace;
        }
      } else {
        OperandIndexes indexes(inst->operand_count());
        absl::c_iota(indexes, 0);
        type_ = HloInstructionType::kInplaceReadWrite;
        inplace_operands_ = indexes;
      }
      break;
    }

    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor: {
      // Only inplace if their input/output type is the same.
      if (inst->shape().element_type() ==
          inst->operand(0)->shape().element_type()) {
        type_ = HloInstructionType::kInplaceReadWrite;
        inplace_operands_ = {0};
      } else {
        type_ = HloInstructionType::kNotInplace;
      }
      break;
    }

    // Inplace read-only ops.
    // These ops are implemented as inplace ops on operand 0.
    case HloOpcode::kAddDependency:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      // All of the above ops are inplace on operand 0.
      type_ = HloInstructionType::kInplaceReadOnly;
      inplace_operands_ = {0};
      break;
    }
    // Inplace ops on the first 2 ops.
    case HloOpcode::kPad: {
      type_ = HloInstructionType::kInplaceReadOnly;
      inplace_operands_ = {0, 1};
      break;
    }
    // Inplace on all operands.
    case HloOpcode::kConcatenate: {
      OperandIndexes indexes(inst->operand_count());
      absl::c_iota(indexes, 0);
      type_ = HloInstructionType::kInplaceReadOnly;
      inplace_operands_ = indexes;
      break;
    }

    // kInplaceGetTupleElement
    case HloOpcode::kGetTupleElement: {
      type_ = HloInstructionType::kInplaceGetTupleElement;
      inplace_operands_ = {0};
      break;
    }

    // Not inplace ops.
    case HloOpcode::kAfterAll:
    case HloOpcode::kAllToAll:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCholesky:
    case HloOpcode::kCompare:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvert:
    case HloOpcode::kConvolution:
    case HloOpcode::kCopy:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kReduce:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRng:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kTupleSelect: {
      type_ = HloInstructionType::kNotInplace;
      break;
    }

    // TODO(T20398): Clamp and Select could be inplace on operand index 1.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      type_ = HloInstructionType::kNotInplace;
      break;
    }

    // Unimplemented ops.
    case HloOpcode::kBitcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kFft:
    case HloOpcode::kTrace:
    default: {
      LOG(FATAL) << "Unrecognized op " << inst->opcode()
                 << ". Classify whether it is an inplace op or not";
      type_ = HloInstructionType::kNotInplace;
    }
  }
  inplace_operands_set_ = absl::flat_hash_set<int64>{inplace_operands_.begin(),
                                                     inplace_operands_.end()};
}

const HloInstructionType& HloInstructionDescription::GetType() const {
  return type_;
}

const OperandIndexes& HloInstructionDescription::GetInplaceOperandIndexes()
    const {
  return inplace_operands_;
}

const OperandSet& HloInstructionDescription::GetInplaceOperandSet() const {
  return inplace_operands_set_;
}

bool HloInstructionDescription::IsInplaceType() const {
  switch (GetType()) {
    case HloInstructionType::kInplaceGetTupleElement:
    case HloInstructionType::kInplaceReadWrite:
    case HloInstructionType::kInplaceReadOnly:
      return true;
    case HloInstructionType::kNotInplace:
    default:
      return false;
  }
}

const std::string HloInstructionDescription::ToString() const {
  std::stringstream str_stream;
  str_stream << "type: ";
  switch (GetType()) {
    case HloInstructionType::kInplaceGetTupleElement: {
      str_stream << "inplace-get-tuple-element";
      break;
    }
    case HloInstructionType::kInplaceReadWrite: {
      str_stream << "inplace-read-write";
      break;
    }
    case HloInstructionType::kInplaceReadOnly: {
      str_stream << "inplace-read-only";
      break;
    }
    default:
    case HloInstructionType::kNotInplace: {
      str_stream << "not-inplace";
      break;
    }
  }
  if (GetInplaceOperandIndexes().size()) {
    str_stream << " inplace_operands: ";
    str_stream << absl::StrJoin(GetInplaceOperandIndexes(), ", ");
  }
  return str_stream.str();
}

bool HloInstructionDescription::ConvertToInplace(
    HloInstruction* inst, HloReachabilityMap* reachability_map,
    InplaceWorkList& worklist) {
  auto inst_description = HloInstructionDescription(inst);
  bool converted;
  switch (inst_description.GetType()) {
    case HloInstructionType::kInplaceGetTupleElement: {
      converted = ConvertToInplaceGetTupleElement(inst);
      break;
    }
    case HloInstructionType::kInplaceReadWrite: {
      converted = ConvertToInplaceReadWrite(inst, reachability_map);
      break;
    }
    case HloInstructionType::kInplaceReadOnly: {
      converted = ConvertToInplaceReadOnly(inst, reachability_map, worklist);
      break;
    }
    default: {
      converted = false;
      break;
    }
  }
  if (converted) {
    MakeUsedInplace(inst);
  }
  return converted;
}

absl::optional<HloInstruction*> GetInplaceModifier(HloInstruction* inst) {
  // Go through all the users, looking through inplace read-only/GTEs.
  std::stack<HloInstruction*> to_visit;
  to_visit.push(inst);
  absl::flat_hash_set<HloInstruction*> visited;
  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.top();
    to_visit.pop();
    if (visited.contains(inst)) {
      continue;
    }
    visited.insert(inst);
    for (HloInstruction* user : inst->users()) {
      auto inplace_description = HloInstructionDescription(user);
      // Returns true if `user` uses `inst` inplace.
      auto is_used_inplace = [inplace_description](HloInstruction* inst,
                                                   HloInstruction* user) {
        return absl::c_any_of(inplace_description.GetInplaceOperandIndexes(),
                              [&inst, &user](int64 inplace_idx) {
                                return user->operand(inplace_idx) == inst;
                              });
      };
      switch (inplace_description.GetType()) {
        case HloInstructionType::kInplaceReadWrite: {
          if (IsLoweredInplace(user) && is_used_inplace(inst, user)) {
            // We have found a user which will modify the output.
            return user;
          }
          break;
        }
        case HloInstructionType::kInplaceReadOnly:
        case HloInstructionType::kInplaceGetTupleElement: {
          if (IsLoweredInplace(user) && is_used_inplace(inst, user)) {
            // We need to check through this user.
            to_visit.push(user);
          }
          break;
        }
        default: { break; }
      }
    }
  }

  return absl::nullopt;
}

bool IsOutputModifiedInplace(HloInstruction* inst) {
  return GetInplaceModifier(inst).has_value();
}
}  // namespace poplarplugin
}  // namespace xla
