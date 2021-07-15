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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"

#include <queue>
#include <set>
#include <stack>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

absl::optional<int64> GetOperandIndexForNodeId(
    const HloMatcherNode& pattern_node, const NodeId& operand_id) {
  auto it = absl::c_find(pattern_node.GetOperands(), operand_id);
  return it != pattern_node.GetOperands().end()
             ? absl::optional<int64>(
                   std::distance(pattern_node.GetOperands().begin(), it))
             : absl::nullopt;
}

bool IsValidCandidate(
    NodeId candidate_node_id, HloInstruction* candidate_inst,
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  // Make sure we haven't already tried this pairing.
  if (invalid_pairings[candidate_node_id].count(candidate_inst) != 0) {
    return false;
  }

  // This is not a valid candidate if we already matched this candidate NodeId.
  if (parital_matching.count(candidate_node_id) > 0) {
    return false;
  }

  // This is not a valid candidate if we already matched this candidate
  // instruction.
  auto it = absl::c_find_if(parital_matching,
                            [&](std::pair<NodeId, HloInstruction*> iso_pair) {
                              return iso_pair.second == candidate_inst;
                            });
  if (it != parital_matching.end()) {
    return false;
  }

  HloMatcherNode matched_node = pattern.GetPatternNodes()[candidate_node_id];

  // Check the node matches.
  if (!matched_node.Matches(candidate_inst)) {
    return false;
  }

  // Check that the operands match up
  auto operand_ids =
      pattern.GetNodesToOperandsMatcherGraph()[candidate_node_id];
  for (NodeId operand_id : operand_ids) {
    auto operand_idx = *GetOperandIndexForNodeId(matched_node, operand_id);

    auto it = parital_matching.find(operand_id);
    if (it == parital_matching.end()) {
      continue;
    }
    HloInstruction* operand_inst = it->second;
    if (candidate_inst->mutable_operand(operand_idx) != operand_inst) {
      return false;
    }
  }

  return true;
}

absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>
FindValidCandidates(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>> targets;
  // Find possible isomorphisms.
  // We first look for target nodes by traversing from an instruction to it's
  // operand.
  for (auto iso_pair : parital_matching) {
    NodeId node_id = iso_pair.first;
    HloInstruction* matched_inst = iso_pair.second;
    HloMatcherNode matcher_node = pattern.GetPatternNodes()[node_id];

    // Go through the operands and check whether they are valid isomorphisms.
    auto operand_ids = pattern.GetNodesToOperandsMatcherGraph()[node_id];
    for (NodeId operand_id : operand_ids) {
      auto operand_idx = *GetOperandIndexForNodeId(matcher_node, operand_id);
      HloInstruction* operand_inst = matched_inst->mutable_operand(operand_idx);
      if (IsValidCandidate(operand_id, operand_inst, parital_matching,
                           invalid_pairings, pattern)) {
        targets[operand_id].insert(operand_inst);
      }
    }
  }

  if (targets.empty()) {
    // If there are no targets, then traverse from an instruction to it's users.
    for (auto iso_pair : parital_matching) {
      NodeId operand_id = iso_pair.first;
      HloInstruction* matched_inst = iso_pair.second;

      auto node_ids = pattern.GetOperandsToNodesMatcherGraph()[operand_id];
      // We label every possible user with a node_id if the user uses the
      // matched_inst at correct index.
      for (NodeId node_id : node_ids) {
        HloMatcherNode matcher_node = pattern.GetPatternNodes()[node_id];
        for (HloInstruction* user_inst : matched_inst->users()) {
          auto operand_idx =
              *GetOperandIndexForNodeId(matcher_node, operand_id);

          if (operand_idx < user_inst->operand_count() &&
              user_inst->mutable_operand(operand_idx) == matched_inst) {
            if (IsValidCandidate(node_id, user_inst, parital_matching,
                                 invalid_pairings, pattern)) {
              targets[node_id].insert(user_inst);
            }
          }
        }
      }
    }
  }
  return targets;
}

// We determine whether a proposed state is valid iff given the current state
// and the proposed pairing, every predecessors of the proposed pairing has
// a valid pairing and that every successor has at least one valid pairing.
bool IsValidState(
    const NodeId candidate_node_id, const HloInstruction* candidate_inst,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  HloMatcherNode matcher_node = pattern.GetPatternNodes()[candidate_node_id];
  // Check that all the predecessors of the proposed pairing make sense.
  // For each operand we need a pairing which is not invalid.
  for (NodeId operand_id :
       pattern.GetNodesToOperandsMatcherGraph()[candidate_node_id]) {
    auto operand_idx = *GetOperandIndexForNodeId(matcher_node, operand_id);
    const HloInstruction* operand_inst = candidate_inst->operand(operand_idx);
    HloMatcherNode operand_node = pattern.GetPatternNodes()[operand_id];
    if (!operand_node.Matches(operand_inst) ||
        invalid_pairings[operand_id].contains(operand_inst)) {
      return false;
    }
  }

  // Check that all the successors of the proposed pairing make sense.
  // For each user we need at least one pairing which is not invalid.
  for (NodeId user_id :
       pattern.GetOperandsToNodesMatcherGraph()[candidate_node_id]) {
    bool has_valid_target = false;
    HloMatcherNode user_matcher_node = pattern.GetPatternNodes()[user_id];
    for (const HloInstruction* user_inst : candidate_inst->users()) {
      auto operand_idx =
          *GetOperandIndexForNodeId(user_matcher_node, candidate_node_id);

      if (operand_idx < user_inst->operand_count() &&
          user_inst->operand(operand_idx) == candidate_inst) {
        if (user_matcher_node.Matches(user_inst) &&
            !invalid_pairings[user_id].contains(user_inst)) {
          has_valid_target = true;
        }
      }
    }

    if (!has_valid_target) {
      return false;
    }
  }
  return true;
}

// We use the VF2 algorithm - published paper "An Improved Algorithm for
// Matching Large Graphs" by L. P. Cordella, P. Foggia, C. Sansone, M. Vento -
// with one difference being we are matching DAGs - we only ever visit nodes
// connected to the partial matching.
bool MatchDAGIsomorphism(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  // Base condition - we matched the pattern if we matched every node in it and
  // the graph is valid.
  if (parital_matching.size() == pattern.GetPatternNodes().size()) {
    // TODO A proof to show that this is not required.
    // A check that goes through the isomorphism and checks that the mapping is
    // correct.
    // For every iso mapping pair, go through the operands and check they match
    // up.
    for (auto iso_pair : parital_matching) {
      NodeId node_id = iso_pair.first;
      HloInstruction* inst = iso_pair.second;
      HloMatcherNode matched_node = pattern.GetPatternNodes()[node_id];

      for (unsigned operand_idx = 0;
           operand_idx <
           pattern.GetPatternNodes()[node_id].GetOperands().size();
           operand_idx++) {
        NodeId operand_id =
            pattern.GetPatternNodes()[node_id].GetOperands()[operand_idx];
        HloInstruction* operand_inst = parital_matching[operand_id];
        if (inst->mutable_operand(operand_idx) != operand_inst) {
          return false;
        }
      }
    }
    return true;
  } else {
    // Given the current state, find nodes that when matched also give a valid
    // state and traverse those.
    auto candidates =
        FindValidCandidates(parital_matching, invalid_pairings, pattern);
    for (auto pair : candidates) {
      NodeId candidate_node_id = pair.first;
      for (HloInstruction* candidate_inst : pair.second) {
        // Search space pruning - Check that the state we are about to make is
        // valid.
        if (IsValidState(candidate_node_id, candidate_inst, invalid_pairings,
                         pattern)) {
          // Match the DAG with the new pairing candidate_node_id <->
          // candidate_inst added.
          parital_matching[candidate_node_id] = candidate_inst;

          if (MatchDAGIsomorphism(parital_matching, invalid_pairings,
                                  pattern)) {
            return true;
          }

          // If this was not a successful match then we need to remove it from
          // partial matching.
          parital_matching.erase(candidate_node_id);
        }
        // If this state is not valid, then we can use this pairing to prune
        // the search space.
        invalid_pairings[candidate_node_id].insert(candidate_inst);
      }
    }
    // If this partial matching was not valid, then we need to restore
    // invalid_pairings to the original state, as these pairings are *only*
    // invalid with the current invalid partial matching.
    for (auto pair : candidates) {
      NodeId candidate_node_id = pair.first;
      for (HloInstruction* candidate_inst : pair.second) {
        invalid_pairings[candidate_node_id].erase(candidate_inst);
      }
    }
    return false;
  }
}

// Start of the DAG match - we start the search from output[0].
bool MatchDAGIsomorphism(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    HloInstruction* first_output, const HloMatcherPattern& pattern) {
  NodeId first_output_node_id = pattern.GetOutputs()[0];
  absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>
      invalid_pairings;
  if (IsValidCandidate(first_output_node_id, first_output, parital_matching,
                       invalid_pairings, pattern)) {
    // Add the initial matching for the first output.
    parital_matching[first_output_node_id] = first_output;
    return MatchDAGIsomorphism(parital_matching, invalid_pairings, pattern);
  }
  return false;
}

}  // namespace

// HloMatcherOpcodeTarget
HloMatcherOpcodeTarget::HloMatcherOpcodeTarget(const HloOpcode& opcode)
    : opcode_(opcode){};
HloMatcherOpcodeTarget::HloMatcherOpcodeTarget(const HloMatcherOpcode& opcode)
    : opcode_(opcode){};

const bool HloMatcherOpcodeTarget::IsHloOpcode() const {
  return opcode_.index() == 0;
}

const HloOpcode HloMatcherOpcodeTarget::GetHloOpcode() const {
  return absl::get<HloOpcode>(opcode_);
}

const bool HloMatcherOpcodeTarget::IsHloMatcherOpcode() const {
  return !IsHloOpcode();
}

const HloMatcherOpcode HloMatcherOpcodeTarget::GetHloMatcherOpcode() const {
  return absl::get<HloMatcherOpcode>(opcode_);
}

// HloMatcherNode
HloMatcherNode::HloMatcherNode(HloMatcherOpcodeTarget opcode_target,
                               NodeOperands operands)
    : opcode_target_(opcode_target),
      operands_(operands),
      node_conditions_({}) {}

HloMatcherNode::HloMatcherNode(HloMatcherOpcodeTarget opcode_target,
                               NodeOperands operands,
                               NodeCondition node_condition)
    : opcode_target_(opcode_target),
      operands_(operands),
      node_conditions_({node_condition}) {}

HloMatcherNode::HloMatcherNode(
    HloMatcherOpcodeTarget opcode_target, NodeOperands operands,
    const std::vector<NodeCondition>& node_conditions)
    : opcode_target_(opcode_target),
      operands_(operands),
      node_conditions_(node_conditions) {}

const HloMatcherOpcodeTarget& HloMatcherNode::GetOpcodeTarget() const {
  return opcode_target_;
}

const NodeOperands& HloMatcherNode::GetOperands() const { return operands_; }

const std::vector<NodeCondition>& HloMatcherNode::GetNodeConditions() const {
  return node_conditions_;
}

const bool HloMatcherNode::Matches(const HloInstruction* inst) const {
  bool opcode_match = false;
  // If the target is an opcode, then it must match
  if (GetOpcodeTarget().IsHloOpcode()) {
    opcode_match = GetOpcodeTarget().GetHloOpcode() == inst->opcode();
  } else {
    switch (GetOpcodeTarget().GetHloMatcherOpcode()) {
      case HloMatcherOpcode::kAnyOpcode: {
        opcode_match = true;
        break;
      }
      default: {
        opcode_match = false;
        break;
      }
    }
  }
  if (opcode_match) {
    return absl::c_all_of(GetNodeConditions(),
                          [inst](const NodeCondition& condition) -> bool {
                            return condition(inst);
                          });
  } else {
    return false;
  }
}

HloMatcherPattern::HloMatcherPattern(PatternType type,
                                     PatternMetaTarget meta_target,
                                     PatternInputs inputs,
                                     PatternOutputs outputs,
                                     Pattern pattern_nodes)
    : HloMatcherPattern(type, PatternReplaceFn(), meta_target, inputs,
                        PatternInplaceDescriptionFn(), outputs, pattern_nodes) {
}

HloMatcherPattern::HloMatcherPattern(PatternType type,
                                     PatternReplaceFn replace_fn,
                                     PatternMetaTarget meta_target,
                                     PatternInputs inputs,
                                     PatternOutputs outputs,
                                     Pattern pattern_nodes)
    : HloMatcherPattern(type, replace_fn, meta_target, inputs,
                        PatternInplaceDescriptionFn(), outputs, pattern_nodes) {
}

HloMatcherPattern::HloMatcherPattern(
    PatternType type, PatternMetaTarget meta_target, PatternInputs inputs,
    PatternOutputs outputs, PatternInplaceDescriptionFn inplace_description_fn,
    Pattern pattern_nodes)
    : HloMatcherPattern(type, PatternReplaceFn(), meta_target, inputs,
                        inplace_description_fn, outputs, pattern_nodes) {}

HloMatcherPattern::HloMatcherPattern(
    PatternType type, PatternReplaceFn replace_fn,
    PatternMetaTarget meta_target, PatternInputs inputs,
    PatternInplaceDescriptionFn inplace_description_fn, PatternOutputs outputs,
    Pattern pattern_nodes)
    : type(type),
      replace_fn(replace_fn),
      meta_target(meta_target),
      inputs(inputs),
      inplace_description_fn(inplace_description_fn),
      outputs(outputs),
      pattern_nodes(pattern_nodes),
      pattern_graphs(VerifyAndGetGraphs()) {}

const PatternType& HloMatcherPattern::GetType() const { return type; }

const PatternReplaceFn& HloMatcherPattern::GetReplaceFn() const {
  return replace_fn;
}

const PatternMetaTarget& HloMatcherPattern::GetMetaTarget() const {
  return meta_target;
}

const PatternInputs& HloMatcherPattern::GetInputs() const { return inputs; }

const PatternInplaceDescriptionFn& HloMatcherPattern::GetInplaceDescriptionFn()
    const {
  return inplace_description_fn;
}

const PatternOutputs& HloMatcherPattern::GetOutputs() const { return outputs; }

const Pattern& HloMatcherPattern::GetPatternNodes() const {
  return pattern_nodes;
};

const MatcherGraph& HloMatcherPattern::GetNodesToOperandsMatcherGraph() const {
  return pattern_graphs.first;
};

const MatcherGraph& HloMatcherPattern::GetOperandsToNodesMatcherGraph() const {
  return pattern_graphs.second;
};

std::pair<MatcherGraph, MatcherGraph> HloMatcherPattern::VerifyAndGetGraphs() {
  const std::string prefix = "[Pattern " + type + "] ";

  // A pattern needs to have an output.
  if (outputs.size() == 0) {
    throw std::invalid_argument(
        prefix + "Pattern has no outputs, at least one required.");
  }

  // Make sure inputs are unique and that they point to a label in the pattern.
  absl::flat_hash_set<NodeId> inputs_set;
  for (auto input : inputs) {
    if (input < 0 || input >= static_cast<int64>(pattern_nodes.size())) {
      throw std::invalid_argument(prefix + "Input with label " +
                                  std::to_string(input) +
                                  " does not exist in the pattern.");
    }

    if (inputs_set.count(input)) {
      throw std::invalid_argument(
          prefix + "Input with label " + std::to_string(input) +
          " already defined. Pattern inputs need to be unique.");
    }
    inputs_set.insert(input);
  }

  // Make sure outputs are unique and that they point to a label in the pattern.
  absl::flat_hash_set<NodeId> outputs_set;
  for (auto output : outputs) {
    if (output < 0 || output >= static_cast<int64>(pattern_nodes.size())) {
      throw std::invalid_argument(prefix + "Output with label " +
                                  std::to_string(output) +
                                  " does not exist in the pattern.");
    }

    if (outputs_set.count(output)) {
      throw std::invalid_argument(
          prefix + "Output with label " + std::to_string(output) +
          " already defined. Pattern outputs need to be unique.");
    }
    outputs_set.insert(output);
  }

  // Check that an output is not an input or vice versa.
  absl::flat_hash_set<NodeId> input_output_overlap;
  for (auto input : inputs_set) {
    if (outputs_set.contains(input)) {
      throw std::invalid_argument(
          prefix + "An input is not allowed to be an output (labels " +
          absl::StrJoin(input_output_overlap, ", ") + ").");
    }
  }

  const auto get_operands = [this, &prefix](NodeId label) {
    // Verify that the node with label is defined in the pattern.
    if (label < 0 || label >= static_cast<int64>(pattern_nodes.size())) {
      throw std::invalid_argument(prefix + "Unknown node " +
                                  std::to_string(label) +
                                  " which was not defined in the pattern.");
    }
    return pattern_nodes[label].GetOperands();
  };

  // Create a graph.
  MatcherGraph operands_to_nodes(outputs, get_operands);
  MatcherGraph nodes_to_operands = operands_to_nodes.Transpose();

  // Check that an input doesn't have operands.
  for (auto input : inputs) {
    if (!nodes_to_operands[input].empty()) {
      throw std::invalid_argument(
          prefix + "Input with label " + std::to_string(input) +
          " has an input - this is currently not supported.");
    }
  }

  // Verify that the graph is connected - i.e. any two pairs of nodes in the
  // pattern can be reached.
  // The strategy is to perform a traversal where the next node is either one of
  // the child edges or parent edges which have not yet been visited.
  absl::flat_hash_set<NodeId> visited;
  std::stack<NodeId> to_visit;
  to_visit.push(outputs[0]);

  while (!to_visit.empty()) {
    NodeId current_node = to_visit.top();
    to_visit.pop();
    visited.insert(current_node);

    auto candidates = operands_to_nodes[current_node];
    candidates.insert(nodes_to_operands[current_node].begin(),
                      nodes_to_operands[current_node].end());
    for (auto candidate : candidates) {
      // Only traverse unvisited nodes.
      bool traverse = visited.count(candidate) == 0;
      if (traverse) {
        to_visit.push(candidate);
      }
    }
  }

  for (size_t label = 0; label < pattern_nodes.size(); label++) {
    if (visited.find(label) == visited.end()) {
      throw std::invalid_argument(prefix + "Node with label " +
                                  std::to_string(label) +
                                  " is disconnected from the graph. The "
                                  "graph needs to be connected.");
    }
  }

  return {nodes_to_operands, operands_to_nodes};
}

HloInstruction* HloMatcherMatched::GetMetaTarget() const {
  return instruction_mapping.at(pattern.GetMetaTarget());
}

std::vector<HloInstruction*> HloMatcherMatched::MapInstructions(
    const std::vector<NodeId>& nodes,
    const std::vector<HloInstruction*>& forced_parameters) const {
  std::vector<HloInstruction*> insts;
  insts.reserve(nodes.size() + forced_parameters.size());
  for (NodeId node : nodes) {
    insts.push_back(instruction_mapping.at(node));
  }
  absl::c_copy(forced_parameters, std::back_inserter(insts));
  return insts;
}

std::vector<HloInstruction*> HloMatcherMatched::GetInputs(
    const std::vector<HloInstruction*>& forced_arguments) const {
  return MapInstructions(pattern.GetInputs(), forced_arguments);
}

std::vector<HloInstruction*> HloMatcherMatched::GetOutputs() const {
  return MapInstructions(pattern.GetOutputs());
}

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       struct CompilerAnnotations& annotations,
                       bool root_computation_only,
                       bool requires_unique_sharding,
                       unsigned look_through_max_depth,
                       bool restart_search_after_match)
    : patterns_(std::move(patterns)),
      annotations_(annotations),
      root_computation_only_(root_computation_only),
      requires_unique_sharding_(requires_unique_sharding),
      look_through_max_depth_(look_through_max_depth),
      restart_search_after_match_(restart_search_after_match) {}

// A set of sets of ops which are associative [ (A+B)+C = A+(B+C) ]
static std::set<HloOpcode> associative_opcodes = {
    HloOpcode::kMultiply,
    HloOpcode::kAdd,
};

// Return a set of instructions which, given a root instruction, can be
// rearranged and still retain their algebraic meaning. For instance:
//
// (A+B)+(C+sin(D)) contains three '+ operations which are part of an
// associative set, and can be rearranged into A+(B+C)+sin(D), or
// ((A+B)+C)+sin(D), or any other similar form.

namespace {
struct ToVisitCompare {
  template <typename ValueType>
  bool operator()(const ValueType& a, const ValueType& b) const {
    return std::make_tuple(a.first->unique_id(), a.second) <
           std::make_tuple(b.first->unique_id(), b.second);
  }
};
}  // namespace

std::set<HloInstruction*> HloMatcher::GetAssociativeSet(HloInstruction* root) {
  std::set<std::pair<HloInstruction*, int>, ToVisitCompare> to_visit = {
      {root, 0}};
  std::set<HloInstruction*> result;

  if (associative_opcodes.count(root->opcode()) == 0) {
    return result;
  }

  while (to_visit.size() > 0) {
    auto current = to_visit.begin();
    auto current_inst = current->first;
    auto current_depth = current->second;

    to_visit.erase(current);

    if (current_inst->opcode() == root->opcode() &&
        ShapeUtil::Equal(current_inst->shape(), root->shape())) {
      result.insert(current_inst);
      for (int64 i = 0; i < current_inst->operand_count(); i++) {
        auto* operand = current_inst->mutable_operand(i);
        if (result.count(operand) == 0 && operand->user_count() == 1 &&
            current_depth < static_cast<int64>(look_through_max_depth_)) {
          to_visit.insert({operand, current_depth + 1});
        }
      }
    }
  }
  return result;
}

// This function finds the mext matching operation by skipping over
// associative operations.  It is, effectively, rearranging the graph
// like this, if the pattern is attached to 'B' and (C+B) is a good match
// for the pattern and (A+B) isn't.
//
// B-             B-
//   +---+--   ->   +-----+-
// A-    |        C-      |
//       |                |
// C-----|        A-------|
//
// The re-arrangement is captured in the trace, and the actual
// re-arragement is done in the ReorderGraph function.
absl::optional<Trace> HloMatcher::FindNextMatchingOp(
    HloInstruction* user, HloInstruction* inst, const HloOpcode desiredOpcode,
    const std::set<HloInstruction*>& assoc_set) {
  // Non recursive depth first DAG traversal to try and find an inst with
  // right opcode using associativity
  std::stack<Trace> to_visit;
  // The list of instructions visited while searching for each pattern
  std::set<HloInstruction*> visited = {user};

  // If we ignored an AddDependency op, then `inst` won't be an operand of
  // `user`, so we give up
  if (!user->IsUserOf(inst)) {
    return absl::nullopt;
  }

  // Don't bother looking if there are no associative ops
  if (assoc_set.size() == 0) {
    return absl::nullopt;
  }

  // The starting ops must both be associative
  if (assoc_set.count(user) == 0 || assoc_set.count(inst) == 0) {
    return absl::nullopt;
  }

  // Traverse from inst
  Trace start_trace = {{user, user->operand_index(inst)}};
  to_visit.push(start_trace);
  while (!to_visit.empty()) {
    // Get value off the stack
    auto current = to_visit.top();
    to_visit.pop();

    HloInstruction* current_inst =
        current.back().inst->mutable_operand(current.back().op_idx);
    visited.insert(current_inst);

    for (int64 i = 0; i < current_inst->operand_count(); i++) {
      auto* operand = current_inst->mutable_operand(i);

      auto next_trace = current;
      next_trace.push_back({current_inst, i});

      // Check if this operand matches
      if (operand->opcode() == desiredOpcode) {
        next_trace.push_back({operand, -1});
        return next_trace;
      }

      // Add operands if they are in the associative set
      if (assoc_set.count(operand) > 0) {
        to_visit.push(next_trace);
      }
    }
  }

  return absl::nullopt;
}

StatusOr<bool> HloMatcher::MatchPatternSingleOutput(
    HloInstruction* root, const HloMatcherPattern& pattern,
    HloMatcherMatched& match) {
  match.instruction_mapping[pattern.GetOutputs()[0]] = root;

  std::set<HloInstruction*> associative_set = GetAssociativeSet(root);

  // Construct a mapping from a pattern node to all other pattern nodes which
  // use it
  std::vector<std::set<std::pair<unsigned int, unsigned int>>> node_mapping(
      pattern.GetPatternNodes().size());

  // Create lookup for input indexes to parameter number
  std::map<NodeId, int64> input_id_to_param_num;
  for (size_t i = 0; i < pattern.GetInputs().size(); i++) {
    input_id_to_param_num[pattern.GetInputs()[i]] = i;
  }

  const auto is_input = [&input_id_to_param_num](const NodeId pid) {
    return input_id_to_param_num.count(pid);
  };

  for (unsigned int node_num = 0; node_num < pattern.GetPatternNodes().size();
       node_num++) {
    for (unsigned int op_idx = 0;
         op_idx < pattern.GetPatternNodes()[node_num].GetOperands().size();
         op_idx++) {
      node_mapping[pattern.GetPatternNodes()[node_num].GetOperands()[op_idx]]
          .insert({node_num, op_idx});
    }

    if (node_num) {
      match.instruction_mapping[node_num] = nullptr;
    }
  }

  for (unsigned int node_num = 0; node_num < pattern.GetPatternNodes().size();
       node_num++) {
    HloInstruction* inst = match.instruction_mapping[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern.GetPatternNodes()[node_num]);
    if (node.GetOpcodeTarget().IsHloOpcode()) {
      HloOpcode target_opcode = node.GetOpcodeTarget().GetHloOpcode();
      if (target_opcode != HloOpcode::kParameter) {
        if (target_opcode != inst->opcode()) {
          // Try to find an op using associativity, unless this is the first
          // node or search depth is 0 or this inst is used more than once
          if (node_num != 1 || look_through_max_depth_ == 0 ||
              inst->user_count() != 1) {
            return false;
          }
          unsigned int user_node_num = node_mapping[node_num].begin()->first;
          auto* user = match.instruction_mapping[user_node_num];
          auto optional_trace =
              FindNextMatchingOp(user, inst, target_opcode, associative_set);
          // Check whether we managed to find a match
          if (!optional_trace) {
            return false;
          }
          Trace found = *optional_trace;

          match.instruction_mapping[node_num] = found.back().inst;
          inst = found.back().inst;
          match.replacement_traces.push_back(found);
        }
      }
    }
    // Check the match.
    if (!node.Matches(inst)) {
      return false;
    }

    if (!is_input(node_num)) {
      if ((node.GetOperands().size() > 0) &&
          (static_cast<size_t>(inst->operand_count()) !=
           node.GetOperands().size())) {
        return false;
      }

      for (unsigned int i = 0; i < node.GetOperands().size(); i++) {
        HloInstruction* operand = inst->mutable_operand(i);

        // Look through AddDepedency nodes
        if (operand->opcode() == HloOpcode::kAddDependency) {
          match.dependency_predecessors.push_back(operand);
          operand = operand->mutable_operand(0);
        }

        size_t n = node.GetOperands()[i];

        if (n >= match.instruction_mapping.size()) {
          return InvalidArgument("Invalid matcher reference ", n);
        }

        if (match.instruction_mapping[n] != nullptr) {
          // Instructions can only match once
          if (match.instruction_mapping[n] != operand) {
            return false;
          }
        } else {
          // Each instruction can match only one entry in the pattern
          auto it =
              absl::c_find_if(match.instruction_mapping,
                              [&](std::pair<NodeId, HloInstruction*> iso_pair) {
                                return iso_pair.second == operand;
                              });
          if (it != match.instruction_mapping.end()) {
            return false;
          }

          match.instruction_mapping[n] = operand;
        }
      }
    }
  }
  return true;
}

StatusOr<bool> HloMatcher::MatchPattern(HloInstruction* root,
                                        const unsigned pattern_idx) {
  const auto& pattern = patterns_[pattern_idx];
  HloMatcherMatched match(root->parent(), pattern_idx, pattern);

  bool matched = false;
  if (pattern.GetOutputs().size() == 1) {
    // TODO - T5965
    // We still use the old algorithm for the matching of patterns with a single
    // output because that algorithm supports associative look through matching.
    // Remove this algorithm once the new algorithm supports it.
    TF_ASSIGN_OR_RETURN(matched,
                        MatchPatternSingleOutput(root, pattern, match));
  } else {
    matched = MatchDAGIsomorphism(match.instruction_mapping, root, pattern);
  }

  if (matched) {
    // Optional unique device this fusion will be performed on.
    absl::optional<int64> sharding_device = absl::nullopt;
    if (requires_unique_sharding_) {
      // Check that all the instructions have compatible sharding - i.e. all
      // non-input instructions in the pattern are using the same unique device.
      absl::flat_hash_set<int64> sharding_devices;

      for (auto pair : match.instruction_mapping) {
        NodeId id = pair.first;
        HloInstruction* inst = pair.second;
        const bool is_input =
            absl::c_find(pattern.GetInputs(), id) != pattern.GetInputs().end();
        if (!is_input && inst->has_sharding()) {
          auto sharding = inst->sharding();
          // We ignore sharding if any of the following are true:
          // * it's not supported sharding information.
          // * it's a (wide) constant.
          bool ignore_sharding = !IsSupportedSharding(sharding);
          ignore_sharding |=
              (inst->opcode() == HloOpcode::kConstant) ||
              (inst->opcode() == HloOpcode::kBroadcast &&
               inst->operand(0)->opcode() == HloOpcode::kConstant) ||
              IsWideConstant(inst);
          if (!ignore_sharding) {
            sharding_devices.insert(*sharding.UniqueDevice());
          }
        }
      }

      if (sharding_devices.size() == 1) {
        // If we have one sharding device then we use that sharding information
        // for the whole fusion.
        sharding_device = *std::begin(sharding_devices);
      } else if (sharding_devices.size() > 1) {
        // Multiple devices.
        return false;
      }
    }
    return HandleMatch(match, sharding_device);
  } else {
    return false;
  }
}

StatusOr<bool> HloMatcher::MatchPatternStart(HloComputation* computation) {
  bool matched = false;

  // Find any matches for the set patterns, note that we conditionally
  // restart the search after every match.
  bool start_from_root = true;
  while (start_from_root) {
    start_from_root = false;

    for (unsigned i = 0; i < patterns_.size(); i++) {
      TF_ASSIGN_OR_RETURN(bool found_match, FindMatch(computation, i));
      if (found_match) {
        matched = true;

        if (restart_search_after_match_) {
          start_from_root = true;
          break;
        }
      }
    }
  }

  return matched;
}

StatusOr<bool> HloMatcher::FindMatch(HloComputation* computation,
                                     const unsigned pattern_idx) {
  // Non recursive depth first DAG traversal to match the specified pattern.
  bool found_match = false;
  const auto& pattern = patterns_[pattern_idx];

  std::vector<HloInstruction*> to_visit;
  // The list of instructions visited while searching for each pattern
  std::set<HloInstruction*> visited;

  // Traverse from root
  to_visit = FindUnreachableRoots(computation);
  // Visit the root instruction first
  to_visit.insert(to_visit.begin(), computation->root_instruction());
  while (!to_visit.empty()) {
    HloInstruction* inst = to_visit.back();
    to_visit.pop_back();
    const auto insert_result = visited.insert(inst);
    const bool duplicate = insert_result.second == false;
    if (duplicate) {
      continue;
    }
    // A pattern can have multiple outputs. We start the pattern match when
    // we find an instruction which matches the first output of the pattern.
    auto output_0_node = pattern.GetPatternNodes()[pattern.GetOutputs()[0]];
    if (output_0_node.Matches(inst)) {
      // When a pattern is fully matched, a replacement (defined by the
      // subclass) is also performed. If pattern_matches is true when we return
      // from MatchPattern then inst is no longer valid, since it's been
      // replaced. To find the new instructions we keep track of the users of
      // the original, as any new instruction will be an operand of those.
      // FIXME: Make replaced instructions an explicit part of the HandleMatch
      // interface, that way we can iterate over the new instructions directly.
      auto pattern_users = inst->users();

      // Try matching the whole pattern
      TF_ASSIGN_OR_RETURN(bool pattern_matches,
                          MatchPattern(inst, pattern_idx));
      if (pattern_matches) {
        VLOG(1) << "Matched pattern type " << pattern.GetType() << ".";
        found_match = true;
        if (restart_search_after_match_) {
          break;
        } else {
          // New instructions will be operands of the original users, so we
          // visit the operands to visit the new instructions. Previously
          // visited operands will be skipped.
          for (HloInstruction* user : pattern_users) {
            // Try and preserve the DFS order by only checking users that have
            // already been visited.
            if (visited.count(user) > 0) {
              const HloInstruction::InstructionVector& operands =
                  user->operands();
              to_visit.insert(to_visit.end(), operands.begin(), operands.end());
            }
          }
          // We have to finish here as inst is no longer valid.
          continue;
        }
      }
    }
    for (HloInstruction* operand : inst->operands()) {
      to_visit.push_back(operand);
    }
  }

  return found_match;
}

StatusOr<bool> HloMatcher::Run(HloModule* module) {
  bool matched = false;

  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    TF_ASSIGN_OR_RETURN(matched, MatchPatternStart(comp));
  } else {
    // Copy list of computations as we will be introducing new ones
    std::vector<HloComputation*> comps = module->MakeComputationPostOrder();
    for (auto* comp : comps) {
      if (!comp->IsFusionComputation() && !IsPopOpsFusion(comp)) {
        TF_ASSIGN_OR_RETURN(bool pattern_matched, MatchPatternStart(comp));
        matched |= pattern_matched;
      }
    }
  }

  return matched;
}

void HloMatcher::ReorderGraph(const HloMatcherMatched& matched) {
  // This reordering relies on associativity:
  // For instance, if we have "[root] add1 add2 … addN [target]",
  // we relink root to have [target] as operand and make first
  // parent of [root] a new root:
  // "add1 add2 … addN [root] [target]".

  for (auto trace : matched.replacement_traces) {
    auto root = trace[0];
    auto root_parent = trace[1];
    auto target_user = trace[trace.size() - 2];
    auto target = trace[trace.size() - 1];

    TF_CHECK_OK(root.inst->ReplaceAllUsesWith(root_parent.inst));
    TF_CHECK_OK(
        target_user.inst->ReplaceOperandWith(target_user.op_idx, root.inst));
    TF_CHECK_OK(root.inst->ReplaceOperandWith(root.op_idx, target.inst));
  }
}

Status HloMatcher::RemoveUnusedInstructions(const HloMatcherMatched& matched) {
  HloComputation* computation = matched.computation;
  const auto& pattern = patterns_[matched.pattern_idx];
  // Remove all the dead instructions in the graph after outlining.
  // DF Traversal from every output node - note that we can't call
  // RemoveInstructionAndUnusedOperands as it doesn't allow us to remove state
  // full ops.
  for (NodeId output_node_id : pattern.GetOutputs()) {
    std::queue<NodeId> to_visit;
    to_visit.push(output_node_id);
    absl::flat_hash_set<NodeId> visited;

    while (!to_visit.empty()) {
      NodeId node_id = to_visit.front();
      to_visit.pop();
      HloInstruction* inst = matched.instruction_mapping.at(node_id);

      // Don't remove nodes already visited or the instructions with users.
      if (visited.count(node_id) != 0 || inst->user_count() != 0) {
        continue;
      }

      for (auto operand_id :
           pattern.GetNodesToOperandsMatcherGraph()[node_id]) {
        to_visit.push(operand_id);
      }

      visited.insert(node_id);

      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
    }
  }
  return Status::OK();
}

StatusOr<HloInstruction*> HloMatcher::OutlineExpressionFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name,
    const absl::optional<int64> sharding_device,
    std::vector<HloInstruction*>&& forced_parameters) {
  const auto& pattern = patterns_[matched.pattern_idx];
  return !pattern.GetReplaceFn()
             ? OutlineFusionFromComputation(matched, outlined_computation_name,
                                            sharding_device,
                                            std::move(forced_parameters))
             : OutlineCustomOpFromComputation(
                   matched, outlined_computation_name, sharding_device,
                   std::move(forced_parameters));
}

StatusOr<HloInstruction*> HloMatcher::OutlineFusionFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name,
    const absl::optional<int64> sharding_device,
    std::vector<HloInstruction*>&& forced_parameters) {
  HloComputation* computation = matched.computation;
  const auto& pattern = patterns_[matched.pattern_idx];
  HloModule* module = computation->parent();

  // Unlink the AddDependency instructions from the matched pattern
  for (auto* dep : matched.dependency_predecessors) {
    dep->ReplaceAllUsesWith(dep->mutable_operand(0));
  }

  // We need to update the graph with any instructions that will be reordered.
  ReorderGraph(matched);
  // A map from original instructions to their new counterparts
  absl::flat_hash_map<NodeId, HloInstruction*> outlined;
  // A set of nodes which we have already outlined.
  absl::flat_hash_set<NodeId> outlined_node_ids;
  // A set of nodes which we can outline because all the operands have been
  // outlined.
  std::set<NodeId> to_outline;
  // Arguments to the new computation.
  std::vector<HloInstruction*> arguments;
  // A node can be outlined if all the operands have been outlined and it has
  // not been outlined yet.
  const auto can_outline = [&](NodeId node_id) {
    for (auto operand_id : pattern.GetNodesToOperandsMatcherGraph()[node_id]) {
      if (outlined_node_ids.count(operand_id) == 0) {
        return false;
      }
    }
    return outlined_node_ids.count(node_id) == 0;
  };

  // First outline all the parameters.
  auto builder = HloComputation::Builder(outlined_computation_name);
  for (unsigned parameter_num = 0; parameter_num < pattern.GetInputs().size();
       parameter_num++) {
    NodeId node_id = pattern.GetInputs()[parameter_num];
    HloInstruction* param_input = matched.instruction_mapping.at(node_id);
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(parameter_num, param_input->shape(),
                                        StrCat("arg_", parameter_num)));
    outlined[node_id] = param;
    outlined_node_ids.insert(node_id);
    arguments.push_back(param_input);
    // Check what we can outline.
    absl::c_copy_if(pattern.GetOperandsToNodesMatcherGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }
  // Add all the instructions which have no dependencies to be outlined as well
  // (for example constants).
  for (auto pair : pattern.GetNodesToOperandsMatcherGraph()) {
    NodeId node_id = pair.first;
    auto edges = pair.second;
    if (edges.empty() && outlined_node_ids.count(node_id) == 0) {
      to_outline.insert(node_id);
    }
  }

  // For keeping track of any AfterAll instructions found in the outline
  std::vector<HloInstruction*> after_all;

  // Now outline all the remaining nodes
  while (!to_outline.empty()) {
    // Get an instruction which is ready to be outlined.
    NodeId node_id = *to_outline.begin();
    to_outline.erase(node_id);

    HloInstruction* old_inst = matched.instruction_mapping.at(node_id);
    HloInstruction* new_inst = builder.AddInstruction(old_inst->Clone());

    for (auto* u : old_inst->users()) {
      if (u->opcode() == HloOpcode::kAfterAll) {
        after_all.push_back(u);
      }
    }

    outlined[node_id] = new_inst;
    outlined_node_ids.insert(node_id);
    // Replace all the operands
    for (int64 operand = 0; operand < new_inst->operand_count(); ++operand) {
      auto& operands = pattern.GetPatternNodes()[node_id].GetOperands();
      if (operand >= operands.size()) {
        return InvalidArgument(
            "Operand index out of range ", operand, " / ", operands.size(),
            ". Pattern and instruction operands don't not match. "
            "One possible scenario could be "
            "HloMatcherOpcode::kAnyOpcode not specified as input.");
      }
      NodeId operand_id = operands[operand];
      TF_RETURN_IF_ERROR(
          new_inst->ReplaceOperandWith(operand, outlined[operand_id]));
    }
    // Check if we can outline more instructions.
    absl::c_copy_if(pattern.GetOperandsToNodesMatcherGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }

  // Sanity check - make sure we have outlined everything.
  if (outlined.size() != pattern.GetPatternNodes().size()) {
    return InternalError(
        "Failed to outline a pattern correctly - not all "
        "instructions have been outlined. ",
        outlined.size(), " ", pattern.GetPatternNodes().size());
  }
  // If we have multiple outputs then create a root tuple - otherwise output[0]
  // is the root.
  HloInstruction* root;
  if (pattern.GetOutputs().size() > 1) {
    std::vector<HloInstruction*> outputs;
    absl::c_transform(pattern.GetOutputs(), std::back_inserter(outputs),
                      [&](NodeId node_id) { return outlined[node_id]; });
    root = builder.AddInstruction(HloInstruction::CreateTuple(outputs));
  } else {
    root = outlined[pattern.GetOutputs()[0]];
  }

  // Add forced parameters as arguments - DCE does not remove unused parameters.
  // This allows us to link and easily maintain outputs of a fwd pass to the bwd
  // pass.
  for (unsigned i = 0; i < forced_parameters.size(); i++) {
    HloInstruction* inst = forced_parameters[i];
    const unsigned parameter_num = arguments.size() + i;
    builder.AddInstruction(HloInstruction::CreateParameter(
        parameter_num, inst->shape(), StrCat("arg_", parameter_num)));
    arguments.push_back(inst);
  }

  // Creates a fusion call to the nested computation.
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(builder.Build(root));

  // Ensure that all parameters are a dependency of the root
  for (auto* param : fusion_computation->parameter_instructions()) {
    if (param->user_count() == 0) {
      param->AddControlDependencyTo(root);
    }
  }

  HloInstruction* fusion =
      matched.computation->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom, arguments,
          fusion_computation));

  fusion_computation->SetFusionInstruction(fusion);

  auto* old = matched.instruction_mapping.at(pattern.GetMetaTarget());

  PoplarBackendConfig backend_config;
  auto* cfg = backend_config.mutable_fusion_config();
  if (old->opcode() == HloOpcode::kConvolution) {
    *(cfg->mutable_window()) = old->window();
    *(cfg->mutable_dimension_numbers()) = old->convolution_dimension_numbers();
    cfg->set_feature_group_count(old->feature_group_count());
    cfg->set_batch_group_count(old->batch_group_count());
  }

  if (pattern.GetInplaceDescriptionFn()) {
    auto inplace_descriptions = pattern.GetInplaceDescriptionFn()(matched);
    for (const auto& inplace_description : inplace_descriptions) {
      auto* proto = cfg->add_inplace_descriptions();
      *proto = inplace_description.ToProto();
    }
  }

  TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

  fusion->set_metadata(old->metadata());
  if (sharding_device) {
    fusion->set_sharding(HloSharding::AssignDevice(*sharding_device));
  }
  fusion->set_frontend_attributes(old->frontend_attributes());

  // Replace the uses with the new outputs.
  if (pattern.GetOutputs().size() > 1) {
    // For multiple outputs use GTEs.
    for (unsigned tuple_id = 0; tuple_id < pattern.GetOutputs().size();
         tuple_id++) {
      NodeId node_id = pattern.GetOutputs()[tuple_id];
      HloInstruction* old_inst = matched.instruction_mapping.at(node_id);
      HloInstruction* gte =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              old_inst->shape(), fusion, tuple_id));
      TF_RETURN_IF_ERROR(old_inst->ReplaceAllUsesWith(gte));
    }
  } else {
    HloInstruction* old_inst =
        matched.instruction_mapping.at(pattern.GetOutputs()[0]);
    TF_RETURN_IF_ERROR(old_inst->ReplaceAllUsesWith(fusion));
  }

  // Create new dependencies in place of the old ones
  for (auto* dep : matched.dependency_predecessors) {
    auto new_dep =
        computation->AddInstruction(HloInstruction::CreateAddDependency(
            fusion->mutable_operand(0), dep->mutable_operand(1)));
    TF_RETURN_IF_ERROR(fusion->ReplaceOperandWith(0, new_dep));
  }

  HloInstructionSet deps_set;
  for (auto dep : matched.dependency_predecessors) {
    deps_set.insert(dep);
  }

  for (auto dep : deps_set) {
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(dep));
  }

  // Move the AfterAll instructions to the fusion output
  for (auto* u : after_all) {
    TF_RETURN_IF_ERROR(u->ReplaceOperandWithDifferentShape(0, fusion));
  }

  TF_RETURN_IF_ERROR(RemoveUnusedInstructions(matched));

  return fusion;
}

StatusOr<HloInstruction*> HloMatcher::OutlineCustomOpFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name,
    const absl::optional<int64> sharding_device,
    std::vector<HloInstruction*>&& forced_parameters) {
  HloComputation* computation = matched.computation;
  const auto& pattern = patterns_[matched.pattern_idx];
  HloModule* module = computation->parent();

  auto& replace_fn = pattern.GetReplaceFn();
  if (!replace_fn) {
    return InternalError("Replace function was not specified in pattern.");
  }

  if (pattern.GetInplaceDescriptionFn()) {
    return InvalidArgument(
        "Pattern with replacement function can't have inplace description "
        "specified.");
  }

  if (!forced_parameters.empty()) {
    return InvalidArgument(
        "Pattern with replacement function can't have forced parameters "
        "specified.");
  }

  for (auto* dep : matched.dependency_predecessors) {
    dep->ReplaceAllUsesWith(dep->mutable_operand(0));
  }

  // We need to update the graph with any instructions that will be reordered.
  ReorderGraph(matched);

  auto* old_meta_target = matched.GetMetaTarget();

  HloInstructionSet after_all;
  for (auto pair : matched.instruction_mapping) {
    for (auto user : pair.second->users()) {
      if (user->opcode() == HloOpcode::kAfterAll) {
        after_all.insert(user);
      }
    }
  }

  std::vector<HloInstruction*> inputs = matched.GetInputs();

  TF_ASSIGN_OR_RETURN(PatternInstructionOutputs outputs, replace_fn(matched));
  for (HloInstruction* inst : outputs) {
    if (inst->parent() != computation) {
      return InternalError(
          "Output returned from replacement function was not added to "
          "matched.computation.");
    }
  }

  auto pattern_outputs = matched.GetOutputs();
  if (outputs.size() != pattern_outputs.size()) {
    return InternalError(
        "Replacement function returned wrong number of outputs.");
  }

  HloInstruction* new_meta_target = nullptr;
  auto metadata = old_meta_target->metadata();
  auto frontend_attrs = old_meta_target->frontend_attributes();
  for (std::size_t i = 0; i < pattern_outputs.size(); ++i) {
    HloInstruction* pattern_output = pattern_outputs[i];
    HloInstruction* output = outputs[i];
    if (pattern_output == old_meta_target) {
      new_meta_target = output;
    }
    TF_RETURN_IF_ERROR(pattern_output->ReplaceAllUsesWith(output));
    output->set_metadata(metadata);
    if (sharding_device) {
      output->set_sharding(HloSharding::AssignDevice(*sharding_device));
    }
    output->set_frontend_attributes(frontend_attrs);
  }

  if (!new_meta_target) {
    return InternalError("CustomOp replacement couldn't find new meta target.");
  }

  for (HloInstruction* inst : after_all) {
    TF_RETURN_IF_ERROR(
        inst->ReplaceOperandWithDifferentShape(0, new_meta_target));
  }

  TF_RETURN_IF_ERROR(RemoveUnusedInstructions(matched));

  return new_meta_target;
}

}  // namespace poplarplugin
}  // namespace xla
