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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_MATCHER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_MATCHER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

using NodeId = int64;
using NodeOperands = std::vector<NodeId>;
using NodeCondition = std::function<bool(const HloInstruction*)>;
using MatcherGraph = MetaGraph<NodeId>;

enum class HloMatcherOpcode {
  kAnyOpcode,
};

// A class which allows us to extend the HloOpcode enum for special cases for
// the HloMatcher.
class HloMatcherOpcodeTarget {
 public:
  HloMatcherOpcodeTarget(const HloOpcode& opcode);
  HloMatcherOpcodeTarget(const HloMatcherOpcode& opcode);

  const bool IsHloOpcode() const;
  const HloOpcode GetHloOpcode() const;

  const bool IsHloMatcherOpcode() const;
  const HloMatcherOpcode GetHloMatcherOpcode() const;

 private:
  absl::variant<HloOpcode, HloMatcherOpcode> opcode_;
};

class HloMatcherNode {
 public:
  HloMatcherNode(HloMatcherOpcodeTarget opcode_target, NodeOperands operands);

  HloMatcherNode(HloMatcherOpcodeTarget opcode_target, NodeOperands operands,
                 NodeCondition node_condition);

  HloMatcherNode(HloMatcherOpcodeTarget opcode_target, NodeOperands operands,
                 const std::vector<NodeCondition>& node_conditions);

  const HloMatcherOpcodeTarget& GetOpcodeTarget() const;
  const NodeOperands& GetOperands() const;
  const std::vector<NodeCondition>& GetNodeConditions() const;

  // Checks whether the instruction matches this node.
  const bool Matches(const HloInstruction* inst) const;

 private:
  // The opcode target of the instruction to match
  HloMatcherOpcodeTarget opcode_target_;

  // A list of operands of this instruction. A positive number refers to one of
  // the other entries in the match pattern. A negative number indicates that
  // this operand will be a parameter to the fused subgraph.  If multiple match
  // nodes have the same negative number, then the same instruction must be
  // the operand to each match node. The parameter number is given by the value
  // in the matching position in the parameter_indices list.
  NodeOperands operands_;

  // These functions will be called with the instruction. Only if all the
  // functions return true does the matching proceed.
  const std::vector<NodeCondition> node_conditions_;
};

using PatternType = std::string;
using PatternMetaTarget = NodeId;
using PatternInputs = std::vector<NodeId>;
using PatternOutputs = std::vector<NodeId>;
using Pattern = std::vector<HloMatcherNode>;

class HloMatcherPattern;
struct HloMatcherMatched;
using PatternInstructionOutputs = std::vector<HloInstruction*>;
using PatternInplaceDescriptionFn =
    std::function<HloPoplarUseDescriptions(const HloMatcherMatched&)>;
using PatternReplaceFn = std::function<StatusOr<PatternInstructionOutputs>(
    const HloMatcherMatched&)>;

class HloMatcherPattern {
 public:
  HloMatcherPattern() = delete;

  HloMatcherPattern(PatternType type, PatternMetaTarget meta_target,
                    PatternInputs inputs, PatternOutputs outputs,
                    Pattern pattern);

  HloMatcherPattern(PatternType type, PatternMetaTarget meta_target,
                    PatternInputs inputs, PatternOutputs outputs,
                    PatternInplaceDescriptionFn inplace_description_fn,
                    Pattern pattern);

  HloMatcherPattern(PatternType type, PatternReplaceFn replace_fn,
                    PatternMetaTarget meta_target, PatternInputs inputs,
                    PatternOutputs outputs, Pattern pattern);

  HloMatcherPattern(PatternType type, PatternReplaceFn replace_fn,
                    PatternMetaTarget meta_target, PatternInputs inputs,
                    PatternInplaceDescriptionFn inplace_description_fn,
                    PatternOutputs outputs, Pattern pattern);

  const PatternType& GetType() const;

  const PatternReplaceFn& GetReplaceFn() const;

  const PatternMetaTarget& GetMetaTarget() const;

  const PatternInputs& GetInputs() const;

  const PatternInplaceDescriptionFn& GetInplaceDescriptionFn() const;

  const PatternOutputs& GetOutputs() const;

  const Pattern& GetPatternNodes() const;

  const MatcherGraph& GetNodesToOperandsMatcherGraph() const;

  const MatcherGraph& GetOperandsToNodesMatcherGraph() const;

 private:
  // The name to give the extracted fused graph.
  PatternType type;

  // Replace function. If it specified, instead of outlining as a fusion,
  // matcher will call this function and insert instruction instead.
  PatternReplaceFn replace_fn;

  // The index of the op within the fusion which should have its op_metadata
  // copied to the kFusion instruction.
  PatternMetaTarget meta_target;

  // If op is an input then don't include this instruction in the fusion. The
  // fused subgraph will have a parameter where this instruction would be, and
  // the index of that parameter is given by the relative index in the inputs
  // vector.
  // Example:
  // inputs = {2, 1}
  // Then the instruction with label 2 will be a parameter instruction with
  // index 0 and the instruction with label 1 will be a parameter instruction
  // with index 1.
  PatternInputs inputs;

  // Function used to retrieve information how the fusion inputs alias any of
  // the fusion outputs.
  PatternInplaceDescriptionFn inplace_description_fn;

  // If an op is an output then replace all the uses of this node in the
  // computation with the output tensor from this fusion. If there is more than
  // one output then the fusion returns a tuple and the output tensor tuple
  // index is determined by the relative index in the outputs.
  // Example:
  // outputs = {2, 0}
  // will insert two GTE instructions into the graph, where GTE with tuple_index
  // == 0 will correspond to output tensor with label 2 and GTE with tuple_index
  // == 1 will correspond to output tensor with label 0.
  PatternOutputs outputs;

  // A vector of HloMatcherNode, describing the pattern to match.
  Pattern pattern_nodes;

  // Structures used to represent this pattern - the first graph represents the
  // connections between nodes and their operands, the second graph represents
  // the connections between operands and their usage nodes.
  std::pair<MatcherGraph, MatcherGraph> pattern_graphs;

  // This function verifies that the pattern is correct. We define a pattern
  // correct if the following conditions are all met:
  // * It has at least one output.
  // * The graph is connected.
  std::pair<MatcherGraph, MatcherGraph> VerifyAndGetGraphs();
};

struct InstructionIndex {
  HloInstruction* inst;
  int64 op_idx;
};

using Trace = std::vector<InstructionIndex>;

struct HloMatcherMatched {
  HloComputation* computation;
  unsigned pattern_idx;
  const HloMatcherPattern& pattern;
  absl::flat_hash_map<NodeId, HloInstruction*> instruction_mapping;
  std::vector<Trace> replacement_traces;
  std::vector<HloInstruction*> dependency_predecessors;

  HloMatcherMatched(HloComputation* computation, const unsigned pattern_idx,
                    const HloMatcherPattern& pattern)
      : computation(computation), pattern_idx(pattern_idx), pattern(pattern) {}

  HloInstruction* GetMetaTarget() const;
  std::vector<HloInstruction*> GetInputs(
      const std::vector<HloInstruction*>& forced_parameters = {}) const;
  std::vector<HloInstruction*> GetOutputs() const;
  std::vector<HloInstruction*> MapInstructions(
      const std::vector<NodeId>& nodes,
      const std::vector<HloInstruction*>& forced_parameters = {}) const;
};

using ReplacedInstructions = std::vector<HloInstruction*>;

class HloMatcher : public HloModulePass {
 public:
  // By default never look through associative ops
  HloMatcher(const std::vector<HloMatcherPattern>& patterns,
             struct CompilerAnnotations& annotations, bool root_only,
             bool requires_unique_sharding = false,
             unsigned look_through_max_level = 0,
             bool restart_search_after_match = true);

  ~HloMatcher() override = default;

  absl::string_view name() const override { return "hlo-matcher"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // The list of patterns to try to find in the computations
  std::vector<HloMatcherPattern> patterns_;

 protected:
  // Outlines the given match and return the instruction which calls the
  // outlined computation.
  StatusOr<HloInstruction*> OutlineExpressionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name,
      const absl::optional<int64> sharding_device,
      std::vector<HloInstruction*>&& forced_parameters = {});

  // The instruction annotations from the compiler
  struct CompilerAnnotations& annotations_;

 private:
  virtual StatusOr<bool> HandleMatch(
      HloMatcherMatched& match,
      const absl::optional<int64> sharding_device) = 0;

  Status RemoveUnusedInstructions(const HloMatcherMatched& matched);

  StatusOr<HloInstruction*> OutlineFusionFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name,
      const absl::optional<int64> sharding_device,
      std::vector<HloInstruction*>&& forced_parameters);

  StatusOr<HloInstruction*> OutlineCustomOpFromComputation(
      const HloMatcherMatched& matched,
      const std::string& outlined_computation_name,
      const absl::optional<int64> sharding_device,
      std::vector<HloInstruction*>&& forced_parameters);

  StatusOr<bool> MatchPatternStart(HloComputation*);
  StatusOr<bool> FindMatch(HloComputation*, const unsigned pattern_idx);

  StatusOr<bool> MatchPattern(HloInstruction* inst, const unsigned pattern_idx);

  std::set<HloInstruction*> GetAssociativeSet(HloInstruction*);

  absl::optional<Trace> FindNextMatchingOp(HloInstruction* user,
                                           HloInstruction* inst,
                                           const HloOpcode desiredOpcode,
                                           const std::set<HloInstruction*>&);

  StatusOr<bool> MatchPatternSingleOutput(HloInstruction* root,
                                          const HloMatcherPattern& pattern,
                                          HloMatcherMatched& match);

  void ReorderGraph(const HloMatcherMatched& matched);

  bool root_computation_only_;
  bool requires_unique_sharding_;
  unsigned look_through_max_depth_;
  const bool restart_search_after_match_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
