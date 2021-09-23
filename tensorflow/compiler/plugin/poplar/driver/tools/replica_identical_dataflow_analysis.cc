/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/replica_identical_dataflow_analysis.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsResultIdentical(const ValueCategoryTree& tree) {
  return tree.element(RootShapeIndex()) == ValueReplicaCategory::Identical;
}

// Check whether shape is a wrapper around wrapped_shape.
bool IsWrapperTuple(const Shape& shape, const Shape& wrapped_shape) {
  if (shape.IsTuple() && ShapeUtil::TupleElementCount(shape) == 1) {
    return ShapeUtil::GetTupleElementShape(shape, /*index*/ 0) == wrapped_shape;
  }

  return false;
}

bool IsVisitable(CallGraph& call_graph, HloComputation* computation) {
  const auto called = !call_graph.GetComputationCallers(computation).empty();
  if (called) {
    const auto& node = call_graph.GetNode(computation);
    return node.context() != CallContext::kParallel;
  }

  return false;
}

// Create a new ValueCategoryTree whose nodes have
// ValueReplicaCategory::Identity if the corresponding nodes in both lhs and rhs
// are also identical.
ValueCategoryTree MakeValuesIdenticalIfTreeElementsAre(
    const ValueCategoryTree& lhs, const ValueCategoryTree& rhs) {
  // We can ignore any layout differences since shape layouts are not used
  // when lowering to Poplar.
  CHECK(Shape::Equal().IgnoreLayout()(lhs.shape(), rhs.shape()));

  ValueCategoryTree merged_categories(lhs.shape(),
                                      ValueReplicaCategory::Differing);
  merged_categories.ForEachMutableElement(
      [&](const ShapeIndex& index, ValueReplicaCategory* category) {
        // Both ValueCategoryTrees have the same shape, so we can merge them
        // by just comparing elements.
        const auto lhs_category = lhs.element(index);
        if (lhs_category == rhs.element(index)) {
          *category = lhs_category;
        }
      });

  return merged_categories;
}

// Create a map of overrides for the parameters of the given HloComputation
// using the given categories. This can be used to find the value categories of
// comp as if it were called when parameters that have the value categories
// given in parameter_categories
absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
CreateParameterOverridesFromCategories(
    const HloComputation* comp, const ValueCategoryTree& parameter_categories) {
  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      parameter_overrides;

  if (comp->num_parameters() == 1) {
    // For a single parameter just copy the whole tree.
    const HloInstruction* parameter = comp->parameter_instruction(0);

    auto value_category = ValueCategoryTree(parameter->shape());
    value_category.CopySubtreeFrom(parameter_categories, RootShapeIndex(),
                                   RootShapeIndex());

    parameter_overrides[parameter] = std::move(value_category);
  } else {
    const std::vector<HloInstruction*>& params = comp->parameter_instructions();
    CHECK_EQ(params.size(), parameter_categories.shape().tuple_shapes_size());

    // For multiple parameters copy the subtree for each parameter.
    for (auto i = 0u; i < params.size(); ++i) {
      auto* param = params[i];

      auto value_categories = ValueCategoryTree(param->shape());
      value_categories.CopySubtreeFrom(parameter_categories, {i},
                                       RootShapeIndex());

      parameter_overrides[param] = std::move(value_categories);
    }
  }

  return parameter_overrides;
}
}  // namespace

std::ostream& operator<<(std::ostream& stream,
                         const ValueReplicaCategory& category) {
  stream << "ValueReplicaCategory::";
  switch (category) {
    case ValueReplicaCategory::Identical:
      stream << "Identical";
      break;
    case ValueReplicaCategory::Differing:
      stream << "Differing";
      break;
    case ValueReplicaCategory::Unknown:
      stream << "Unknown";
      break;
    default:
      CHECK(false) << "Got unexpected ValueReplicaCategory";
      break;
  }
  return stream;
}

ValuesIdenticalAcrossReplicasVisitor::ValuesIdenticalAcrossReplicasVisitor(
    const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
        category_overrides)
    : value_category_mapping_(category_overrides) {
  // We don't want to visit instructions with an overide since that will negate
  // the effect of having an override
  MarkOverridesAsVisited(category_overrides);
}

const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
ValuesIdenticalAcrossReplicasVisitor::ValueCategoryMapping() const {
  return value_category_mapping_;
}

bool ValuesIdenticalAcrossReplicasVisitor::Visited(
    const HloComputation* comp) const {
  return value_category_mapping_.contains(comp->root_instruction());
}

Status ValuesIdenticalAcrossReplicasVisitor::Postprocess(
    const HloInstruction* inst) {
  const auto category = value_category_mapping_[inst].element(RootShapeIndex());
  VLOG(3) << "Instruction '" << inst->name() << "' has value category '"
          << category << "'.";

  return Status::OK();
}

Status ValuesIdenticalAcrossReplicasVisitor::DefaultAction(
    const HloInstruction* inst) {
  // If an instruction has side effects then calling it multiple times
  // with the same operands might produce different results.
  const bool is_identical =
      !inst->HasSideEffect() && AllOperandsIdentical(inst);
  return SetAllInstructionValuesToIdenticalOrDiffering(inst, is_identical);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleCall(
    const HloInstruction* call) {
  HloComputation* comp = call->to_apply();

  if (IsFunction(call) || IsAnyPipelineStageOpOrResourceUpdate(call)) {
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(comp, call));
    return Status::OK();
  } else if (IsPipelineOp(call)) {
    // Handle the pipeline like a repeat loop since it'll be
    // invoked several times.
    // A pipeline is only replica identical if its body and
    // its `gradient_accumulation_count` instruction is. This
    // instruction is passed as an operand to the body, so its
    // value category will be encoded in the tuple result of the
    // pipeline body. Which means that if its not identical then
    // neither is the pipeline.
    VLOG(3) << "Handling pipeline '" << comp->name() << "' as repeat loop.";
    return HandleRepeatLoop(call, comp, GetPipelineRepeatCount(call));
  } else if (IsRepeatLoop(call)) {
    return HandleRepeatLoop(call, comp, GetRepeatLoopCount(call));
  }

  return SetAllInstructionValuesToDiffering(call);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleConditional(
    const HloInstruction* inst) {
  const std::vector<HloComputation*>& branches = inst->branch_computations();

  // Since we don't know which branch will be taken we have to makes sure that
  // any value we set as identical is identical in all branches and that the
  // replicas will all take the same branch. This way if a value is identical we
  // know that it is regardless of the branch taken.
  const bool same_branch_per_replica =
      IsResultIdentical(value_category_mapping_.at(inst->operand(0)));
  if (same_branch_per_replica) {
    CHECK(!branches.empty());

    // Get the value categories for each branch and merge them together so
    // that the only values that are identical are those that are identical
    // in all branches.
    VLOG(3) << "HandleConditional visit branch 0.";
    TF_ASSIGN_OR_RETURN(
        value_category_mapping_[inst],
        VisitSubComputation(branches[0],
                            value_category_mapping_.at(inst->operand(1))));

    for (auto branch_index = 1u; branch_index < branches.size();
         ++branch_index) {
      const HloComputation* branch = branches[branch_index];
      const HloInstruction* branch_arg = inst->operand(branch_index + 1);

      VLOG(3) << "HandleConditional visit branch " << branch_index << ".";
      TF_ASSIGN_OR_RETURN(
          const ValueCategoryTree branch_categories,
          VisitSubComputation(branch, value_category_mapping_.at(branch_arg)));

      value_category_mapping_[inst] = MakeValuesIdenticalIfTreeElementsAre(
          branch_categories, value_category_mapping_[inst]);
    }
    return Status::OK();
  }

  // If different replicas take different branches then some replicas will visit
  // true_comp and some false_comp, so the values will be differing across
  // replicas.
  for (auto* branch : branches) {
    TF_RETURN_IF_ERROR(SetAllInstructionValuesToDiffering(branch));
  }
  return SetAllInstructionValuesToDiffering(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleCustomCall(
    const HloInstruction* inst) {
  if (IsPoplarInstruction(PoplarOp::AllGather, inst)) {
    return HandleAllGather(inst);
  } else if (IsPoplarInstruction(PoplarOp::AssumeEqualAcrossReplicas, inst)) {
    // AssumeEqual is a special case were we always want it to be treated
    // as replica identical.
    return SetAllInstructionValuesToIdentical(inst);
  } else if (IsPoplarInstruction(PoplarOp::UserOp, inst)) {
    return HandleUserOp(inst);
  }

  return DefaultAction(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleAllReduce(
    const HloInstruction* inst) {
  const bool reduce_all_replicas = inst->replica_groups().empty();
  return SetAllInstructionValuesToIdenticalOrDiffering(inst,
                                                       reduce_all_replicas);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleFusion(
    const HloInstruction* inst) {
  if (IsPopOpsFusion(inst, "wide_const")) {
    return SetAllInstructionValuesToIdenticalOrDiffering(inst, true);
  }

  return DefaultAction(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleGetTupleElement(
    const HloInstruction* inst) {
  auto* tuple = inst->operand(0);
  auto tuple_index = inst->tuple_index();

  auto value_categories = ValueCategoryTree(inst->shape());
  value_categories.CopySubtreeFrom(value_category_mapping_.at(tuple),
                                   {tuple_index}, RootShapeIndex());
  value_category_mapping_[inst] = std::move(value_categories);

  return Status::OK();
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleTuple(
    const HloInstruction* inst) {
  auto value_categories = ValueCategoryTree(inst->shape());
  // Setup the child nodes of the shape tree.
  for (auto i = 0l; i < inst->operand_count(); ++i) {
    value_categories.CopySubtreeFrom(
        value_category_mapping_.at(inst->operand(i)), RootShapeIndex(), {i});
  }
  value_category_mapping_[inst] = std::move(value_categories);

  // Setup the root node of the shape tree.
  SetInstrucionValueToIdenticalOrDiffering(inst, RootShapeIndex(),
                                           AllOperandsIdentical(inst));
  return Status::OK();
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleTupleSelect(
    const HloInstruction* inst) {
  // Since we don't know which tuple the select will return we can only
  // say a value is identical if its in both on/off tuples and the pred
  // is identical across replicas - so we get the same tuple across all
  // replicas.
  const bool sample_tuple_per_replica =
      IsResultIdentical(value_category_mapping_.at(inst->operand(0)));
  if (sample_tuple_per_replica) {
    const ValueCategoryTree& on_value_categories =
        value_category_mapping_.at(inst->operand(1));
    const ValueCategoryTree& off_value_categories =
        value_category_mapping_.at(inst->operand(2));

    value_category_mapping_[inst] = MakeValuesIdenticalIfTreeElementsAre(
        on_value_categories, off_value_categories);
    return Status::OK();
  }

  return SetAllInstructionValuesToDiffering(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleWhile(
    const HloInstruction* inst) {
  HloComputation* while_condition = inst->while_condition();
  HloComputation* while_body = inst->while_body();

  // We don't know ahead of time how many iterations, if any, the while
  // loop will run for. Due to that we can only say that an output value
  // is identical when the loop runs for the same number of iterations across
  // all replicas and if it's identical when the loop doesn't run, runs once
  // and runs multiple times (each of which correspond to different behaviours
  // of the conditional). Outside of those conditions we can't be sure what
  // values are identical, as it may differ depending on the number of
  // iterations.

  VLOG(3) << "HandleWhile visit conditional start.";
  TF_ASSIGN_OR_RETURN(const ValueCategoryTree conditional_start_categories,
                      VisitSubComputation(while_condition, inst));
  if (IsResultIdentical(conditional_start_categories)) {
    VLOG(3) << "HandleWhile visit body iter0.";
    TF_ASSIGN_OR_RETURN(const ValueCategoryTree body_iter0_categories,
                        VisitSubComputation(while_body, inst));

    // At this point we know that the conditional is replica idential
    // initially, now we need to check that it remains so for iter0 and
    // iter1. This way we can know that the replicas have the same number
    // of iterations.
    VLOG(3) << "HandleWhile visit conitional iter0.";
    TF_ASSIGN_OR_RETURN(
        const ValueCategoryTree conditional_iter0_categories,
        VisitSubComputation(while_condition, body_iter0_categories));
    if (IsResultIdentical(conditional_iter0_categories)) {
      // We treat this point as the loops fixed point since we expect
      // the input and output categories from here to be the same.
      VLOG(3) << "HandleWhile visit body iter1.";
      TF_ASSIGN_OR_RETURN(
          const ValueCategoryTree body_iter1_categories,
          VisitSubComputation(while_body, body_iter0_categories));

      VLOG(3) << "HandleWhile visit conitional iter1.";
      TF_ASSIGN_OR_RETURN(
          const ValueCategoryTree conditional_iter1_categories,
          VisitSubComputation(while_condition, body_iter1_categories));
      if (IsResultIdentical(conditional_iter1_categories)) {
        // At this point we know that the replicas have the same
        // number of iterations, so we just need to find which
        // values of the loop itertions are also identical.

        const ValueCategoryTree& body_start_categories =
            value_category_mapping_.at(inst->operand(0));

        value_category_mapping_[inst] = MakeValuesIdenticalIfTreeElementsAre(
            body_start_categories,
            MakeValuesIdenticalIfTreeElementsAre(body_iter0_categories,
                                                 body_iter1_categories));
        return Status::OK();
      }
    }
  }

  // If the conditional isn't identical then the number of iterations
  // differ between the replicas and so will the loop body.
  TF_RETURN_IF_ERROR(SetAllInstructionValuesToDiffering(while_body));
  return SetAllInstructionValuesToDiffering(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleAllGather(
    const HloInstruction* inst) {
  const HloModule* module = inst->GetModule();
  const int64 replica_count = module->config().replica_count();

  const auto* all_gather = Cast<HloPoplarAllGatherInstruction>(inst);
  const PoplarReplicaGroups replica_group =
      all_gather->GetPoplarReplicaGroups();

  VLOG(3) << "HandleAllGather checking whether group " << replica_group
          << " contains all " << replica_count << " replicas.";
  // A default constructed PoplarReplicaGroups refers to a single group
  // containing all replicas. Similarly a consective PoplarReplicaGroups of
  // size replica_count also represents a single group containing all replicas.
  const bool gather_all_replicas =
      replica_group == PoplarReplicaGroups() ||
      replica_group == PoplarReplicaGroups::Consecutive(replica_count);
  return SetAllInstructionValuesToIdenticalOrDiffering(all_gather,
                                                       gather_all_replicas);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleRepeatLoop(
    const HloInstruction* call, const HloComputation* body,
    int64 repeat_count) {
  CHECK_GT(repeat_count, 0);

  if (repeat_count > 1) {
    // To determine the value categories of a repeat body we need to run the
    // visitor twice, since the body is first run with initial values (set via
    // the calls operands) and then with the results from the previous call to
    // body. Hence we run the visitor with the initial value categories given to
    // body and then with the value categories of that first iteration, which we
    // use as the actual categories.
    VLOG(3) << "HandleRepeat visit body iter0.";
    TF_ASSIGN_OR_RETURN(ValueCategoryTree body_iter0_categories,
                        VisitSubComputation(body, call));

    // If a repeat body has only a single parameter then its ROOT value can be
    // one of two shapes. Either a single value that matches the shape of the
    // parameter or a single value (of the correct shape) in a tuple. If
    // its a wrapper then we need to unwrap it to get the correct
    // ValueCategoryTree to use, otherwise we will have a shape mismatch.
    const bool unwrap_root =
        body->num_parameters() == 1 &&
        IsWrapperTuple(body_iter0_categories.shape(),
                       body->parameter_instruction(0)->shape());
    if (unwrap_root) {
      TF_ASSIGN_OR_RETURN(body_iter0_categories,
                          body_iter0_categories.SubShapeTree({0}));
    }

    VLOG(3) << "HandleRepeat visit body iter1.";
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(body, body_iter0_categories));
  } else {
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(body, call));
  }

  return Status::OK();
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleUserOp(
    const HloInstruction* inst) {
  const auto* user_op = Cast<HloUserOpInstruction>(inst);

  // This set contains the output indices that are replica identical. So if our
  // user op returns a tuple then the values of that tuple are identical if
  // their index is in this set. Otherwise we only produce a single value and so
  // are replica identical if the set contains index 0.
  const absl::flat_hash_set<int64>& replica_identical_output_indices =
      user_op->ReplicaIdenticalOutputIndices();

  const auto user_op_shape = user_op->shape();
  if (user_op_shape.IsTuple()) {
    value_category_mapping_[inst] = ValueCategoryTree(user_op_shape);

    bool all_values_identical = true;
    for (int64 i = 0, end = ShapeUtil::TupleElementCount(user_op_shape);
         i < end; ++i) {
      const bool output_identical =
          replica_identical_output_indices.contains(i);
      all_values_identical &= output_identical;

      SetInstrucionValueToIdenticalOrDiffering(inst, {i}, output_identical);
    }
    SetInstrucionValueToIdenticalOrDiffering(inst, RootShapeIndex(),
                                             all_values_identical);
  } else {
    CHECK_LE(replica_identical_output_indices.size(), 1);
    return SetAllInstructionValuesToIdenticalOrDiffering(
        user_op, replica_identical_output_indices.contains(0));
  }

  return Status::OK();
}

StatusOr<ValueCategoryTree>
ValuesIdenticalAcrossReplicasVisitor::VisitSubComputation(
    const HloComputation* comp, const HloInstruction* call) {
  // To determine the value categories of a particular call we run the visitor
  // with a set of overrides so that the categories of the computations
  // parameters are the same as the calls operands. This should produce the
  // same result as if the computations parameters were replaced with the
  // calls operands but without the overhead of creating a new
  // module/computation etc.
  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      parameter_overrides = CreateParameterOverridesForCall(call, comp);

  return VisitSubComputation(comp, parameter_overrides);
}

StatusOr<ValueCategoryTree>
ValuesIdenticalAcrossReplicasVisitor::VisitSubComputation(
    const HloComputation* comp, const ValueCategoryTree& parameter_categories) {
  const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      parameter_overrides =
          CreateParameterOverridesFromCategories(comp, parameter_categories);
  return VisitSubComputation(comp, parameter_overrides);
}

StatusOr<ValueCategoryTree>
ValuesIdenticalAcrossReplicasVisitor::VisitSubComputation(
    const HloComputation* comp,
    const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
        parameter_overrides) {
  ValuesIdenticalAcrossReplicasVisitor comp_visitor(parameter_overrides);

  VLOG(3) << "Running replica dataflow analysis on '" << comp->name()
          << "' computation.";
  TF_RETURN_IF_ERROR(comp->Accept(&comp_visitor));

  for (auto& item : comp_visitor.value_category_mapping_) {
    auto item_it = value_category_mapping_.find(item.first);
    if (item_it == value_category_mapping_.end()) {
      value_category_mapping_.insert(std::move(item));
    } else {
      // Instructions may be visited multiple times if they're called via
      // a loop or another computation with repeat like semantics. In these
      // cases we say that a value is identical if it's everytime it's visited.
      auto* inst = item.first;
      value_category_mapping_[inst] = MakeValuesIdenticalIfTreeElementsAre(
          value_category_mapping_[inst], item.second);
    }
  }

  return value_category_mapping_.at(comp->root_instruction());
}

absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
ValuesIdenticalAcrossReplicasVisitor::CreateParameterOverridesForCall(
    const HloInstruction* call, const HloComputation* comp) const {
  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      parameter_overrides;

  const std::vector<HloInstruction*>& params = comp->parameter_instructions();
  CHECK_EQ(params.size(), call->operand_count());

  for (auto i = 0u; i < params.size(); ++i) {
    parameter_overrides[params[i]] =
        value_category_mapping_.at(call->operand(i));
  }

  return parameter_overrides;
}

bool ValuesIdenticalAcrossReplicasVisitor::AllOperandsIdentical(
    const HloInstruction* inst) const {
  bool all_operands_identical = true;
  for (auto* operand : inst->operands()) {
    const auto operand_identical =
        IsResultIdentical(value_category_mapping_.at(operand));
    all_operands_identical &= operand_identical;
  }

  return all_operands_identical;
}

Status ValuesIdenticalAcrossReplicasVisitor::SetAllInstructionValuesToIdentical(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(inst,
                                                       /*identical*/ true);
}

Status ValuesIdenticalAcrossReplicasVisitor::SetAllInstructionValuesToDiffering(
    const HloComputation* comp) {
  struct RecursiveSetDifferingVisitor : ConstDfsHloVisitorWithDefault {
    RecursiveSetDifferingVisitor(
        ValuesIdenticalAcrossReplicasVisitor& outer_visitor)
        : outer_visitor_(outer_visitor) {}

    Status DefaultAction(const HloInstruction* inst) override {
      TF_RETURN_IF_ERROR(
          outer_visitor_.SetAllInstructionValuesToDiffering(inst));

      for (auto* comp : inst->called_computations()) {
        RecursiveSetDifferingVisitor visitor(outer_visitor_);
        TF_RETURN_IF_ERROR(comp->Accept(&visitor));
      }

      return Status::OK();
    }

    ValuesIdenticalAcrossReplicasVisitor& outer_visitor_;
  };

  // Recursivly set all instructions within the given computation
  // to be replica differing.
  RecursiveSetDifferingVisitor set_differing_visitor(*this);
  return comp->Accept(&set_differing_visitor);
}

Status ValuesIdenticalAcrossReplicasVisitor::SetAllInstructionValuesToDiffering(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(inst,
                                                       /*identical*/ false);
}

Status ValuesIdenticalAcrossReplicasVisitor::
    SetAllInstructionValuesToIdenticalOrDiffering(const HloInstruction* inst,
                                                  bool identical) {
  const auto category = identical ? ValueReplicaCategory::Identical
                                  : ValueReplicaCategory::Differing;
  value_category_mapping_[inst] = ValueCategoryTree(inst->shape(), category);
  return Status::OK();
}

void ValuesIdenticalAcrossReplicasVisitor::
    SetInstrucionValueToIdenticalOrDiffering(const HloInstruction* inst,
                                             const ShapeIndex& value_index,
                                             bool identical) {
  const auto category = identical ? ValueReplicaCategory::Identical
                                  : ValueReplicaCategory::Differing;
  *value_category_mapping_[inst].mutable_element(value_index) = category;
}

void ValuesIdenticalAcrossReplicasVisitor::MarkOverridesAsVisited(
    const absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>&
        category_overrides) {
  for (auto& item : category_overrides) {
    auto* inst = item.first;
    SetVisitState(inst->unique_id(), kVisited);
  }
}

Status ReplicaIdenticalDataflowAnalysis::Run(const HloModule* module) {
  auto call_graph = CallGraph::Build(module);
  if (call_graph->IsFlattened()) {
    VLOG(3) << "Starting replica dataflow analysis on entry computation.";

    auto* entry_computation = module->entry_computation();
    TF_RETURN_IF_ERROR(entry_computation->Accept(&value_category_visitor_));

    // Make sure all relevant subcomputations are visited.
    for (auto* comp : module->computations()) {
      const bool is_visitable =
          !Analysed(comp) && IsVisitable(*call_graph, comp);
      if (is_visitable) {
        VLOG(3) << "Running replica dataflow analysis on '" << comp->name()
                << "' computation.";

        TF_RETURN_IF_ERROR(comp->Accept(&value_category_visitor_));
      }
    }

    return Status::OK();
  }

  return FailedPrecondition(
      "Expected the call graph of the module to be flat.");
}

bool ReplicaIdenticalDataflowAnalysis::Analysed(
    const HloComputation* comp) const {
  return value_category_visitor_.Visited(comp);
}

StatusOr<ValueReplicaCategory> ReplicaIdenticalDataflowAnalysis::ValueCategory(
    const HloInstruction* inst, const ShapeIndex& value_index) {
  auto& value_category_mapping = value_category_visitor_.ValueCategoryMapping();
  auto inst_it = value_category_mapping.find(inst);
  if (inst_it != value_category_mapping.end()) {
    const auto category = inst_it->second.element(value_index);
    CHECK_NE(category, ValueReplicaCategory::Unknown);

    return category;
  }

  return InternalErrorStrCat(
      "Value category for instruction '", inst->name(),
      "' not found. Run the visitor on its module to find its value "
      "category.");
}

StatusOr<bool> ReplicaIdenticalDataflowAnalysis::IsValueIdenticalAcrossReplicas(
    const HloInstruction* inst, const ShapeIndex& value_index) {
  TF_ASSIGN_OR_RETURN(ValueReplicaCategory category,
                      ValueCategory(inst, value_index));
  return category == ValueReplicaCategory::Identical;
}

}  // namespace poplarplugin
}  // namespace xla
