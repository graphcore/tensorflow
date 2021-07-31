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
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
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

// Create a new ValueCategoryTree whose nodes have
// ValueReplicaCategory::Identity if the corresponding nodes in both lhs and rhs
// are also identical.
ValueCategoryTree MakeValuesIdenticalIfTreeElementsAre(
    const ValueCategoryTree& lhs, const ValueCategoryTree& rhs) {
  CHECK_EQ(lhs.shape(), rhs.shape());

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

Status ValuesIdenticalAcrossReplicasVisitor::DefaultAction(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(
      inst, AllOperandsIdentical(inst));
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleCall(
    const HloInstruction* call) {
  HloComputation* comp = call->to_apply();

  if (IsFunction(call)) {
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(comp, call));
    return Status::OK();
  } else if (IsRepeatLoop(call)) {
    return HandleRepeatLoop(call, comp);
  }

  return SetAllInstructionValuesToDiffering(call);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleConditional(
    const HloInstruction* inst) {
  const HloComputation* true_comp = inst->true_computation();
  const HloComputation* false_comp = inst->false_computation();

  // Since we don't know which branch will be taken we have to makes sure that
  // any value we set as identical is identical in both branches and that the
  // replicas will all take the same branch. This way if a value is identical we
  // know that it is regardless of the branch taken.
  const bool same_branch_per_replica =
      IsResultIdentical(value_category_mapping_.at(inst->operand(0)));
  if (same_branch_per_replica) {
    TF_ASSIGN_OR_RETURN(
        const ValueCategoryTree true_branch_categories,
        VisitSubComputation(true_comp,
                            value_category_mapping_.at(inst->operand(1))));

    TF_ASSIGN_OR_RETURN(
        const ValueCategoryTree false_branch_categories,
        VisitSubComputation(false_comp,
                            value_category_mapping_.at(inst->operand(2))));

    value_category_mapping_[inst] = MakeValuesIdenticalIfTreeElementsAre(
        true_branch_categories, false_branch_categories);
    return Status::OK();
  }

  // If different replicas take different branches then some replicas will visit
  // true_comp and some false_comp, so the values will be differing across
  // replicas.
  TF_RETURN_IF_ERROR(SetAllInstructionValuesToDiffering(true_comp));
  TF_RETURN_IF_ERROR(SetAllInstructionValuesToDiffering(false_comp));
  return SetAllInstructionValuesToDiffering(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleCustomCall(
    const HloInstruction* inst) {
  if (IsPoplarInstruction(PoplarOp::AllGather, inst)) {
    const auto* all_gather = Cast<HloPoplarAllGatherInstruction>(inst);
    // A default constructed PoplarReplicaGroups refers to a single group
    // containing all replicas
    const auto gather_all_replicas =
        all_gather->GetPoplarReplicaGroups() == PoplarReplicaGroups();
    return SetAllInstructionValuesToIdenticalOrDiffering(all_gather,
                                                         gather_all_replicas);
  }

  return SetAllInstructionValuesToDiffering(inst);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleAllReduce(
    const HloInstruction* inst) {
  const bool reduce_all_replicas = inst->replica_groups().empty();
  return SetAllInstructionValuesToIdenticalOrDiffering(inst,
                                                       reduce_all_replicas);
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleFusion(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(
      inst, IsPopOpsFusion(inst, "wide_const"));
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

  TF_ASSIGN_OR_RETURN(const ValueCategoryTree conditional_start_categories,
                      VisitSubComputation(while_condition, inst));
  if (IsResultIdentical(conditional_start_categories)) {
    TF_ASSIGN_OR_RETURN(const ValueCategoryTree body_iter0_categories,
                        VisitSubComputation(while_body, inst));

    // At this point we know that the conditional is replica idential
    // initially, now we need to check that it remains so for iter0 and
    // iter1. This way we can know that the replicas have the same number
    // of iterations.
    TF_ASSIGN_OR_RETURN(
        const ValueCategoryTree conditional_iter0_categories,
        VisitSubComputation(while_condition, body_iter0_categories));
    if (IsResultIdentical(conditional_iter0_categories)) {
      // We treat this point as the loops fixed point since we expect
      // the input and output categories from here to be the same.
      TF_ASSIGN_OR_RETURN(
          const ValueCategoryTree body_iter1_categories,
          VisitSubComputation(while_body, body_iter0_categories));

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

Status ValuesIdenticalAcrossReplicasVisitor::HandleRepeatLoop(
    const HloInstruction* call, const HloComputation* body) {
  CHECK_GT(GetRepeatLoopCount(call), 0);

  if (GetRepeatLoopCount(call) > 1) {
    // To determine the value categories of a repeat body we need to run the
    // visitor twice, since the body is first run with initial values (set via
    // the calls operands) and then with the results from the previous call to
    // body. Hence we run the visitor with the initial value categories given to
    // body and then with the value categories of that first iteration, which we
    // use as the actual categories.
    TF_ASSIGN_OR_RETURN(const ValueCategoryTree body_iter0_categories,
                        VisitSubComputation(body, call));
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(body, body_iter0_categories));
  } else {
    TF_ASSIGN_OR_RETURN(value_category_mapping_[call],
                        VisitSubComputation(body, call));
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
  auto module_call_graph = CallGraph::Build(module);
  if (module_call_graph->IsFlattened()) {
    auto* entry_computation = module->entry_computation();
    return entry_computation->Accept(&value_category_visitor_);
  }

  return FailedPrecondition(
      "Expected the call graph of the module to be flat.");
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
      "' not found. Run the visitor on its computation to find its value "
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
