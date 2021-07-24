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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

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
      CHECK(false);
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

StatusOr<ValueReplicaCategory>
ValuesIdenticalAcrossReplicasVisitor::ValueCategory(
    const HloInstruction* inst, const ShapeIndex& value_index) const {
  auto inst_it = value_category_mapping_.find(inst);
  if (inst_it != value_category_mapping_.end()) {
    const auto category = inst_it->second.element(value_index);
    CHECK_NE(category, ValueReplicaCategory::Unknown);

    return category;
  }

  return InternalErrorStrCat(
      "Value category for instruction '", inst->name(),
      "' not found. Run the visitor on its computation to find its value "
      "category.");
}

Status ValuesIdenticalAcrossReplicasVisitor::DefaultAction(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(
      inst, AllOperandsIdentical(inst));
}

Status ValuesIdenticalAcrossReplicasVisitor::HandleCall(
    const HloInstruction* inst) {
  auto* comp = inst->to_apply();

  if (IsFunction(inst)) {
    return SetAllInstructionValuesToMatchComputationRoot(inst, comp);
  }

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

bool ValuesIdenticalAcrossReplicasVisitor::AllOperandsIdentical(
    const HloInstruction* inst) const {
  bool all_operands_identical = true;
  for (auto* operand : inst->operands()) {
    const auto root_category =
        value_category_mapping_.at(operand).element(RootShapeIndex());
    all_operands_identical &= root_category == ValueReplicaCategory::Identical;
  }

  return all_operands_identical;
}

Status ValuesIdenticalAcrossReplicasVisitor::
    SetAllInstructionValuesToMatchComputationRoot(const HloInstruction* caller,
                                                  const HloComputation* comp) {
  // To determine the value categories of a particular call we run the visitor
  // with a set of overrides so that the categories of the computations
  // parameters are the same as the callers operands. This should produce the
  // same result as if the computations parameters were replaced with the
  // callers operands but without the overhead of creating a new
  // module/computation etc.
  absl::flat_hash_map<const HloInstruction*, ValueCategoryTree>
      parameter_overrides;

  auto& params = comp->parameter_instructions();
  CHECK_EQ(params.size(), caller->operand_count());

  for (auto i = 0u; i < params.size(); ++i) {
    parameter_overrides[params[i]] =
        value_category_mapping_.at(caller->operand(i));
  }

  ValuesIdenticalAcrossReplicasVisitor comp_visitor(parameter_overrides);
  comp->Accept(&comp_visitor);

  // Note that even though comp->root_instruction is already in
  // value_category_mapping_ we can't assign it from that since
  // absl::flat_hash_map does not have reference stability, so the reference we
  // try to copy from gets invalidated if value_category_mapping_ has to be
  // reallocated.
  value_category_mapping_[caller] =
      comp_visitor.value_category_mapping_.at(comp->root_instruction());

  // Since the caller is part of a flattened module we get a unique computation
  // per caller, so we can just move across the sub visitor instructions without
  // worrying about collisions.
  value_category_mapping_.insert(
      std::make_move_iterator(comp_visitor.value_category_mapping_.begin()),
      std::make_move_iterator(comp_visitor.value_category_mapping_.end()));

  return Status::OK();
}

Status ValuesIdenticalAcrossReplicasVisitor::SetAllInstructionValuesToIdentical(
    const HloInstruction* inst) {
  return SetAllInstructionValuesToIdenticalOrDiffering(inst,
                                                       /*identical*/ true);
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
  return value_category_visitor_.ValueCategory(inst, value_index);
}

StatusOr<bool> ReplicaIdenticalDataflowAnalysis::IsValueIdenticalAcrossReplicas(
    const HloInstruction* inst, const ShapeIndex& value_index) {
  TF_ASSIGN_OR_RETURN(ValueReplicaCategory category,
                      ValueCategory(inst, value_index));
  return category == ValueReplicaCategory::Identical;
}

}  // namespace poplarplugin
}  // namespace xla
