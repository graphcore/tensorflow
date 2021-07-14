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

ReplicaIdenticalDataflowAnalysis::ReplicaIdenticalDataflowAnalysis(
    const HloInstruction* root_inst) {
  root_inst->Accept(&value_category_visitor_);
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
