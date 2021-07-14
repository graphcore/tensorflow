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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_

#include <ostream>
#include <unordered_map>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/core/lib/core/status.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {

enum class ValueReplicaCategory { Unknown = 0, Identical, Differing };
std::ostream& operator<<(std::ostream& stream,
                         const ValueReplicaCategory& category);

// Visitor for traversing a computation to determine the values of each
// instruction that will be identical across replicas.
class ValuesIdenticalAcrossReplicasVisitor
    : public ConstDfsHloVisitorWithDefault {
 public:
  // Return the ValueReplicaCategory for the given instruction/value_index or an
  // error if the instruction has not already been visited.
  StatusOr<ValueReplicaCategory> ValueCategory(
      const HloInstruction* inst, const ShapeIndex& value_index) const;

  Status DefaultAction(const HloInstruction* inst) override;

  Status HandleCustomCall(const HloInstruction* inst) override;
  Status HandleAllReduce(const HloInstruction* inst) override;
  Status HandleFusion(const HloInstruction* inst) override;
  Status HandleGetTupleElement(const HloInstruction* inst) override;
  Status HandleTuple(const HloInstruction* inst) override;

#define HandleAsReplicaIdentical(TYPE)                       \
  Status Handle##TYPE(const HloInstruction* inst) override { \
    return SetAllInstructionValuesToIdentical(inst);         \
  }

  HandleAsReplicaIdentical(Parameter);
  HandleAsReplicaIdentical(Constant);

#undef HandleAsReplicaIdentical

#define HandleAsReplicaDiffering(TYPE)                       \
  Status Handle##TYPE(const HloInstruction* inst) override { \
    return SetAllInstructionValuesToDiffering(inst);         \
  }

  HandleAsReplicaDiffering(Infeed);
  HandleAsReplicaDiffering(Rng);

  // TODO(T41162): Add support for loops/computations.
  HandleAsReplicaDiffering(Call);
  HandleAsReplicaDiffering(Conditional);
  HandleAsReplicaDiffering(While);

#undef HandleAsReplicaDiffering

 private:
  bool AllOperandsIdentical(const HloInstruction* inst) const;

  Status SetAllInstructionValuesToIdentical(const HloInstruction* inst);
  Status SetAllInstructionValuesToDiffering(const HloInstruction* inst);
  Status SetAllInstructionValuesToIdenticalOrDiffering(
      const HloInstruction* inst, bool identical);

  void SetInstrucionValueToIdenticalOrDiffering(const HloInstruction* inst,
                                                const ShapeIndex& value_index,
                                                bool identical);

  using ValueCategoryTree = ShapeTree<ValueReplicaCategory>;
  std::unordered_map<const HloInstruction*, ValueCategoryTree>
      value_category_mapping_;
};

// Analyse the computation from the given root instruction to find values
// which are identical across replicas.
class ReplicaIdenticalDataflowAnalysis {
 public:
  explicit ReplicaIdenticalDataflowAnalysis(const HloInstruction* root_inst);

  StatusOr<ValueReplicaCategory> ValueCategory(
      const HloInstruction* inst,
      const ShapeIndex& value_index = RootShapeIndex());

  StatusOr<bool> IsValueIdenticalAcrossReplicas(
      const HloInstruction* inst,
      const ShapeIndex& value_index = RootShapeIndex());

 private:
  ValuesIdenticalAcrossReplicasVisitor value_category_visitor_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REPLICA_IDENTICAL_DATAFLOW_ANALYSIS_H_
