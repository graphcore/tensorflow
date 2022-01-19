/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/instruction_colocator_helper.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "google/protobuf/util/message_differencer.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_many.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
namespace {
int64 ByteSizeOfIncludingTuple(const Shape& shape) {
  if (shape.IsTuple()) {
    int64 result = 0;

    for (auto i = 0; i < shape.tuple_shapes_size(); ++i) {
      result += ByteSizeOfIncludingTuple(shape.tuple_shapes(i));
    }

    return result;
  }

  if (shape.IsOpaque()) {
    return 0;
  }

  return ShapeUtil::ByteSizeOf(shape);
}

HloInstruction* AddReplacementInstruction(
    HloComputation* comp, HloInstruction* original,
    std::unique_ptr<HloInstruction> replacement) {
  auto new_inst = comp->AddInstruction(std::move(replacement));
  CopyShardingIfPresent(original, new_inst);

  return new_inst;
}

StatusOr<HloInstruction*> AddGTEInstruction(HloInstruction* operand,
                                            int64 index,
                                            HloInstruction* to_replace) {
  TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                      MakeGetTupleElementHlo(operand, index));
  // Mark it as inplace.
  MakeUsedInplace(gte);
  CopyShardingIfPresent(to_replace, gte);

  return gte;
}

// Returns a concatenated tuple shape from the shapes of the given instructions.
static Shape CombineShapes(const std::vector<HloInstruction*>& to_combine) {
  std::vector<Shape> combined_shapes;
  // Lower bound for size.
  combined_shapes.reserve(to_combine.size());

  for (HloInstruction* inst : to_combine) {
    const Shape& shape = inst->shape();
    if (shape.IsTuple()) {
      for (const Shape& tuple_shape : shape.tuple_shapes()) {
        combined_shapes.push_back(tuple_shape);
      }
    } else {
      combined_shapes.push_back(shape);
    }
  }
  return ShapeUtil::MakeTupleShape(combined_shapes);
}

// Returns a concatenated list of the operands of the given instructions.
static std::vector<HloInstruction*> CombineOperands(
    const std::vector<HloInstruction*>& to_combine) {
  std::vector<HloInstruction*> combined_operands;
  // Lower bound for size.
  combined_operands.reserve(to_combine.size());

  for (HloInstruction* inst : to_combine) {
    for (HloInstruction* operand : inst->operands()) {
      combined_operands.push_back(operand);
    }
  }
  return combined_operands;
}
}  // namespace

InstructionColocatorHelper::InstructionColocatorHelper(
    bool requires_matching_element_types)
    : id_(GetNextID()),
      requires_matching_element_types_(requires_matching_element_types) {}

int64 InstructionColocatorHelper::GetID() const { return id_; }

int64 InstructionColocatorHelper::GetNextID() {
  static int64 id = 0;
  return id++;
}

int64 InstructionColocatorHelper::ByteSizeOf(const HloInstruction* inst) const {
  return ByteSizeOfIncludingTuple(inst->shape());
}

bool InstructionColocatorHelper::CanColocateExtra(
    const HloInstruction* a, const HloInstruction* b) const {
  return true;
}

bool InstructionColocatorHelper::CanColocate(const HloInstruction* a,
                                             const HloInstruction* b) const {
  if (!CanColocate(a) || !CanColocate(b)) {
    return false;
  }

  // We don't support tuple shapes on inputs.
  for (const HloInstruction* inst : {a, b}) {
    for (const HloInstruction* operand : inst->operands()) {
      if (operand->shape().IsTuple()) {
        return false;
      }
    }
  }

  // Make sure a and b have compatible sharding.
  if (!CanColocateSharding(a, b)) {
    return false;
  }

  // The two instructions must have the same inplaceness.
  if (IsLoweredInplace(a) != IsLoweredInplace(b)) {
    return false;
  }

  // The two instructions must have the same element type if
  // requires_matching_element_types_ is true for this colocator helper.
  if (requires_matching_element_types_ &&
      (a->shape().element_type() != b->shape().element_type())) {
    return false;
  }

  return CanColocateExtra(a, b);
}

bool InstructionColocatorHelper::CanColocateSharding(
    const HloInstruction* a, const HloInstruction* b) const {
  return a->has_compatible_sharding(b);
}

bool InstructionColocatorHelperPtrComparator::operator()(
    const InstructionColocatorHelper* const& lhs,
    const InstructionColocatorHelper* const& rhs) const {
  if (rhs == nullptr) {
    // Nothing compares less than nullptr.
    return false;
  }
  if (lhs == nullptr) {
    return true;
  }
  return lhs->GetID() < rhs->GetID();
}

StatusOr<HloInstruction*>
InstructionColocatorHelper::CombineColocatedInstructions(
    const std::vector<HloInstruction*>& to_combine) const {
  HloInstruction* archetype = to_combine.front();

  Shape shape = CombineShapes(to_combine);
  std::vector<HloInstruction*> operands = CombineOperands(to_combine);

  // Add the new instruction.
  HloInstruction* new_inst = AddReplacementInstruction(
      archetype->parent(), archetype,
      archetype->CloneWithNewOperands(shape, operands));

  return new_inst;
}

StatusOr<std::vector<HloInstruction*>>
InstructionColocatorHelper::CombineAndReplaceColocatedInstructions(
    std::vector<HloInstruction*> to_combine) const {
  const uint64 cluster_size = to_combine.size();
  if (cluster_size == 1) {
    return to_combine;
  }

  TF_ASSIGN_OR_RETURN(auto* new_inst, CombineColocatedInstructions(to_combine));
  auto* comp = new_inst->parent();

  // Replace all the users.
  std::vector<HloInstruction*> result;
  result.push_back(new_inst);

  // Delete old instructions and replace uses.
  int64 output_index_offset = 0;
  for (HloInstruction* old_inst : to_combine) {
    TF_RETURN_IF_ERROR(new_inst->CopyAllControlDepsFrom(old_inst));
    TF_RETURN_IF_ERROR(old_inst->DropAllControlDeps());

    if (old_inst->shape().IsTuple()) {
      // Record the number of users before adding the users from old_inst.
      // This is used to iterate through only the new users.
      int64 new_users_offset = new_inst->user_count();
      TF_RETURN_IF_ERROR(old_inst->ReplaceAllUsesWithDifferentShape(new_inst));

      // Iterate through new users and update GTE indexes.
      for (auto itr = new_inst->users().begin() + new_users_offset;
           itr != new_inst->users().end(); ++itr) {
        HloInstruction* gte = *itr;
        CHECK(gte->opcode() == HloOpcode::kGetTupleElement);
        gte->set_tuple_index(gte->tuple_index() + output_index_offset);
      }
      // Used to calculate updated gte index values.
      output_index_offset += old_inst->operand_count();
    } else {
      // Add a GTE to unpack the result from the output tuple.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * gte,
          AddGTEInstruction(new_inst, output_index_offset, old_inst));
      TF_RETURN_IF_ERROR(old_inst->ReplaceAllUsesWith(gte));
      // Include new GTE in result.
      result.push_back(gte);
      ++output_index_offset;
    }

    // Remove old instruction.
    TF_RETURN_IF_ERROR(comp->ForceRemoveInstruction(old_inst));
  }
  return result;
}

namespace {
// Manager for all the colocators.
class InstructionColocatorHelperManager {
 public:
  static InstructionColocatorHelperManager& GetInstance() {
    static InstructionColocatorHelperManager instance;
    return instance;
  }

  void AddInstructionColocatorHelper(
      std::unique_ptr<InstructionColocatorHelper> colocator) {
    colocators.push_back(std::move(colocator));
    colocators_refs.push_back(colocators.back().get());
  }

  const std::vector<const InstructionColocatorHelper*>&
  GetAllInstructionColocatorHelpers() const {
    return colocators_refs;
  }

 private:
  InstructionColocatorHelperManager() {}

  std::vector<std::unique_ptr<InstructionColocatorHelper>> colocators;
  std::vector<const InstructionColocatorHelper*> colocators_refs;
};

// Registrar
class InstructionColocatorHelperRegistrar {
 public:
  InstructionColocatorHelperRegistrar(
      std::unique_ptr<InstructionColocatorHelper> colocator) {
    InstructionColocatorHelperManager::GetInstance()
        .AddInstructionColocatorHelper(std::move(colocator));
  }

  InstructionColocatorHelperRegistrar() = delete;
};

#define REGISTER_INSTRUCTION_COLOCATOR_HELPER(colocator)               \
  namespace {                                                          \
  static InstructionColocatorHelperRegistrar                           \
      registrar__colocator__##colocator##__object(                     \
          std::unique_ptr<InstructionColocatorHelper>(new colocator)); \
  }

// Colocator helper which is used to combine multiple all reduce instructions.
class AllReduceColocatorHelper : public InstructionColocatorHelper {
 public:
  AllReduceColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return inst->opcode() == HloOpcode::kAllReduce;
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_reduce_buffer_size;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    auto replica_group_cmp = [](const xla::ReplicaGroup& g1,
                                const xla::ReplicaGroup& g2) {
      return google::protobuf::util::MessageDifferencer::Equals(g1, g2);
    };

    // Make sure the same to_apply() computation and replica groups are used.
    return *a->to_apply() == *b->to_apply() &&
           absl::c_equal(a->replica_groups(), b->replica_groups(),
                         replica_group_cmp);
  }
};

// Colocator helper which is used to combine multiple inter-IPU copies.
class InterIpuCopyColocatorHelper : public InstructionColocatorHelper {
 public:
  InterIpuCopyColocatorHelper()
      : InstructionColocatorHelper(/*requires_matching_element_types=*/false) {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::InterIpuCopy)(inst);
  }

  bool CanColocateSharding(const HloInstruction* a,
                           const HloInstruction* b) const override {
    // We can merge inter-IPU copies with different sharding information.
    return true;
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_inter_ipu_copies_buffer_size;
  }

  StatusOr<HloInstruction*> CombineColocatedInstructions(
      const std::vector<HloInstruction*>& to_combine) const override {
    TF_ASSIGN_OR_RETURN(
        auto* new_inst,
        InstructionColocatorHelper::CombineColocatedInstructions(to_combine));
    std::vector<HloSharding> new_sharding;
    for (const auto* inst : to_combine) {
      new_sharding.push_back(inst->sharding());
    }

    CHECK_EQ(new_inst->shape().tuple_shapes_size(), new_sharding.size());
    new_inst->set_sharding(HloSharding::Tuple(new_inst->shape(), new_sharding));

    return new_inst;
  }
};

class ReduceScatterColocatorHelper : public InstructionColocatorHelper {
 public:
  ReduceScatterColocatorHelper()
      : InstructionColocatorHelper(
            /*requires_matching_element_types=*/false) {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::ReduceScatter)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_reduce_scatter_buffer_size;
  }

  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    auto* ra = Cast<HloReduceScatterInstruction>(a);
    auto* rb = Cast<HloReduceScatterInstruction>(b);

    return ra->GetCollectiveOperator() == rb->GetCollectiveOperator() &&
           ra->GetPoplarReplicaGroups() == rb->GetPoplarReplicaGroups();
  }
};

class SendToHostColocatorHelper : public InstructionColocatorHelper {
 public:
  SendToHostColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::SendToHost)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_send_recv_cluster_size;
  }

  int64 ByteSizeOf(const HloInstruction* inst) const override {
    // The size of the SendToHost is determined by the size of its inputs.
    return absl::c_accumulate(
        inst->operands(), int64{0}, [](int64 acc, const HloInstruction* input) {
          return acc + ByteSizeOfIncludingTuple(input->shape());
        });
  }

  StatusOr<std::vector<HloInstruction*>> CombineAndReplaceColocatedInstructions(
      std::vector<HloInstruction*> to_combine) const override {
    CHECK(!to_combine.empty());
    if (to_combine.size() == 1) {
      return to_combine;
    }

    std::vector<HloSendToHostInstruction*> old_sends;
    for (HloInstruction* inst : to_combine) {
      old_sends.push_back(Cast<HloSendToHostInstruction>(inst));
    }

    auto* first_send = old_sends[0];
    HloComputation* comp = first_send->parent();

    std::vector<HloInstruction*> new_operands = CombineOperands(to_combine);
    const Shape new_shape = CombineShapes(to_combine);

    std::vector<std::string> new_rendezvous_keys;
    for (auto* old_send : old_sends) {
      CHECK_EQ(old_send->operand_count(), 1);
      CHECK_EQ(old_send->RendezvousKeys().size(), 1);
      new_rendezvous_keys.push_back(old_send->RendezvousKeys()[0]);
    }

    // Add the new instruction.
    HloInstruction* new_send = AddReplacementInstruction(
        comp, first_send,
        first_send->CloneWithNewOperandsAndRendezvousKeys(
            new_shape, new_operands, new_rendezvous_keys));

    for (auto* old_send : old_sends) {
      TF_RETURN_IF_ERROR(new_send->CopyAllControlDepsFrom(old_send));
      TF_RETURN_IF_ERROR(old_send->DropAllControlDeps());
      // This will do safety checks like confirming that there are no
      // users of send instruction.
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(old_send));
    }

    return std::vector<HloInstruction*>{new_send};
  }
};

class RecvFromHostColocatorHelper : public InstructionColocatorHelper {
 public:
  RecvFromHostColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::RecvFromHost)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_send_recv_cluster_size;
  }

  StatusOr<std::vector<HloInstruction*>> CombineAndReplaceColocatedInstructions(
      std::vector<HloInstruction*> to_combine) const override {
    CHECK(!to_combine.empty());
    if (to_combine.size() == 1) {
      return to_combine;
    }

    std::vector<HloRecvFromHostInstruction*> old_recvs;
    for (HloInstruction* inst : to_combine) {
      old_recvs.push_back(Cast<HloRecvFromHostInstruction>(inst));
    }

    auto* first_recv = old_recvs[0];
    HloComputation* comp = first_recv->parent();

    std::vector<HloInstruction*> new_operands = CombineOperands(to_combine);
    const Shape new_shape = CombineShapes(to_combine);

    std::vector<std::string> new_rendezvous_keys;
    for (auto* old_recv : old_recvs) {
      CHECK_EQ(old_recv->RendezvousKeys().size(), 1);
      new_rendezvous_keys.push_back(old_recv->RendezvousKeys()[0]);
    }

    // Add the new instruction.
    HloInstruction* new_recv = AddReplacementInstruction(
        comp, first_recv,
        first_recv->CloneWithNewOperandsAndRendezvousKeys(
            new_shape, new_operands, new_rendezvous_keys));

    std::vector<HloInstruction*> result;
    result.push_back(new_recv);

    // Replace all the users.
    for (uint64 i = 0; i != old_recvs.size(); ++i) {
      HloInstruction* old_recv = old_recvs[i];
      // Add a GTE to unpack the new_recv result.
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          AddGTEInstruction(new_recv, i, old_recv));
      result.push_back(gte);

      // Replace the old inst.
      TF_RETURN_IF_ERROR(new_recv->CopyAllControlDepsFrom(old_recv));
      TF_RETURN_IF_ERROR(old_recv->DropAllControlDeps());
      TF_RETURN_IF_ERROR(old_recv->ReplaceAllUsesWith(gte));
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(old_recv));
    }

    return result;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    // They must have the same number of operands, otherwise we
    // are not able to match them up with the rendezvous keys.
    return a->operand_count() == b->operand_count();
  }
};

// Colocator helper which is used to combine multiple gradient accumulations and
// all reduce instructions.
class StatefulGradientAccumulationAllReduceColocatorHelper
    : public InstructionColocatorHelper {
 public:
  StatefulGradientAccumulationAllReduceColocatorHelper()
      : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(
        PoplarOp::StatefulGradientAccumulateAndAllReduce)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_reduce_buffer_size;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    auto a_cast = Cast<HloStatefulGradientAccumulateAndAllReduce>(a);
    auto b_cast = Cast<HloStatefulGradientAccumulateAndAllReduce>(b);
    // Make accumulate the same number of batches.
    return a_cast->MiniBatchesToAccumulate() ==
           b_cast->MiniBatchesToAccumulate();
  }
};

// Colocator helper which is used to combine multiple gradient accumulations
// with momentum and all reduce/normalize instructions.
class StatefulGradientAccumulateWithMomentumAndAllReduceWithNormColocatorHelper
    : public InstructionColocatorHelper {
 public:
  StatefulGradientAccumulateWithMomentumAndAllReduceWithNormColocatorHelper()
      : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(
        PoplarOp::StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm)(
        inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_reduce_buffer_size;
  }

  StatusOr<std::vector<HloInstruction*>> CombineAndReplaceColocatedInstructions(
      std::vector<HloInstruction*> to_combine) const override {
    const uint64 cluster_size = to_combine.size();
    if (cluster_size == 1) {
      return to_combine;
    }

    HloComputation* comp = to_combine.front()->parent();
    // The inputs to a gradient accumulation with momentum are:
    // 0 - accumulator
    // 1 - gradient
    // 2 - momentum
    // And outputs are:
    // 0 - accumulator
    // 1 - gradient
    // Given N instructions, the combined instruction inputs are:
    // 0 to N-1  - accumulators
    // N to 2N-1 - gradients
    // 2N        - momentum
    // And outputs are:
    // 0 to N-1  - accumulators
    // N to 2N-1 - gradients
    std::vector<HloInstruction*> operands(2 * cluster_size + 1);
    std::vector<Shape> output_shapes(2 * cluster_size);
    for (uint64 i = 0; i != cluster_size; ++i) {
      HloInstruction* accum = to_combine[i]->mutable_operand(0);
      HloInstruction* grad = to_combine[i]->mutable_operand(1);
      operands[i] = accum;
      output_shapes[i] = accum->shape();
      operands[cluster_size + i] = grad;
      output_shapes[cluster_size + i] = grad->shape();
    }
    // Momentum operand.
    operands[2 * cluster_size] = to_combine.front()->mutable_operand(2);

    auto shape = ShapeUtil::MakeTupleShape(output_shapes);

    // Add the new instruction.
    HloInstruction* new_inst = AddReplacementInstruction(
        comp, to_combine.front(),
        to_combine.front()->CloneWithNewOperands(shape, operands));

    // Replace all the users.
    std::vector<HloInstruction*> result(3 * cluster_size + 1);
    result[0] = new_inst;
    for (uint64 i = 0; i != cluster_size; ++i) {
      HloInstruction* inst = to_combine[i];
      // GTE to get the new accumulator.
      TF_ASSIGN_OR_RETURN(HloInstruction * gte_accum,
                          AddGTEInstruction(new_inst, i, inst));
      result[i + 1] = gte_accum;

      // GTE to get the new gradient.
      TF_ASSIGN_OR_RETURN(HloInstruction * gte_grad,
                          AddGTEInstruction(new_inst, cluster_size + i, inst));
      result[cluster_size + i + 1] = gte_grad;

      // Create a tuple instruction so that we can easily replace all the users.
      auto tuple = comp->AddInstruction(
          HloInstruction::CreateTuple({gte_accum, gte_grad}));
      MakeUsedInplace(tuple);
      CopyShardingIfPresent(new_inst, tuple);
      result[2 * cluster_size + i + 1] = tuple;

      // Replace the old inst.
      TF_RETURN_IF_ERROR(tuple->CopyAllControlDepsFrom(inst));
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(tuple));
      TF_RETURN_IF_ERROR(comp->ForceRemoveInstruction(inst));
    }
    return result;
  }

 protected:
  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    // Only colocate if they have the same momentum.
    if (a->operand(2) != b->operand(2)) {
      return false;
    }
    auto a_cast =
        Cast<HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm>(a);
    auto b_cast =
        Cast<HloStatefulGradientAccumulateWithMomentumAndAllReduceWithNorm>(b);
    // Make accumulate the same number of batches.
    return a_cast->MiniBatchesToAccumulate() ==
           b_cast->MiniBatchesToAccumulate();
  }
};

// Colocator combines simple reductions into ReduceMany instructions.
class ReduceManyColocatorHelper : public InstructionColocatorHelper {
 public:
  ReduceManyColocatorHelper()
      : InstructionColocatorHelper(
            /*requires_matching_element_types=*/false) {}

  bool CanColocate(const HloInstruction* inst) const override {
    return (inst->opcode() == HloOpcode::kReduce) || IsReductionFusion(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_reduce_many_buffer_size;
  }

  StatusOr<HloInstruction*> CombineColocatedInstructions(
      const std::vector<HloInstruction*>& to_combine) const override {
    HloInstruction* archetype = to_combine.front();

    std::vector<HloInstruction*> new_operands = CombineOperands(to_combine);
    const Shape new_shape = CombineShapes(to_combine);

    std::vector<ReductionInfo> reductions_info;
    reductions_info.reserve(to_combine.size());
    for (HloInstruction* old_reduce : to_combine) {
      const auto old_operand_count = old_reduce->operand_count();
      const bool with_scale = old_operand_count == 3;
      CHECK(old_operand_count == 2 || with_scale);

      TF_ASSIGN_OR_RETURN(ReductionInfo reduction_info,
                          GetReductionInfo(old_reduce, with_scale));
      reductions_info.push_back(reduction_info);
    }

    // Add the new instruction.
    HloInstruction* new_inst = AddReplacementInstruction(
        archetype->parent(), archetype,
        CreatePoplarReduceMany(new_operands, new_shape, reductions_info));

    return new_inst;
  }
};

// A colocator to combine several AllGather instructions into a single
// instruction.
class AllGatherColocatorHelper : public InstructionColocatorHelper {
 public:
  AllGatherColocatorHelper()
      : InstructionColocatorHelper(
            /*requires_matching_element_types=*/false) {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::AllGather)(inst);
  }

  bool CanColocateExtra(const HloInstruction* a,
                        const HloInstruction* b) const override {
    auto a_rg =
        Cast<HloPoplarAllGatherInstruction>(a)->GetPoplarReplicaGroups();
    auto b_rg =
        Cast<HloPoplarAllGatherInstruction>(b)->GetPoplarReplicaGroups();

    return a_rg == b_rg;
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_all_gather_buffer_size;
  }
};

}  // namespace

REGISTER_INSTRUCTION_COLOCATOR_HELPER(InterIpuCopyColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(AllReduceColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(ReduceScatterColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(
    StatefulGradientAccumulationAllReduceColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(
    StatefulGradientAccumulateWithMomentumAndAllReduceWithNormColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(ReduceManyColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(AllGatherColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(SendToHostColocatorHelper)
REGISTER_INSTRUCTION_COLOCATOR_HELPER(RecvFromHostColocatorHelper)

const std::vector<const InstructionColocatorHelper*>&
GetAllInstructionColocatorHelpers() {
  return InstructionColocatorHelperManager::GetInstance()
      .GetAllInstructionColocatorHelpers();
}

absl::optional<const InstructionColocatorHelper*> GetInstructionColocatorHelper(
    const HloInstruction* inst) {
  for (auto colocator : GetAllInstructionColocatorHelpers()) {
    if (colocator->CanColocate(inst)) {
      return colocator;
    }
  }
  return absl::nullopt;
}

bool CanColocate(const HloInstruction* a, const HloInstruction* b) {
  auto colocator = GetInstructionColocatorHelper(a);
  return colocator ? (*colocator)->CanColocate(a, b) : false;
}

#undef REGISTER_INSTRUCTION_COLOCATOR_HELPER
}  // namespace poplarplugin
}  // namespace xla
