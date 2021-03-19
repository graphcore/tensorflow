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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

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

  return ShapeUtil::ByteSizeOf(shape);
}
}  // namespace

InstructionColocatorHelper::InstructionColocatorHelper() : id_(GetNextID()) {}

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

  // Make sure a and b have compitable sharding.
  if (!a->has_compatible_sharding(b)) {
    return false;
  }

  // The two instructions must have the same element type.
  if (a->shape().element_type() != b->shape().element_type()) {
    return false;
  }

  // The two instructions must have the same inplaceness.
  if (IsLoweredInplace(a) != IsLoweredInplace(b)) {
    return false;
  }

  return CanColocateExtra(a, b);
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

StatusOr<std::vector<HloInstruction*>>
InstructionColocatorHelper::CombineAndReplaceColocatedInstructions(
    std::vector<HloInstruction*> to_combine) const {
  const uint64 cluster_size = to_combine.size();
  if (cluster_size == 1) {
    return to_combine;
  }

  HloComputation* comp = to_combine[0]->parent();
  // Be default merge all the operands.
  // Combine all the shapes into a single one.
  std::vector<Shape> shapes(cluster_size);
  absl::c_transform(to_combine, shapes.begin(),
                    [](HloInstruction* inst) { return inst->shape(); });
  auto shape = ShapeUtil::MakeTupleShape(shapes);

  // The new list of operands
  auto operands = absl::c_accumulate(
      to_combine, std::vector<HloInstruction*>{},
      [](std::vector<HloInstruction*>& accum, HloInstruction* inst) {
        accum.insert(accum.end(), inst->operands().begin(),
                     inst->operands().end());

        return accum;
      });

  // Add the new instruction.
  HloInstruction* new_inst = comp->AddInstruction(
      to_combine[0]->CloneWithNewOperands(shape, operands));
  // Copy the sharding information if there was any.
  if (to_combine[0]->has_sharding()) {
    new_inst->set_sharding(to_combine[0]->sharding());
  }

  // Replace all the users.
  std::vector<HloInstruction*> result(cluster_size + 1);
  result[0] = new_inst;
  for (uint64 i = 0; i != cluster_size; ++i) {
    HloInstruction* inst = to_combine[i];
    // Add a GTE to unpack the new_inst result.
    auto gte = comp->AddInstruction(
        HloInstruction::CreateGetTupleElement(inst->shape(), new_inst, i));
    // Mark it as inplace.
    MakeUsedInplace(gte);
    result[i + 1] = gte;

    // Replace the old inst.
    TF_RETURN_IF_ERROR(new_inst->CopyAllControlDepsFrom(inst));
    TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
    TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(gte));
    TF_RETURN_IF_ERROR(comp->ForceRemoveInstruction(inst));
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

#define REGISTER_INSTRUCTION_COLLOCATOR_HELPER(colocator)              \
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
    // Make sure the same to_apply() computation is used.
    return *a->to_apply() == *b->to_apply();
  }
};

// Colocator helper which is used to combine multiple inter IPU copies.
class InterIpuCopyColocatorHelper : public InstructionColocatorHelper {
 public:
  InterIpuCopyColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::IpuInterCopy)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_inter_ipu_copies_buffer_size;
  }
};

class ReduceScatterColocatorHelper : public InstructionColocatorHelper {
 public:
  ReduceScatterColocatorHelper() : InstructionColocatorHelper() {}

  bool CanColocate(const HloInstruction* inst) const override {
    return IsPoplarInstruction(PoplarOp::ReduceScatter)(inst);
  }

  int64 GetColocateBufferSize(
      const CompilerInformation& information) const override {
    return information.max_reduce_scatter_buffer_size;
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

    std::vector<Shape> new_shapes;
    std::vector<HloInstruction*> new_operands;
    std::vector<std::string> new_rendezvous_keys;
    for (auto* old_send : old_sends) {
      CHECK_EQ(old_send->operand_count(), 1);
      CHECK_EQ(old_send->RendezvousKeys().size(), 1);
      new_shapes.push_back(old_send->shape());
      new_operands.push_back(old_send->mutable_operand(0));
      new_rendezvous_keys.push_back(old_send->RendezvousKeys()[0]);
    }

    const auto new_shape = ShapeUtil::MakeTupleShape(new_shapes);

    // Add the new instruction.
    HloInstruction* new_send =
        comp->AddInstruction(first_send->CloneWithNewOperandsAndRendezvousKeys(
            new_shape, new_operands, new_rendezvous_keys));

    // Copy the sharding information if there was any.
    if (first_send->has_sharding()) {
      new_send->set_sharding(first_send->sharding());
    }

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

    std::vector<Shape> new_shapes;
    std::vector<HloInstruction*> new_operands;
    std::vector<std::string> new_rendezvous_keys;
    for (auto* old_recv : old_recvs) {
      CHECK_EQ(old_recv->RendezvousKeys().size(), 1);
      new_shapes.push_back(old_recv->shape());
      new_operands.insert(new_operands.end(), old_recv->operands().cbegin(),
                          old_recv->operands().cend());
      new_rendezvous_keys.push_back(old_recv->RendezvousKeys()[0]);
    }

    const auto new_shape = ShapeUtil::MakeTupleShape(new_shapes);

    // Add the new instruction.
    HloInstruction* new_recv =
        comp->AddInstruction(first_recv->CloneWithNewOperandsAndRendezvousKeys(
            new_shape, new_operands, new_rendezvous_keys));

    // Copy the sharding information if there was any.
    if (first_recv->has_sharding()) {
      new_recv->set_sharding(first_recv->sharding());
    }

    std::vector<HloInstruction*> result;
    result.push_back(new_recv);

    // Replace all the users.
    for (uint64 i = 0; i != old_recvs.size(); ++i) {
      HloInstruction* old_recv = old_recvs[i];
      // Add a GTE to unpack the new_recv result.
      auto* gte = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          old_recv->shape(), new_recv, i));
      MakeUsedInplace(gte);
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

    HloComputation* comp = to_combine[0]->parent();
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
    operands[2 * cluster_size] = to_combine[0]->mutable_operand(2);

    auto shape = ShapeUtil::MakeTupleShape(output_shapes);

    // Add the new instruction.
    HloInstruction* new_inst = comp->AddInstruction(
        to_combine[0]->CloneWithNewOperands(shape, operands));
    // Copy the sharding information if there was any.
    if (to_combine[0]->has_sharding()) {
      new_inst->set_sharding(to_combine[0]->sharding());
    }

    // Replace all the users.
    std::vector<HloInstruction*> result(3 * cluster_size + 1);
    result[0] = new_inst;
    for (uint64 i = 0; i != cluster_size; ++i) {
      HloInstruction* inst = to_combine[i];
      // GTE to get the new accumulator.
      auto gte_accum = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(output_shapes[i], new_inst, i));
      MakeUsedInplace(gte_accum);
      result[i + 1] = gte_accum;

      // GTE to get the new gradient.
      auto gte_grad =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              output_shapes[cluster_size + i], new_inst, cluster_size + i));
      MakeUsedInplace(gte_grad);
      result[cluster_size + i + 1] = gte_grad;

      // Create a tuple instruction so that we can easily replace all the users.
      auto tuple = comp->AddInstruction(
          HloInstruction::CreateTuple({gte_accum, gte_grad}));
      MakeUsedInplace(tuple);
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

}  // namespace

REGISTER_INSTRUCTION_COLLOCATOR_HELPER(InterIpuCopyColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(AllReduceColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(ReduceScatterColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(
    StatefulGradientAccumulationAllReduceColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(
    StatefulGradientAccumulateWithMomentumAndAllReduceWithNormColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(SendToHostColocatorHelper)
REGISTER_INSTRUCTION_COLLOCATOR_HELPER(RecvFromHostColocatorHelper)

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

#undef REGISTER_INSTRUCTION_COLLOCATOR_HELPER
}  // namespace poplarplugin
}  // namespace xla
