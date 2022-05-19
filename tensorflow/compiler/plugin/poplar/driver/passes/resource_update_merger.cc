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

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "google/protobuf/util/message_differencer.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/resource_update_merger.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
namespace xla {
namespace poplarplugin {
namespace {

StatusOr<std::vector<std::vector<HloInstruction*>>> FilterResourceUpdates(
    const HloModule* const module) {
  std::vector<std::vector<HloInstruction*>> resource_updates;
  for (auto comp : module->MakeComputationPostOrder()) {
    std::vector<HloInstruction*> comp_insts;
    for (auto inst : comp->MakeInstructionPostOrder()) {
      if (IsResourceUpdate(inst)) {
        comp_insts.push_back(inst);
        TF_RETURN_IF_ERROR(ConvertAllUsersToGTEs(inst).status());
      }
    }
    resource_updates.emplace_back(std::move(comp_insts));
  }
  return resource_updates;
}

Status VerifyResourceUpdateUsers(const std::vector<HloInstruction*> insts) {
  // If we have a ResourceUpdate that is used by a GTE or another
  // ResourceUpdate, then that ResourceUpdate can be merged.
  // If it's user is neither, then the user must be a root instruction.
  for (auto* inst : insts) {
    auto* root = inst->parent()->root_instruction();

    std::function<Status(HloInstruction*)> verify_user;
    verify_user = [&verify_user, root](HloInstruction* user) -> Status {
      const auto is_root = root == user;
      const auto is_gte = user->opcode() == HloOpcode::kGetTupleElement;
      const auto is_ru = IsResourceUpdate(user);

      const auto is_single_element_tuple =
          user->opcode() == HloOpcode::kTuple &&
          ShapeUtil::TupleElementCount(user->shape()) == 1;

      if (!is_root && !is_gte && !is_ru && !is_single_element_tuple) {
        return InternalError(
            "Cannot merge ResourceUpdate with a user that is not a "
            "GetTupleElement, Tuple or ResourceUpdate that is not also the "
            "root instruction.");
      }

      if (is_gte) {
        for (auto u : user->users()) {
          TF_RETURN_IF_ERROR(verify_user(u));
        }
      }

      return Status::OK();
    };

    for (auto user : inst->users()) {
      TF_RETURN_IF_ERROR(verify_user(user));
    }
  }
  return Status::OK();
}

Status DeduplicateGradientAccumulationCount(HloInstruction* merged_ru) {
  // For each previous (now merged) RU.
  HloInstruction* grad_accum_inst = nullptr;
  auto merged_ru_comp = merged_ru->to_apply();
  absl::flat_hash_map<HloComputation*,
                      std::pair<HloInstruction*, HloInstruction*>>
      to_remove;
  for (auto inst : merged_ru_comp->MakeInstructionPostOrder()) {
    if (inst->opcode() != HloOpcode::kCall) {
      continue;
    }

    // Find it's GradientAccumulationCount instruction.
    auto call_comp = inst->to_apply();

    auto instructions = call_comp->MakeInstructionPostOrder();
    auto call_comp_grad_accum_inst =
        absl::c_find_if(instructions, [](HloInstruction* i) -> bool {
          return IsPoplarInstruction(PoplarOp::GradientAccumulationCount)(i);
        });

    // If it doesn't have one.
    if (call_comp_grad_accum_inst == instructions.end()) {
      return InternalError(
          "Merged ResourceUpdate sub computation does not "
          "have a GradientAccumulationCount instruction.");
    }

    auto* gac_val_inst = (*call_comp_grad_accum_inst)->mutable_operand(0);

    if (grad_accum_inst == nullptr) {
      // If this is the first we have found, then clone to add to the new
      // merged RU.
      auto shape = (*call_comp_grad_accum_inst)->shape();

      auto* new_gac_val_inst =
          merged_ru_comp->AddInstruction(gac_val_inst->Clone());

      grad_accum_inst = merged_ru_comp->AddInstruction(
          (*call_comp_grad_accum_inst)
              ->CloneWithNewOperands(shape, {new_gac_val_inst}));
    }

    to_remove[call_comp] =
        std::make_pair(gac_val_inst, *call_comp_grad_accum_inst);
  }

  // Remove.
  for (auto r : to_remove) {
    HloComputation* comp;
    std::pair<HloInstruction*, HloInstruction*> insts;
    std::tie(comp, insts) = r;

    TF_RETURN_IF_ERROR(comp->RemoveInstruction(insts.second));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(insts.first));
  }

  return Status::OK();
}

Status CreateReplacementInstruction(
    HloComputation::Builder& comp_builder, HloInstruction* ru_call,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& new_insts,
    std::vector<HloInstruction*>& new_ru_operands) {
  // original_ru_call will be replaced with a non-RU call in the new
  // RU's computation that uses the original calls computation.
  auto* ru_comp = ru_call->to_apply();

  // If we have any operands to original_ru_call that aren't RU's
  // then they will need to be provided to the merged RU as params
  // to be passed on to the new non-RU call within the new RU.
  // If however, the operand is another RU then we substitute
  // the operand directly as it's replacement non-RU call will also
  // exist within the new RU computation and should have been
  // created already.
  using Sig = std::vector<HloInstruction*>(HloInstruction*,
                                           std::vector<HloInstruction*>);

  std::function<Sig> get_operand_chain;
  get_operand_chain =
      [&get_operand_chain, &new_ru_operands, &comp_builder, &new_insts](
          HloInstruction* inst, std::vector<HloInstruction*> operand_chain = {})
      -> std::vector<HloInstruction*> {
    // Add the operand to the chain, if it's a GTE then we recurse. If it's
    // an RU then we have a complete chain.
    auto* prev = !operand_chain.empty() ? operand_chain.back() : nullptr;
    operand_chain.push_back(inst);

    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      auto* gte_op = inst->mutable_operand(0);
      if (gte_op->opcode() == HloOpcode::kGetTupleElement ||
          IsResourceUpdate(gte_op)) {
        return get_operand_chain(gte_op, operand_chain);
      }
    }

    if (IsResourceUpdate(inst)) {
      return operand_chain;
    }

    if (inst->opcode() == HloOpcode::kTuple) {
      if (prev != nullptr && prev->opcode() == HloOpcode::kGetTupleElement) {
        operand_chain.pop_back();
        inst = prev;
      } else {
        return get_operand_chain(inst->mutable_operand(0), operand_chain);
      }
    }

    // If it's not an RU, create a param.
    auto* param = comp_builder.AddInstruction(HloInstruction::CreateParameter(
        new_ru_operands.size(), inst->shape(), inst->name() + "_param"));
    operand_chain.back() = param;
    new_insts[inst] = param;

    // Store the original operand as it is needed as an operand to the
    // merged RU.
    new_ru_operands.push_back(inst);
    return operand_chain;
  };

  for (auto* operand : ru_call->operands()) {
    auto op_chain = get_operand_chain(operand, {});

    for (auto op_itr = op_chain.rbegin(); op_itr != op_chain.rend(); ++op_itr) {
      auto op = *op_itr;

      // Params will have been added by get_operand_chain, so we can skip.
      if (op->opcode() == HloOpcode::kParameter) {
        continue;
      }

      if (op->opcode() == HloOpcode::kGetTupleElement) {
        // If we already have a replacement for op, then skip.
        if (new_insts.find(op) != new_insts.end()) {
          continue;
        }

        auto* prev_op = *(op_itr - 1);
        HloInstruction* prev_replacement =
            prev_op->opcode() != HloOpcode::kParameter ? new_insts[prev_op]
                                                       : prev_op;

        auto replacement = HloInstruction::CreateGetTupleElement(
            op->shape(), prev_replacement, op->tuple_index());

        auto name = absl::StrFormat("%s_replacement", op->name());
        replacement->SetAndSanitizeName(name);
        new_insts[op] = comp_builder.AddInstruction(std::move(replacement));
        continue;
      }

      // Look up replacement non-RU call and set it's operands.
      if (IsResourceUpdate(op)) {
        auto* replacement_call = new_insts[op];

        for (int64_t idx = 0; idx < op->operands().size(); idx++) {
          auto* ru_op = op->operands()[idx];
          // Find replacement operand.
          if (new_insts.find(ru_op) == new_insts.end()) {
            return InternalError(
                "No replacement operand for merged ResourceUpdate "
                "call instruction found.");
          }

          auto ru_op_replacement = new_insts.find(ru_op);
          TF_RETURN_IF_ERROR(replacement_call->ReplaceOperandWith(
              idx, ru_op_replacement->second));
        }
        continue;
      }

      // We have an operand that doesn't have a replacement
      // parameter, GTE, Tuple or RU.
      return InternalError("Found operand for which there is no replacement.");
    }
  }

  // Create the new, non-RU call and add it to the new RU's computation.
  std::vector<HloInstruction*> new_operands;
  auto old_operands = ru_call->operands();
  std::transform(old_operands.begin(), old_operands.end(),
                 std::back_inserter(new_operands),
                 [&new_insts](HloInstruction* op) { return new_insts[op]; });
  auto new_call_inst = HloInstruction::CreateCall(
      ru_comp->root_instruction()->shape(), new_operands, ru_comp);

  new_call_inst->SetAndSanitizeName(ru_call->name() + "_non_ru_call");

  auto* new_call = comp_builder.AddInstruction(std::move(new_call_inst));
  new_insts[ru_call] = new_call;

  return Status::OK();
}

StatusOr<std::pair<std::unique_ptr<HloInstruction>,
                   std::vector<std::tuple<int64_t, int64_t, HloInstruction*>>>>
GetNewCompRoot(
    const std::vector<HloInstruction*>& ru_insts,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& inst_map,
    HloComputation::Builder& comp_builder) {
  // For each ResourceUpdate, we need to map the outputs that go to the
  // original computations root to their counterparts in the new, merged
  // computation if that RU's output is not the input of another RU.
  std::vector<HloInstruction*> out_inst;
  std::vector<std::tuple<int64_t, int64_t, HloInstruction*>> replacements;
  for (auto* ru : ru_insts) {
    auto* root = ru->parent()->root_instruction();

    std::function<HloInstruction*(HloInstruction*, HloInstruction*)>
        find_op_to_replace;
    find_op_to_replace = [&find_op_to_replace, ru](
                             HloInstruction* inst,
                             HloInstruction* operand) -> HloInstruction* {
      if (operand == ru) {
        return inst;
      }

      if (operand->opcode() == HloOpcode::kGetTupleElement) {
        return find_op_to_replace(operand, operand->mutable_operand(0));
      }

      if (operand->opcode() == HloOpcode::kTuple) {
        for (auto* op : operand->operands()) {
          auto* op_op = find_op_to_replace(operand, op);
          if (op_op != nullptr) {
            return op_op;
          }
        }
      }

      return nullptr;
    };

    // Find which operand of the root tuple is the output of ru.
    std::vector<std::tuple<int64_t, int64_t, HloInstruction*>> used_in;
    for (int64_t idx = 0; idx < root->operands().size(); idx++) {
      auto* operand = root->mutable_operand(idx);
      auto* to_replace = find_op_to_replace(root, operand);
      if (to_replace != nullptr) {
        used_in.emplace_back(idx, to_replace->tuple_index(), to_replace);
      }
    }

    if (used_in.empty()) {
      continue;
    }

    // Now lookup it's counterpart in the new, merged computation.
    auto new_out = inst_map.find(ru);
    if (new_out == inst_map.end()) {
      return InternalError(
          "Merged ResourceUpdate replacement instruction not found.");
    }

    // Pull out the required element of each non-RU call output.
    for (auto u : used_in) {
      int64_t root_idx, tuple_idx;
      HloInstruction* to_replace;
      std::tie(root_idx, tuple_idx, to_replace) = u;

      auto* call_inst = new_out->second;

      // Create a GTE for this output.
      const auto shape = to_replace->shape();
      auto gte_inst =
          HloInstruction::CreateGetTupleElement(shape, call_inst, tuple_idx);

      auto name = absl::StrFormat("%s_%s_output_%d", gte_inst->name(),
                                  call_inst->name(), tuple_idx);
      gte_inst->SetAndSanitizeName(name);

      auto* gte = comp_builder.AddInstruction(std::move(gte_inst));

      // Push back.
      out_inst.push_back(gte);
      replacements.emplace_back(
          root_idx, static_cast<int64_t>(out_inst.size()) - 1, to_replace);
    }
  }

  // For each found counterpart instruction in the new computation,
  // an entry into a tuple is made. This tuple becomes the root of the
  // new computation. Additionally, we need to keep track of which index
  // in this tuple corresponds to which in the original root instruction
  // such that it can be replaced with a GTE over output of the merged
  // ResourceUpdate computations call.
  auto new_root = HloInstruction::CreateTuple(out_inst);
  new_root->SetAndSanitizeName(new_root->name() + "_ru_root");
  return std::make_pair(std::move(new_root), replacements);
}

Status ReplaceRootOperands(
    HloComputation* comp, HloInstruction* merged_ru,
    HloInstruction* merged_comp_root,
    std::vector<std::tuple<int64_t, int64_t, HloInstruction*>>& replacements) {
  auto* root = comp->root_instruction();

  for (auto r : replacements) {
    int64_t root_idx, ru_out_idx;
    HloInstruction* rep_inst;
    std::tie(root_idx, ru_out_idx, rep_inst) = r;

    TF_RETURN_IF_ERROR(
        rep_inst->ReplaceOperandWithDifferentShape(0, merged_ru));
    rep_inst->set_tuple_index(ru_out_idx);

    auto name =
        absl::StrFormat("%s_root_operand_%d", rep_inst->name(), root_idx);
    rep_inst->SetAndSanitizeName(name);
  }

  return Status::OK();
}

StatusOr<
    std::pair<xla::FrontendAttributes, xla::poplarplugin::PoplarBackendConfig>>
GetFrontendAttributesBackendConfig(const std::vector<HloInstruction*>& insts) {
  auto frontend_attrs = insts.front()->frontend_attributes();
  xla::poplarplugin::PoplarBackendConfig backend_config;
  TF_ASSIGN_OR_RETURN(
      backend_config,
      insts.front()->backend_config<decltype(backend_config)>());

  for (auto* inst : insts) {
    // Check frontend attributes.
    if (!google::protobuf::util::MessageDifferencer::Equals(
            frontend_attrs, inst->frontend_attributes())) {
      return InternalError(
          "ResourceUpdates within computation have inconsistent"
          " frontend attributes.");
    }

    // Check backend config.
    xla::poplarplugin::PoplarBackendConfig cfg;
    TF_ASSIGN_OR_RETURN(cfg, inst->backend_config<decltype(cfg)>());
    if (!google::protobuf::util::MessageDifferencer::Equals(cfg,
                                                            backend_config)) {
      return InternalError(
          "ResourceUpdates within computation have inconsistent"
          " backend configurations.");
    }
  }

  return std::make_pair(frontend_attrs, backend_config);
}

Status MergeResourceUpdates(HloModule* module, HloComputation* comp,
                            const std::vector<HloInstruction*>& insts) {
  TF_RETURN_IF_ERROR(VerifyResourceUpdateUsers(insts));

  // Pull out frontend attributres to use for the merged ResourceUpdate.
  xla::FrontendAttributes frontend_attrs;
  xla::poplarplugin::PoplarBackendConfig backend_cfg;
  TF_ASSIGN_OR_RETURN(std::tie(frontend_attrs, backend_cfg),
                      GetFrontendAttributesBackendConfig(insts));

  // Create a new call target for the resultant *single* resource update.
  auto comp_builder = HloComputation::Builder("merged_resource_update");

  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacement_insts;
  std::vector<HloInstruction*> new_ru_operands;
  for (auto* inst : insts) {
    TF_RETURN_IF_ERROR(CreateReplacementInstruction(
        comp_builder, inst, replacement_insts, new_ru_operands));
  }

  // Get the root of the new computation. This should be a tuple of each
  // ResourceUpdate output that goes to the callers root. Also, get a
  // list of indices to which outputs should be mapped.
  std::unique_ptr<HloInstruction> merged_root;
  std::vector<std::tuple<int64_t, int64_t, HloInstruction*>> root_replacements;
  TF_ASSIGN_OR_RETURN(std::tie(merged_root, root_replacements),
                      GetNewCompRoot(insts, replacement_insts, comp_builder));
  auto* new_root = comp_builder.AddInstruction(std::move(merged_root));

  // Build the new computation and add to the module.
  auto* new_to_apply =
      module->AddEmbeddedComputation(comp_builder.Build(new_root));

  // Create a new ResourceUpdate instruction to launch the new,
  // merged computation and add to the computation.
  auto* new_ru = comp->AddInstruction(HloInstruction::CreateCall(
      new_root->shape(), new_ru_operands, new_to_apply));
  insts.front()->SetupDerivedInstruction(new_ru);

  new_ru->SetAndSanitizeName("merged_resource_update_call");

  // Make new_ru a ResourceUpdate.
  new_ru->set_frontend_attributes(frontend_attrs);
  new_ru->set_backend_config(backend_cfg);

  // Map the outputs of the new ResourceUpdate computation to the output tuple
  // of the original computation.
  TF_RETURN_IF_ERROR(
      ReplaceRootOperands(comp, new_ru, new_root, root_replacements));

  // Finally, deduplicate GradientAccumulationCount instructions.
  TF_RETURN_IF_ERROR(DeduplicateGradientAccumulationCount(new_ru));

  return Status::OK();
}

Status RemoveOldResourceUpdates(const std::vector<HloInstruction*>& insts,
                                HloComputation* comp) {
  std::function<Status(HloInstruction*)> remove_inst;
  remove_inst = [&remove_inst, comp](HloInstruction* inst) {
    if (inst == comp->root_instruction()) {
      return Status::OK();
    }

    for (auto* u : inst->users()) {
      TF_RETURN_IF_ERROR(remove_inst(u));
    }

    if (inst->users().empty()) {
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));
    }

    return Status::OK();
  };

  for (auto iter = insts.rbegin(); iter != insts.rend(); ++iter) {
    TF_RETURN_IF_ERROR(remove_inst(*iter));
  }
  return Status::OK();
}

}  // namespace

StatusOr<bool> ResourceUpdateMerger::Run(HloModule* module) {
  VLOG(2) << "Before ResourceUpdateMerger:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  TF_ASSIGN_OR_RETURN(auto resource_updates, FilterResourceUpdates(module));
  for (const auto& instructions : resource_updates) {
    if (instructions.size() <= 1) {
      continue;
    }

    auto comp = instructions.front()->parent();
    TF_RETURN_IF_ERROR(FixRootInstruction(comp).status());
    TF_RETURN_IF_ERROR(MergeResourceUpdates(module, comp, instructions));
    TF_RETURN_IF_ERROR(RemoveOldResourceUpdates(instructions, comp));

    changed = true;
  }

  if (changed) {
    VLOG(2) << "After ResourceUpdateMerger:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No ResourceUpdate operations were merged.";
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
