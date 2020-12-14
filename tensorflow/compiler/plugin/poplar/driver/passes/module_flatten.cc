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

#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

ModuleFlatten::ModuleFlatten(CompilerAnnotations& annotations)
    : annotations_(annotations) {}

void ModuleFlatten::RemoveMapEntry(HloInstruction* inst) {
  auto* original = annotations_.flattened_inst_map_bwd[inst];
  if (original != nullptr) {
    annotations_.flattened_inst_map_fwd[original] = nullptr;
  }
  annotations_.flattened_inst_map_bwd.erase(inst);
}

StatusOr<bool> ModuleFlatten::Run(HloModule* module) {
  annotations_.flattened_module = absl::make_unique<HloModule>(
      absl::StrCat(module->name(), "-flattened"), module->config());

  auto* flattened = annotations_.flattened_module.get();

  HloCloneContext context(flattened, "flattened");
  auto new_entry = module->entry_computation()->Clone("", &context);
  flattened->AddEntryComputation(std::move(new_entry));

  if (flattened == nullptr) {
    return FailedPrecondition("Failed to clone module %s", module->name());
  }

  // Create a bidirectional mapping of instruction to instruction
  for (auto entry : context.cloned_instructions()) {
    auto* orig_inst = const_cast<HloInstruction*>(entry.first);
    auto* flat_inst = entry.second;
    annotations_.flattened_inst_map_fwd[orig_inst] = flat_inst;
    annotations_.flattened_inst_map_bwd[flat_inst] = orig_inst;
  }

  // Convert kWhile and kConditional to kCall
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(flattened);

  if (!call_graph->IsFlattened()) {
    return FailedPrecondition("Expected that the module %s is flattened",
                              flattened->name());
  }

  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& node) {
        auto* comp = node.computation();
        if (!IsPopOpsFusion(comp)) {
          for (auto call_site : node.callsites()) {
            auto* caller = call_site.instruction();
            if (caller->opcode() == HloOpcode::kCall) {
              RemoveMapEntry(caller);
            } else if (caller->opcode() == HloOpcode::kWhile) {
              RemoveMapEntry(caller);
              auto while_comps =
                  caller->while_condition()->MakeEmbeddedComputationsList();
              while_comps.push_back(caller->while_condition());
              for (auto* while_cond : while_comps) {
                for (auto* cond_inst : while_cond->instructions()) {
                  RemoveMapEntry(cond_inst);
                }
              }
              auto* call_op =
                  caller->parent()->AddInstruction(HloInstruction::CreateCall(
                      caller->shape(), caller->operands(),
                      caller->while_body()));
              caller->parent()->ReplaceInstruction(caller, call_op);
            } else if (caller->opcode() == HloOpcode::kConditional) {
              RemoveMapEntry(caller);
              auto* predicate_op = caller->parent()->AddInstruction(
                  HloInstruction::CreateConstant(
                      LiteralUtil::CreateR0<bool>(true)));
              annotations_.flattened_inst_map_bwd[predicate_op] = nullptr;
              auto* chain_op =
                  caller->parent()->AddInstruction(HloInstruction::CreateCall(
                      caller->shape(), {caller->mutable_operand(1)},
                      caller->branch_computation(0)));
              annotations_.flattened_inst_map_bwd[chain_op] = nullptr;
              for (int i = 2; i < caller->operand_count(); i++) {
                auto* call_op =
                    caller->parent()->AddInstruction(HloInstruction::CreateCall(
                        caller->shape(), {caller->mutable_operand(i)},
                        caller->branch_computation(i - 1)));
                chain_op = caller->parent()->AddInstruction(
                    HloInstruction::CreateTernary(
                        caller->shape(), HloOpcode::kSelect, predicate_op,
                        call_op, chain_op));
                annotations_.flattened_inst_map_bwd[chain_op] = nullptr;
              }
              caller->parent()->ReplaceInstruction(caller, chain_op);
            }
          }
        }
        return Status::OK();
      },
      false));

  // Reset call_graph and flatten all kCall sites
  call_graph = CallGraph::Build(flattened);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& node) {
        auto* inlined_comp = node.computation();
        if (!IsPopOpsFusion(inlined_comp) &&
            inlined_comp->parent()->entry_computation() != inlined_comp) {
          CallSite call_site = node.caller_callsites()[0];
          if (call_site.context() == CallContext::kSequential) {
            auto* caller = call_site.instruction();
            if (caller->opcode() == HloOpcode::kCall) {
              caller->DropAllControlDeps();
              RemoveMapEntry(caller);
              TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap map,
                                  CallInliner::Inline(caller));

              CHECK(map.size() ==
                    static_cast<size_t>(inlined_comp->instruction_count()));

              // Remap/replace any instuctions which have been inlined
              for (auto inlined : map) {
                auto* original =
                    annotations_.flattened_inst_map_bwd[inlined.first];
                annotations_.flattened_inst_map_fwd[original] = inlined.second;
                CHECK(annotations_.flattened_inst_map_bwd.erase(
                          inlined.first) == 1);

                if (inlined.first->opcode() != HloOpcode::kParameter) {
                  annotations_.flattened_inst_map_bwd[inlined.second] =
                      original;
                }
              }
            }
          }
        }
        return Status::OK();
      },
      false));

  flattened->RemoveUnusedComputations();

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
