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

#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {

bool CompatibleShapes(const Shape& l, const Shape& r) {
  // Normal tensors are always acceptable for transferring sharding info
  if (l.IsArray() && r.IsArray()) return true;

  // Tuples must have the same number of components
  auto l_shapes = l.tuple_shapes();
  auto r_shapes = l.tuple_shapes();
  if (l_shapes.size() == r_shapes.size()) {
    return absl::c_equal(
        l_shapes, r_shapes,
        [](const Shape& l, const Shape& r) { return CompatibleShapes(l, r); });
  }

  return false;
}

HloSharding GetDefaultSharding(const Shape& shape) {
  return HloSharding::Single(shape, HloSharding::AssignDevice(0));
}

// Check whether a particular instruction/operand pair has sharding information
// available
bool HasShardingForOperand(const HloInstruction* inst, int operand) {
  const HloInstruction* sharding_inst;
  switch (inst->opcode()) {
    case HloOpcode::kCall: {
      // A call op takes its input sharding from the subcomputation parameter
      auto* comp = inst->to_apply();
      sharding_inst = comp->parameter_instruction(operand);
      break;
    }
    case HloOpcode::kWhile: {
      // A while op takes its input sharding from the subcomputation parameter
      auto* comp = inst->while_body();
      sharding_inst = comp->parameter_instruction(operand);
      break;
    }
    case HloOpcode::kConditional: {
      // A conditional op takes its input sharding from the subcomputation
      // parameter, except for the control parameter (0) which needs to take its
      // sharding from its operand because there is no other place to store it.
      if (operand == 0) {
        sharding_inst = inst->operand(0);
      } else {
        auto* comp = inst->branch_computation(operand - 1);
        sharding_inst = comp->parameter_instruction(0);
      }
      break;
    }
    case HloOpcode::kGetTupleElement: {
      // A GTE must be processed in collection with other GTEs, so claim that
      // there is no sharding information available on its input. See the fn
      // CopyShardingFromUsers.
      return false;
    }
    default: {
      // All other ops hold their own input and output sharding information. In
      // most cases the input sharding and output sharding are identical (for
      // instance an elementwise op takes both inputs from the same IPU as the
      // output is on, and all are single tensors).
      sharding_inst = inst;
      break;
    }
  }
  return sharding_inst->has_sharding();
}

// Set the sharding based on opcode type.  Most operations have only single
// sharding.  Call, While, Conditional, Tuple and GetTupleElement ops can have
// tuple-type sharding.  If a single-type op is passed Tuple-like sharding then
// it will go to the device which is the most used in the tuple.
void SetSharding(HloInstruction* inst, const HloSharding& sharding) {
  if (IsAllowedTupleSharding(inst)) {
    inst->set_sharding(sharding);
  } else {
    if (sharding.IsTuple()) {
      inst->set_sharding(sharding.tuple_elements()[0]);
    } else {
      inst->set_sharding(sharding);
    }
  }
}

HloSharding ConvertToTupleSharding(const Shape& shape,
                                   const std::vector<HloSharding>& shardings) {
  std::vector<HloSharding> all_leaves;
  for (auto& s : shardings) {
    std::vector<HloSharding> leaves;
    if (s.IsTuple()) {
      leaves = s.tuple_elements();
    } else {
      leaves.push_back(s);
    }
    absl::c_copy(leaves, std::back_inserter(all_leaves));
  }

  if (all_leaves.empty()) {
    // Tuple sharding always requires at least one element.
    all_leaves.push_back(HloSharding::AssignDevice(0));
  }

  return HloSharding::Tuple(shape, all_leaves);
}

// Pass in a vector of shardings (tuple or otherwise) and this creates a tuple
// of those inputs, and applies to the instruction.
void SetTupleShardingFromVector(HloInstruction* inst,
                                const std::vector<HloSharding>& shardings) {
  SetSharding(inst, ConvertToTupleSharding(inst->shape(), shardings));
}

bool CopyShardingFromUsers(HloInstruction* inst) {
  if (inst->user_count() == 0) {
    return false;
  }

  // If any user's operand input is available then copy the sharding
  for (auto* u : inst->users()) {
    for (int index = 0; index < u->operand_count(); index++) {
      if (u->operand(index) == inst) {
        if (HasShardingForOperand(u, index)) {
          auto sharding = GetShardingForOperand(u, index);
          SetSharding(inst, sharding);
          return true;
        }
      }
    }
  }

  if (!inst->shape().IsTuple()) {
    return false;
  }

  // Otherwise we need to find the consumer of each element that makes
  // up the tuple. A tuple may have some of its elements unused.
  const int tuple_size = ShapeUtil::TupleElementCount(inst->shape());
  std::vector<HloInstruction*> tuple_users(tuple_size);
  bool has_gte_user = false;
  for (auto* u : inst->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement) {
      if (u->tuple_index() < tuple_size) {
        tuple_users[u->tuple_index()] = u;
        has_gte_user = true;
      }
    }
  }
  // No GTE users found - no sharding to use.
  if (!has_gte_user) {
    return false;
  }

  std::vector<HloSharding> tuple_sharding;
  for (int tuple_index = 0; tuple_index < tuple_size; ++tuple_index) {
    auto* user = tuple_users[tuple_index];
    if (user == nullptr) {
      // Unused tuple outputs are just assigned a default sharding
      auto s = GetDefaultSharding(
          ShapeUtil::GetTupleElementShape(inst->shape(), tuple_index));
      tuple_sharding.push_back(s);
    } else {
      if (user->has_sharding()) {
        tuple_sharding.push_back(GetShardingForOperand(user, 0));
      } else {
        return false;
      }
    }
  }

  // See HloSharding::RequiredLeaves, empty Tuples need one sharding entry
  // and since they don't have any actual tensors associated with them, it
  // doesn't matter which shard they are on.
  if (tuple_size == 0) {
    auto s = GetDefaultSharding(inst->shape());
    tuple_sharding.push_back(s);
  }

  SetTupleShardingFromVector(inst, tuple_sharding);
  return true;
}

bool CopyGteShardingFromOperand(HloInstruction* inst) {
  auto* operand = inst->operand(0);
  if (operand->has_sharding()) {
    int64 tuple_index = inst->tuple_index();
    auto s = GetShardingOfOutputTensor(operand);
    if (!s.IsTuple()) {
      s = HloSharding::SingleTuple(operand->shape(), s);
    }
    auto subsharding = s.GetSubSharding(operand->shape(), {tuple_index});
    if (!inst->has_sharding() || inst->sharding() != subsharding) {
      SetSharding(inst, subsharding);
      return true;
    }
  }

  return false;
}

bool CopyTupleShardingFromOperands(HloInstruction* inst) {
  if (absl::c_all_of(inst->operands(),
                     [](HloInstruction* u) { return u->has_sharding(); })) {
    std::vector<HloSharding> shardings;
    absl::c_transform(
        inst->operands(), std::back_inserter(shardings),
        [](HloInstruction* o) { return GetShardingOfOutputTensor(o); });
    SetTupleShardingFromVector(inst, shardings);
    return true;
  }

  return false;
}

bool CopyShardingFromOperands(HloInstruction* inst) {
  for (int o = 0; o < inst->operand_count(); o++) {
    auto* operand = inst->operand(o);
    if (operand->has_sharding()) {
      if (CompatibleShapes(inst->shape(), operand->shape())) {
        auto s = GetShardingOfOutputTensor(operand);
        SetSharding(inst, s);
        return true;
      }
    }
  }
  return false;
}

bool CopyShardingFromCalledSubcomp(HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional: {
      for (auto* c : inst->called_computations()) {
        auto* root = c->root_instruction();
        if (root->has_sharding() && root->shape() == inst->shape()) {
          auto s = GetShardingOfOutputTensor(root);
          SetSharding(inst, s);
          return true;
        }
      }
    }
    default:
      return false;
  }
}

// Copy sharding information from a callsite (a kCall, kWhile or kConditional
// instruction) to the computations which are called by it.
bool CopyShardingToCalledComputations(const CallSite& site,
                                      HloComputation* comp) {
  bool made_progress = false;
  if (site.context() == CallContext::kSequential) {
    auto* caller = site.instruction();

    // Inputs. For call and while operations, the number of parameters in the
    // computation is the same as the number of operands to the caller.  For a
    // conditional, there is one parameter in each computation, corresponding to
    // one operand on the caller (plus one operand which is the selector)
    auto params = caller->operands();
    switch (caller->opcode()) {
      case HloOpcode::kCall:
      case HloOpcode::kWhile: {
        for (unsigned int p = 0; p < params.size(); p++) {
          if (params[p]->has_sharding() &&
              !comp->parameter_instruction(p)->has_sharding()) {
            auto s = params[p]->sharding();
            SetSharding(comp->parameter_instruction(p), s);
            made_progress |= true;
          }
        }
        break;
      }
      case HloOpcode::kConditional: {
        // The first operand on a conditional instruction is the selection
        // operand.  The remainder apply to each of the called computations
        // in order, one each.
        const auto& comps = caller->called_computations();
        for (unsigned int c = 0; c < comps.size(); c++) {
          if (comps[c] == comp && params[c + 1]->has_sharding() &&
              !comp->parameter_instruction(0)->has_sharding()) {
            auto s = params[c + 1]->sharding();
            SetSharding(comp->parameter_instruction(0), s);
            made_progress |= true;
          }
        }
        break;
      }
      default:
        break;
    }

    // Output.  Don't copy sharding when there is a shape mismatch, which occurs
    // because the conditional subcomputation of the While operation has a
    // boolean scalar output, not the same shape as the while operation.
    if (site.instruction()->has_sharding() &&
        !comp->root_instruction()->has_sharding() &&
        comp->root_instruction()->shape() == site.instruction()->shape()) {
      auto s = caller->sharding();
      SetSharding(comp->root_instruction(), s);
      made_progress |= true;
    }
  }

  return made_progress;
}

StatusOr<bool> ProcessComputation(HloComputation* comp, int attempt) {
  bool done = false;
  while (!done) {
    done = true;
    bool made_progress = false;
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      bool added_sharding = false;

      // If an instruction has no operands, and no users but the root Tuple,
      // then assign default sharding
      if (!inst->has_sharding() && inst->operand_count() == 0 &&
          absl::c_all_of(inst->users(), [](const HloInstruction* inst) {
            return inst == inst->parent()->root_instruction() &&
                   inst->opcode() == HloOpcode::kTuple;
          })) {
        SetSharding(inst, GetDefaultSharding(inst->shape()));
        added_sharding = true;
      }

      // Try taking sharding from the called subcomputation
      if (!inst->has_sharding()) {
        added_sharding = CopyShardingFromCalledSubcomp(inst);
      }

      // Try to take sharding from users
      if (!inst->has_sharding()) {
        added_sharding = CopyShardingFromUsers(inst);
      }

      // Try to take sharding from operands
      if (!inst->has_sharding()) {
        switch (inst->opcode()) {
          case HloOpcode::kGetTupleElement:
            added_sharding = CopyGteShardingFromOperand(inst);
            break;
          case HloOpcode::kTuple:
            added_sharding = CopyTupleShardingFromOperands(inst);
            break;
          case HloOpcode::kCall:
          case HloOpcode::kWhile:
          case HloOpcode::kConditional:
            // These are dealt with by the computation level code
            break;
          default:
            added_sharding = CopyShardingFromOperands(inst);
            break;
        }
      }

      made_progress |= added_sharding;

      if (!inst->has_sharding()) {
        done = false;
      }
    }
    if (!done && !made_progress) {
      switch (attempt) {
        case 0:
          return false;
        case 1:
          // If an input passes through the whole computation and cannot assign
          // some of the nodes, then we pick off a non-Tuple nodes and assign it
          // default sharding.  Tuple nodes are not included because they might
          // be mostly ok, but with only one part preventing them from sharding
          // properly.
          for (auto* inst : comp->instructions()) {
            if (!inst->has_sharding() && !inst->shape().IsTuple()) {
              SetSharding(inst, GetDefaultSharding(inst->shape()));
              break;
            }
          }
          return false;
        case 2:
          // Tuples which are passed through are now considered too
          for (auto* inst : comp->instructions()) {
            if (!inst->has_sharding()) {
              SetSharding(inst, GetDefaultSharding(inst->shape()));
              break;
            }
          }
          return false;
        default:
          return false;
      }
    }
  }

  return true;
}

// Given a pipeline `stage` instruction, set sharding for the `inst`, handling
// tuples.
Status SetPipelineStageInstructionSharding(const HloInstruction* stage,
                                           HloInstruction* inst) {
  auto optional_sharding = stage->sharding().ExtractSingleSharding();
  if (!optional_sharding) {
    return FailedPrecondition("Could not extract single sharding.");
  }
  Shape shape = inst->shape();
  // Outfeeds are a special case, where the sharding matches the sharding of
  // tensors which will be outfed (i.e. operand 0).
  if (inst->opcode() == HloOpcode::kOutfeed) {
    shape = inst->operand(0)->shape();
  }
  // For non empty tuples, we set sharding for each leaf node, otherwise we
  // create single sharding.
  const bool tuple_sharding = shape.IsTuple() &&
                              !ShapeUtil::IsEmptyTuple(shape) &&
                              IsAllowedTupleSharding(inst);

  HloSharding sharding =
      tuple_sharding ? HloSharding::SingleTuple(shape, *optional_sharding)
                     : *optional_sharding;

  inst->set_sharding(sharding);
  return Status::OK();
}

StatusOr<absl::flat_hash_set<const HloComputation*>> ProcessPipeline(
    HloInstruction* pipeline_op, CallGraph* call_graph) {
  absl::flat_hash_set<const HloComputation*> computations_in_pipeline;

  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  // Convert forward stage sharding into tuple sharding.
  for (HloInstruction* fwd_stage : stages.forward) {
    // PipelineFixer checks that fwd stages have sharding.
    CHECK(fwd_stage->has_sharding());
    const HloSharding& sharding = fwd_stage->sharding();
    CHECK(sharding.HasUniqueDevice());
    // Turn it into tuple sharding.
    TF_RETURN_IF_ERROR(
        SetPipelineStageInstructionSharding(fwd_stage, fwd_stage));
  }
  // Mark backward stages with matching sharding from the forward stage.
  for (size_t stage_id = 0; stage_id != stages.backward.size(); ++stage_id) {
    HloInstruction* fwd_stage = stages.forward[stage_id];
    HloInstruction* bwd_stage = stages.backward[stage_id];
    TF_RETURN_IF_ERROR(
        SetPipelineStageInstructionSharding(fwd_stage, bwd_stage));
  }
  // For each stage propagate the sharding information to:
  // 1.  all the subcomputations called by the pipeline stage.
  // 2.  all the user GTEs.
  for (auto& stages : {stages.forward, stages.backward}) {
    for (HloInstruction* stage : stages) {
      // First propagate sharding inside.
      // Get all the computations called.
      TF_ASSIGN_OR_RETURN(absl::flat_hash_set<HloComputation*> called_in_stage,
                          GetAllComputationsCalledBy(stage, call_graph));
      for (HloComputation* comp : called_in_stage) {
        for (HloInstruction* inst : comp->instructions()) {
          // Set sharding for each instruction.
          TF_RETURN_IF_ERROR(SetPipelineStageInstructionSharding(stage, inst));
        }
        computations_in_pipeline.insert(comp);
      }
      // Then propagate sharding to users.
      for (HloInstruction* user : stage->users()) {
        CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
        TF_RETURN_IF_ERROR(SetPipelineStageInstructionSharding(stage, user));
      }
    }
  }
  return computations_in_pipeline;
}

}  // namespace

StatusOr<bool> ShardingPass::Run(HloModule* module) {
  VLOG(2) << "Before ShardingPass:";
  XLA_VLOG_LINES(2, module->ToString());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition(
        "Expected the call graph of the module to be flat.");
  }
  absl::flat_hash_set<const HloComputation*> completed;
  // We first fix sharding for pipelining as it is well defined.
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> pipeline_ops,
                      GetPipelines(module));
  if (pipeline_ops.size()) {
    CHECK_EQ(pipeline_ops.size(), 1);
    TF_ASSIGN_OR_RETURN(
        absl::flat_hash_set<const HloComputation*> completed_in_pipeline,
        ProcessPipeline(pipeline_ops[0], call_graph.get()));
    completed.insert(completed_in_pipeline.begin(),
                     completed_in_pipeline.end());
  }
  // We now propagate the sharding for the rest of the module.
  // Remove unsupported sharding, and sharding on Tuple shaped ops.  We remove
  // sharding from ops which are allowed tuple-type sharding because their
  // sharding should follow the ops which they are sources/sinks for. We also
  // remove sharding from all parameter ops (which probably don't have sharding
  // anyway).
  for (auto* comp : module->computations()) {
    if (completed.contains(comp)) {
      continue;
    }
    for (auto* inst : comp->instructions()) {
      if (inst->has_sharding() && !IsAnyPipelineStageOp(inst)) {
        bool remove_sharding = false;

        auto sharding = inst->sharding();
        if (!IsSupportedSharding(sharding)) {
          remove_sharding = true;
        }

        if (IsAllowedTupleSharding(inst)) {
          remove_sharding = true;
        }

        if (inst->opcode() == HloOpcode::kAfterAll) {
          remove_sharding = true;
        }

        if (inst->opcode() == HloOpcode::kParameter) {
          remove_sharding = true;
        }

        if (remove_sharding) {
          inst->clear_sharding();
        }
      }
    }
  }

  if (!HaveSharding(module)) {
    return false;
  }

  std::vector<HloComputation*> comps = module->MakeComputationPostOrder();
  auto comp_count = comps.size();

  int attempt = 0;
  while (completed.size() != comp_count) {
    bool made_progress = false;
    for (auto* comp : comps) {
      if (!completed.contains(comp)) {
        auto call_graph_node = call_graph->GetNode(comp);

        // Fusion computations are not considered for sharding
        if (IsPopOpsFusion(comp)) {
          completed.insert(comp);
          made_progress |= true;
          continue;
        }

        // Only call/while/if type computations can be sharded. map/sort/reduce
        // ones take the sharding of the caller.
        if (call_graph_node.context() != CallContext::kSequential) {
          completed.insert(comp);
          made_progress |= true;
          continue;
        }

        // Computations which are not called from anywhere are ignored, not
        // including the entry computation.
        if (call_graph_node.callers().size() == 0 &&
            comp != module->entry_computation()) {
          completed.insert(comp);
          made_progress |= true;
          continue;
        }

        if (!HaveSharding(comp)) {
          // Defer computation until its caller has sharding
          continue;
        }

        TF_ASSIGN_OR_RETURN(bool done, ProcessComputation(comp, attempt));

        // Patch up GTE sharding.  GTEs should always have the sharding taken
        // from their operand, not their users.  During the initial copying of
        // sharding info, they are allowed to take the sharding of their users
        // in order to propagate sharding upwards through the graph.
        for (auto* inst : comp->MakeInstructionPostOrder()) {
          if (inst->opcode() == HloOpcode::kGetTupleElement) {
            made_progress |= CopyGteShardingFromOperand(inst);
          }
        }

        // For any called subcomputations which are not complete, copy onto
        // them the input and output sharding from one of their caller
        // instructions
        for (auto site : call_graph_node.callsites()) {
          for (auto* c : site.called_computations()) {
            if (completed.count(c) == 0) {
              made_progress |= CopyShardingToCalledComputations(site, c);
            }
          }
        }

        // Abandoned computation due to application of sharding to a deferred
        // subcomputation.
        if (!done) {
          break;
        }

        // Apply sharding to callers of this computation.  Caller nodes reflect
        // the sharding of the called subcomputation.  Ignore mismatching shapes
        // because the while operation 'condition' subgraph has a different
        // shape output to the operation itself.
        for (auto cs : call_graph_node.caller_callsites()) {
          if (cs.instruction()->shape() == comp->root_instruction()->shape()) {
            SetSharding(cs.instruction(), comp->root_instruction()->sharding());
          }
        }

        // Apply parameter sharding to predicate and body of a while
        for (auto cs : call_graph_node.caller_callsites()) {
          auto* caller = cs.instruction();
          if (caller->opcode() == HloOpcode::kWhile) {
            auto comp_params = comp->parameter_instructions();
            for (auto* c : caller->called_computations()) {
              auto c_params = c->parameter_instructions();
              if (c_params.size() != comp_params.size()) {
                return xla::FailedPrecondition(
                    "Unequal parameter count on %s (%d) and %s (%d)",
                    comp->name().c_str(), comp_params.size(), c->name().c_str(),
                    c_params.size());
              }
              for (size_t p = 0; p < c_params.size(); p++) {
                SetSharding(c_params[p], comp_params[p]->sharding());
              }
            }
          }
        }

        // Patch up GTE sharding again.  Changing the parameter sharding can
        // alter the inputs to a GTE.
        for (auto* inst : comp->MakeInstructionPostOrder()) {
          if (inst->opcode() == HloOpcode::kGetTupleElement) {
            CopyGteShardingFromOperand(inst);
          }
        }

        // Note: after this point, only nodes which do not proceed GTE
        // instructions can be modified.

        // Ensure that all conditional subcomps have the same output sharding
        for (auto cs : call_graph_node.caller_callsites()) {
          auto* caller = cs.instruction();
          if (caller->opcode() == HloOpcode::kConditional) {
            auto sharding = comp->root_instruction()->sharding();
            for (auto* c : caller->called_computations()) {
              SetSharding(c->root_instruction(), sharding);
            }
          }
        }

        // Ensure that the root sharding of a while/repeat/pipeline body matches
        // the input.
        for (auto cs : call_graph_node.caller_callsites()) {
          auto* caller = cs.instruction();
          HloComputation* body = nullptr;
          if (caller->opcode() == HloOpcode::kWhile) {
            body = caller->while_body();
          }

          if (IsRepeatLoop(caller) || IsPipelineOp(caller)) {
            body = caller->to_apply();
          }

          if (body == call_graph_node.computation()) {
            std::vector<HloSharding> shardings;
            absl::c_transform(
                body->parameter_instructions(), std::back_inserter(shardings),
                [](HloInstruction* o) { return GetShardingOfOutputTensor(o); });

            SetSharding(body->root_instruction(),
                        ConvertToTupleSharding(
                            body->root_instruction()->shape(), shardings));
          }
        }

        // Ensure that the callers of this computation have the same sharding as
        // its root
        for (auto cs : call_graph_node.caller_callsites()) {
          if (cs.context() == CallContext::kSequential) {
            auto* caller = cs.instruction();
            if (comp->root_instruction()->shape() == caller->shape()) {
              SetSharding(caller, comp->root_instruction()->sharding());
            }
          }
        }

        completed.insert(comp);
        made_progress |= true;
      }
    }

    if (!made_progress) {
      if (attempt < 2) {
        attempt++;
      } else {
        return xla::FailedPrecondition(
            "Could not apply sharding information to the %s module.",
            module->name().c_str());
      }
    } else {
      attempt = 0;
    }
  }
  VLOG(2) << "After ShardingPass:";
  XLA_VLOG_LINES(2, module->ToString());

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
