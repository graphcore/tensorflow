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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_base.h"

#include <stddef.h>

#include <map>
#include <poplar/CSRFunctions.hpp>
#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/Util.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"

using tensorflow::str_util::StartsWith;

namespace xla {
namespace poplarplugin {

typedef StatusOr<poplar::program::Program> (*CustomCallFn)(
    CompilerResources&, const HloInstruction*, const xla::Shape&, TensorMap&);

Status BaseVisitor::Preprocess(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  bool new_stochastic_rounding_enabled;
  switch (poplar_backend_config.stochastic_rounding()) {
    case THREESTATE_OFF:
      new_stochastic_rounding_enabled = false;
      break;
    case THREESTATE_ON:
      new_stochastic_rounding_enabled = true;
      break;
    // The stochastic_rounding mode should be unambiguously set by
    // the AddStochasticRoundingOptions pass.
    case THREESTATE_UNDEFINED:
    default:
      return InvalidArgumentStrCat(
          "Invalid value for PoplarBackendConfig.stochastic_rounding() in "
          "instruction '",
          inst->name(), "'");
  }
  if (new_stochastic_rounding_enabled !=
      resources_.stochastic_rounding_enabled) {
    poplar::program::Sequence seq({}, debug_name_and_id);
    poplar::setStochasticRounding(GetGraph(resources_, inst), seq,
                                  new_stochastic_rounding_enabled,
                                  {debug_name_and_id, "Preprocess"});
    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    resources_.stochastic_rounding_enabled = new_stochastic_rounding_enabled;
  }

  if (allow_seed_changes_) {
    const auto new_sr_method =
        poplar_backend_config.stochastic_rounding_method();
    poplar::program::Sequence seq({}, debug_name_and_id);

    if (MaybeChangeStochasticRoundingMethod(resources_, inst->name(),
                                            new_sr_method, seq)) {
      TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, seq));
    }
  }

  return Status::OK();
}

BaseVisitor::BaseVisitor(CompilerResources& resources,
                         const poplar::DebugNameAndId& debug_name_and_id)
    : resources_(resources),
      dnai_(debug_name_and_id),
      execution_counters_(resources, debug_name_and_id),
      allow_seed_changes_(resources.enable_experimental_prng_stability) {
  // Push the execution counters onto the stack.
  resources_.execution_counter_scopes.push(&execution_counters_);
}

const Shape& BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status BaseVisitor::HandleHloOp(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->ToString();
  poplar::Graph& graph = GetGraph(resources_, inst);

  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateHloOp(graph, resources_, inst, GetOutputShape(inst), tensor_map));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateCastOp(resources_, inst, GetOutputShape(inst),
                                   tensor_map, debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleTupleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateTupleSelectOp(resources_, inst, GetOutputShape(inst), tensor_map,
                          debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleBitcastConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::program::Sequence seq({}, debug_name_and_id);

  TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                      FindInplaceOutputTensors(tensor_map, resources_, inst,
                                               seq, debug_name_and_id));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(inst->shape()));
  out = out.reinterpret(type);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::HandleAllReduce(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);

  popops::CollectiveOperator op = popops::CollectiveOperator::LOCAL;

  if (IsAllReduceAdd(inst)) {
    op = popops::CollectiveOperator::ADD;
  } else if (IsAllReduceMean(inst)) {
    op = popops::CollectiveOperator::MEAN;
  } else {
    return xla::FailedPrecondition(
        "Unsupported all-reduce reduction computation.");
  }

  TF_ASSIGN_OR_RETURN(auto seq, CreateReplicatedAllReduce(
                                    resources_, inst, GetOutputShape(inst),
                                    tensor_map, op, debug_name_and_id));

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::HandleConstant(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::DebugContext debug_context(debug_name_and_id);
  PoplarOpDefDebugInfo debug_info(debug_context, "HandleConstant");

  poplar::Graph& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor t,
      AddConstantTensor(graph, TensorLocation{inst, 0}, GetOutputShape(inst),
                        inst->literal(), resources_, tensor_map, {debug_info}));

  // If this constant is used inplace then we need to add a copy and use that
  // instead so the original constant value is always preserved.
  bool is_inplace_read_write = IsOutputModifiedInplace(inst);
  if (is_inplace_read_write && t.numElements() != 0) {
    VLOG(1) << "Constant tensor is read/write inplace, adding copy";
    poplar::program::Sequence prog({}, debug_info);
    poplar::Tensor clone = poputil::duplicate(
        graph, t, prog, {debug_info, "clone"},
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

    TF_RETURN_IF_ERROR(AddSequenceForInstruction(inst, prog));
    t = clone;
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::program::Sequence seq({}, debug_name_and_id);

  TF_ASSIGN_OR_RETURN(TensorVectors output_tensors,
                      FindInplaceOutputTensors(tensor_map, resources_, inst,
                                               seq, debug_name_and_id, false));
  CHECK_EQ(output_tensors.size(), 1);
  CHECK_EQ(output_tensors[0].size(), CountShapes(inst->shape()));
  for (size_t i = 0; i < output_tensors[0].size(); i++) {
    poplar::Tensor out;
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output_tensors[0][i]));
  }

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::HandleFusion(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->ToString();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::program::Sequence seq({}, debug_name_and_id);
  poplar::program::Program prog;
  HloComputation* comp = inst->fused_instructions_computation();

  if (IsPopOpsFusion(inst)) {
    // Fusions are handle as Poplar custom ops.
    poplar::Graph& graph = GetGraph(resources_, inst);
    const bool is_poplar_custom_op = GetPoplarCustomOp(inst).has_value();
    if (is_poplar_custom_op) {
      TF_ASSIGN_OR_RETURN(
          prog, CreatePoplarOp(graph, resources_, inst, GetOutputShape(inst),
                               tensor_map, debug_name_and_id));
    } else {
      return xla::FailedPrecondition("Unrecognised fusion instruction %s.",
                                     inst->ToString().c_str());
    }
  } else {
    TF_ASSIGN_OR_RETURN(
        prog, CreateFusionOp(resources_, inst, GetOutputShape(inst), tensor_map,
                             debug_name_and_id));
  }

  seq.add(prog);
  return AddSequenceForInstruction(inst, seq);
};

Status BaseVisitor::HandleCall(HloInstruction* inst) {
  HloComputation* comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : " << comp->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateCallOp(resources_, inst, GetOutputShape(inst),
                                   tensor_map, debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleCustomCall(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateCustomCallOp(resources_, inst, GetOutputShape(inst),
                                         tensor_map, debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateTuple(resources_, inst, tensor_map, debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleMap(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(bool simple_parallel,
                      IsParallelMap(inst, inst->to_apply()));
  if (simple_parallel) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateParallelMap(resources_, inst, GetOutputShape(inst), tensor_map,
                          debug_name_and_id));
    return AddSequenceForInstruction(inst, prog);
  }
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConditional(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateConditionalOp(resources_, inst, GetOutputShape(inst), tensor_map,
                          debug_name_and_id));
  return AddSequenceForInstruction(inst, prog);
}

Status BaseVisitor::HandleReal(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::program::Sequence seq({}, debug_name_and_id);

  TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                      FindInstructionInput(tensor_map, resources_, inst, 0, seq,
                                           debug_name_and_id));

  poplar::Tensor out = GetGraph(resources_, inst).clone(in);
  seq.add(poplar::program::Copy(in, out, false, {debug_name_and_id}));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::HandleAllToAll(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  TF_ASSIGN_OR_RETURN(
      auto seq, CreateReplicatedAllToAll(resources_, inst, GetOutputShape(inst),
                                         tensor_map, debug_name_and_id));

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::HandleAddDependency(HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  poplar::program::Sequence seq({}, debug_name_and_id);

  std::vector<std::string> dep_names;
  GetAllDepNames(inst->operand(1), dep_names);

  VLOG(1) << "Processing " << inst->name() << " on "
          << absl::StrJoin(dep_names, ",");
  TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                      FindInplaceOutputTensors(tensor_map, resources_, inst,
                                               seq, debug_name_and_id, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), CountShapes(inst->operand(0)->shape()));
  for (size_t idx = 0; idx < inputs[0].size(); idx++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, idx, inputs[0][idx]));
  }

  return AddSequenceForInstruction(inst, seq);
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s (%s) not implemented", inst->name().c_str(),
                            HloOpcodeString(inst->opcode()).c_str());
}

poplar::DebugNameAndId BaseVisitor::GetDebugNameAndId(
    const HloInstruction* inst) const {
  return xla::poplarplugin::GetDebugNameAndId(resources_, inst);
}

Status BaseVisitor::HandleAfterAll(HloInstruction* inst) {
  // TODO(shauryas) : figure out how to use this for something useful
  return Status::OK();
}

Status BaseVisitor::HandleInfeed(HloInstruction* inst) {
  return xla::FailedPrecondition(
      "Unsupported use of infeed operation - it's only supported inside of "
      "loops.");
}

Status BaseVisitor::Postprocess(HloInstruction* inst) {
  // Update the progress bar.
  resources_.progress_bar->Update(inst);
  return Status::OK();
}

Status BaseVisitor::FinishVisit(HloInstruction* root) {
  resources_.execution_counter_scopes.pop();
  return FinishScopedVisit(root);
}

ExecutionCounters& BaseVisitor::GetExecutionCounters() {
  return execution_counters_;
}

Status BaseVisitor::AddSequenceForInstruction(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  if (grouped_sequence_indices_.find(inst) != grouped_sequence_indices_.end()) {
    return FailedPrecondition("Already started grouping sequences for %s",
                              inst->ToString().c_str());
  }

  sequences_.push_back({seq});
  return Status::OK();
}

Status BaseVisitor::CreateSequenceGroupedByInstruction(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  const auto new_index = sequences_.size();
  sequences_.push_back({seq});
  CHECK(grouped_sequence_indices_.emplace(inst, new_index).second);
  return Status::OK();
}

Status BaseVisitor::AppendSequenceGroupedByInstruction(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  // If we have seen this instruction before, add to its existing sequence.
  auto found = grouped_sequence_indices_.find(inst);
  if (found != grouped_sequence_indices_.end()) {
    const auto index = found->second;
    CHECK_LT(index, sequences_.size());
    sequences_[index].push_back(seq);
    return Status::OK();
  }

  // Otherwise add a new one and record its index.
  return CreateSequenceGroupedByInstruction(inst, seq);
}

Status BaseVisitor::PrependSequenceGroupedByInstruction(
    const HloInstruction* inst, const poplar::program::Sequence& seq) {
  // If we have seen this instruction before, add to its existing sequence.
  auto found = grouped_sequence_indices_.find(inst);
  if (found != grouped_sequence_indices_.end()) {
    const auto index = found->second;
    CHECK_LT(index, sequences_.size());
    sequences_[index].push_front(seq);
    return Status::OK();
  }

  // Otherwise add a new one and record its index.
  return CreateSequenceGroupedByInstruction(inst, seq);
}

poplar::program::Sequence BaseVisitor::GetRawSequence() const {
  poplar::program::Sequence result;
  for (const auto& per_instruction_sequences : sequences_) {
    for (const auto& s : per_instruction_sequences) {
      result.add(s);
    }
  }
  return result;
}

poplar::program::Sequence BaseVisitor::GetSequence(
    bool copy_execution_counters) {
  if (copy_execution_counters) {
    poplar::program::Sequence seq({}, dnai_);
    TF_CHECK_OK(
        CopyExecutionCountersFromScope(resources_, execution_counters_, seq));
    seq.add(GetRawSequence());
    return seq;
  } else {
    CHECK(execution_counters_.Initialized());
    return GetRawSequence();
  }
}

}  // namespace poplarplugin
}  // namespace xla
