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

#include <algorithm>
#include <poplar/Graph.hpp>
#include <popops/AllTrue.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor_creator.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/repeat_loop_overlap_io_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/repeat_loop_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;
using tensorflow::str_util::StartsWith;

namespace xla {
namespace poplarplugin {
namespace {
class ParallelMapTester : public DfsHloVisitorWithDefault {
 public:
  ParallelMapTester() : _is_ok(true) {}

  Status DefaultAction(HloInstruction* inst) override {
    if (inst->IsElementwise()) {
      return Status::OK();
    } else if (IsPoplibsHloCustomOp(inst) &&
               Cast<HloPoplarInstruction>(inst)->IsPopOpsElementwise()) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kParameter) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kTuple) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kSelect) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kCall) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kMap) {
      return Status::OK();
    } else {
      VLOG(3) << "Map didn't have a parallel computation " << inst->name();
      _is_ok = false;
      return Status::OK();
    }
  }

  bool _is_ok;
};
}  // namespace

StatusOr<bool> IsParallelMap(const HloInstruction* inst,
                             const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

StatusOr<DriverProgramSequence> CreateParallelMap(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  DriverProgramSequence seq(debug_name_and_id);
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64_t op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  MapVisitor visitor(res, inputs, output, debug_name_and_id);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  seq.add(visitor.GetSequence());

  auto outputs = visitor.outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i]));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreateCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  DriverProgramSequence seq(debug_name_and_id);

  if (IsRepeatLoop(inst)) {
    // Version of the repeat operation which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(
        seq, CreateRepeatOp(res, inst, output, tensor_map, debug_name_and_id));
  } else if (IsPipelineOp(inst)) {
    // Version of the pipeline operation which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(seq, CreatePipelineOp(res, inst, output, tensor_map,
                                              debug_name_and_id));
  } else {
    // Version of a function call which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(seq, CreateFunctionOp(res, inst, output, tensor_map,
                                              debug_name_and_id));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto& graph = GetGraph(res, inst);
  if (IsPoplibsHloCustomOp(inst)) {
    VLOG(1) << "Processing " << inst->custom_call_target()
            << " as Poplibs call";
    return CreatePoplarOp(graph, res, inst, output, tensor_map,
                          debug_name_and_id);
  } else {
    return xla::FailedPrecondition("Unrecognised kCustomCall %s.",
                                   inst->ToString().c_str());
  }
}

StatusOr<DriverProgramSequence> CreateFusionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  HloComputation* comp = inst->fused_instructions_computation();
  DriverProgramSequence seq(debug_name_and_id);
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64_t op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }

  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  InplaceDeferredVisitor inplace_visitor(
      res, deferred_inputs, GetInplaceDescription(inst), debug_name_and_id);
  TF_RETURN_IF_ERROR(comp->Accept(&inplace_visitor));

  seq.add(inplace_visitor.GetSequence());
  const TensorOrRemoteBufferVector& outputs = inplace_visitor.outputs();

  for (size_t i = 0; i < outputs.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i]));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverProgramSequence seq(debug_name_and_id);
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(DriverProgramSequence while_seq,
                      CreateWhileOp(res, inst, deferred_inputs, output,
                                    tensor_map, debug_name_and_id));
  seq.add(while_seq);
  return seq;
}

StatusOr<DriverProgramSequence> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  CHECK_EQ(inputs.size(), 1);
  const HloInstruction* input_inst = inst->operand(0);

  ReallocateInputsInfo reallocate_input_info(1);
  reallocate_input_info[0].resize(inputs[0].size());

  auto inplace_description = GetInplaceDescription(inst);
  auto& inplace_operand_set = inplace_description.GetInplaceOperandSet();
  if (inplace_operand_set.contains(0)) {
    for (std::size_t i = 0; i < inputs[0].size(); ++i) {
      const auto& input = inputs[0][i];
      if (input && input->IsTensor() &&
          !input->AsTensor().isParallelWriteable()) {
        reallocate_input_info[0][i] = true;
      }
    }
  }

  auto& graph = GetGraph(res, inst);

  // Create a visitor for the condition computation.
  // Conditional should not change the inputs - therefore it's not inplace.
  // Note that the visitor explicitly doesn't allocate all the input tensors for
  // the conditional computation in order to allow the body to visitor to create
  // the tensors.
  const auto condition_start_method = GetStochasticRoundingMethod(res);
  DeferredVisitor condition_visitor(res, inputs, {debug_name_and_id, "Cond"},
                                    /*allocate_all_input_tensors*/ false);
  const HloComputation* condition_comp = inst->while_condition();
  {
    auto order = condition_comp->parent()
                     ->schedule()
                     .sequence(condition_comp)
                     .instructions();
    TF_RETURN_IF_ERROR(
        condition_comp->AcceptOrdered(&condition_visitor, order));
  }
  const auto condition_end_method = GetStochasticRoundingMethod(res);

  // Create an inplace visitor for the loop body.
  InplaceDeferredVisitor body_visitor(
      res, inputs, GetInplaceDescription(inst), {debug_name_and_id, "Body"},
      {&condition_visitor}, reallocate_input_info);
  const HloComputation* body_comp = inst->while_body();
  {
    auto order =
        body_comp->parent()->schedule().sequence(body_comp).instructions();
    TF_RETURN_IF_ERROR(body_comp->AcceptOrdered(&body_visitor, order));

    // Make sure any deferred inputs to the instruction are pushed up.
    TF_RETURN_IF_ERROR(body_visitor.PropagateDeferredAllocations(
        inst, inputs, debug_name_and_id));
  }

  const uint64 param_count = inputs[0].size();
  const TensorOrRemoteBufferVector& body_inputs = body_visitor.inputs()[0];
  const TensorOrRemoteBufferVector& body_outputs = body_visitor.outputs();
  const TensorOrRemoteBufferVector& cond_inputs = condition_visitor.inputs()[0];
  const TensorOrRemoteBufferVector& cond_outputs = condition_visitor.outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs.");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs.");
  }
  if (cond_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of condition inputs.");
  }
  if (cond_outputs.size() != 1) {
    return xla::FailedPrecondition("Invalid number of condition outputs.");
  }

  ExecutionCounters& cond_counters = condition_visitor.GetExecutionCounters();
  ExecutionCounters& body_counters = body_visitor.GetExecutionCounters();

  DriverProgramSequence main_seq({debug_name_and_id, "main"});
  // Add copies for any inputs which were reallocated.
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence copy_seq,
                      body_visitor.GetPreambleCopies(debug_name_and_id));
  main_seq.add(copy_seq);

  // Only add the zeroing for the execution counters in the condition and body
  // visitors once before the execution of the loop so that they are not reset
  // at the beginning of each iteration.
  main_seq.add(cond_counters.SetInitialValuesToZero());
  main_seq.add(body_counters.SetInitialValuesToZero());

  // Create a sequence and predicate for the condition.
  DriverProgramSequence cond_seq({debug_name_and_id, "condition"});
  poplar::Tensor predicate;
  {
    // Before executing the condition, copy inputs which are required by
    // the condition to cond_inputs.
    for (uint64 i = 0; i < param_count; i++) {
      if (condition_visitor.InputIsUsed(0, i) && body_inputs[i].IsTensor()) {
        cond_seq.add(poplar::program::Copy(body_inputs[i].AsTensor(),
                                           cond_inputs[i].AsTensor(), false,
                                           {debug_name_and_id}));
      }
    }
    cond_seq.add(condition_visitor.GetSequence(
        /*copy_execution_counters*/ false));

    predicate = popops::allTrue(graph, cond_outputs[0].AsTensor(), cond_seq,
                                {debug_name_and_id});
    // Increase the local execution counters at the end of each iteration.
    cond_seq.add(cond_counters.IncrementLiveCounters());
  }
  // Add the aliasing copies for the loop so that the outputs of one iteration
  // are aliased to the inputs of the next one.
  TF_ASSIGN_OR_RETURN(const TensorOrRemoteBufferVector loop_state,
                      body_visitor.AddLoopInputOutputAliasingCopies(
                          graph, body_comp, {debug_name_and_id}));
  // Create a sequence for the body.
  DriverProgramSequence body_seq({debug_name_and_id, "body"});
  {
    body_seq.add(body_visitor.GetSequence(/*copy_execution_counters*/ false));
    // After each loop iteration the condition is run again, so we need to make
    // sure that we're using SR method for the condition before it's executed.
    MaybeChangeStochasticRoundingMethod(res, inst->name() + "_iter_end",
                                        condition_start_method, body_seq);

    // Increase the local execution counters at the end of each iteration.
    body_seq.add(body_counters.IncrementLiveCounters());
  }

  // We always run the condition block after the while body, so the actual SR
  // method that will be active at the end loop will be the one from the
  // condition block. Update the SR method to reflect that.
  MaybeSetStochasticRoundingMethod(res, condition_end_method);

  // Create the while loop.
  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, predicate, body_seq,
                                                {debug_name_and_id}));

  for (uint64 i = 0; i < param_count; i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, loop_state[i]));
  }

  return main_seq;
}

StatusOr<DriverProgramSequence> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverProgramSequence seq(debug_name_and_id);
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(DriverProgramSequence loop_seq,
                      CreateRepeatOp(res, inst, deferred_inputs, output,
                                     tensor_map, debug_name_and_id));
  seq.add(loop_seq);
  return seq;
}

namespace {
bool SingleIPUComputation(const HloComputation* computation) {
  absl::flat_hash_set<int> shard_indices;

  for (auto inst : computation->instructions()) {
    if (inst->has_sharding() && !inst->sharding().HasUniqueDevice()) {
      return false;
    }

    if (inst->has_sharding()) {
      shard_indices.insert(inst->sharding().GetUniqueDevice());
    }
  }

  return shard_indices.size() < 2;
}

StatusOr<bool> ComputationHasIoTileInstructions(
    const HloComputation* computation) {
  for (auto inst : computation->instructions()) {
    TF_ASSIGN_OR_RETURN(const auto tileset, GetTileset(inst));

    if (tileset == TILESET_IO_TILES) {
      return true;
    }
  }

  return false;
}

bool VerifyOpaqueUsers(const HloInstruction* param,
                       const HloInstruction* root) {
  // Matching instructions are trivially correct.
  if (param == root) {
    return true;
  }

  // Mismatched shapes can't be compatible.
  if (param->shape() != root->shape()) {
    return false;
  }

  // Any arrays can be compatible, so long as the shapes match.
  if (param->shape().IsArray() && root->shape().IsArray()) {
    return true;
  }

  // Any tokens can be compatible.
  if (param->shape().IsToken() && root->shape().IsToken()) {
    return true;
  }

  // Opaque, but not matching instructions is invalid.
  if (param->shape().IsOpaque() && (param != root)) {
    return false;
  }

  // For tuple shapes, we check the gte users. This is equivalent to the
  // operands of the corresponding root tuple instructions.
  if (param->shape().IsTuple() && root->shape().IsTuple()) {
    auto users = param->users();

    auto is_gte_pred = [](const HloInstruction* user) -> bool {
      return user->opcode() == HloOpcode::kGetTupleElement;
    };

    // For a GTE user, we walk back to the operand at the tuple index.
    auto handle_gte_user_pred = [&](const HloInstruction* user) -> bool {
      if (user->tuple_index() < root->operand_count()) {
        return VerifyOpaqueUsers(user, root->operand(user->tuple_index()));
      }

      return false;
    };

    // For a non-GTE user, we try all the operands.
    auto handle_user_pred = [&](const HloInstruction* user) -> bool {
      for (std::size_t i = 0; i < root->operand_count(); ++i) {
        if (VerifyOpaqueUsers(user, root->operand(i))) {
          return true;
        }
      }

      return false;
    };

    auto itr = absl::c_stable_partition(users, is_gte_pred);
    return std::all_of(users.begin(), itr, handle_gte_user_pred) &&
           std::all_of(itr, users.end(), handle_user_pred);
  }

  return false;
}

/**
 * We require any opaque arguments to a loop to be passed through in the same
 * position. This is because the opaque values are propogated at compile-time,
 * so don't really interact with the running loop. Requiring this makes the
 * behaviour of opaque arguments consistent with runtime arguments.
 */
Status CheckLoopOpaqueAliasing(CompilerResources& res,
                               const HloInstruction* inst) {
  const HloComputation* loop_body = inst->to_apply();

  const HloInstruction* root = loop_body->root_instruction();
  const std::vector<HloInstruction*> params =
      loop_body->parameter_instructions();

  auto error = InternalErrorStrCat(
      "Opaque type tensor passed to loop", inst->name(),
      ", but does not alias the output. Input opaque tensors must appear in "
      "the root instruction at the same position.");

  // Check whether the input is opaque and whether it is passed through from the
  // parameter.
  if (root->shape().IsOpaque() && root != params[0]) {
    return error;
  }

  // Ignore non-tuple loops.
  if (root->opcode() != HloOpcode::kTuple) {
    return Status::OK();
  }

  // For each input, check whether the input is opaque and whether it is passed
  // through from the parameter to the root in the same position.
  for (std::size_t i = 0u; i < params.size(); ++i) {
    if (!VerifyOpaqueUsers(params[i], root->operand(i))) {
      return error;
    }
  }

  return Status::OK();
}

namespace {
bool CouldHaveSmallGradientAccumulation(const HloInstruction* inst) {
  // Find a resource update instruction.
  const auto insts = inst->to_apply()->instructions();
  auto itr = absl::c_find_if(insts, IsResourceUpdate);

  // If there isn't a resource update, then there isn't a small gradient
  // accumulation loop.
  if (itr == insts.end()) {
    return false;
  }

  // There is a gradient accumulation loop, so test its iteration count.
  const auto batches = GetResourceUpdateBatchesToAccumulate(*itr);
  if (batches && *batches > 1) {
    return false;
  }

  // We found a resource update and the gradient accumulation count was small.
  return true;
}
}  // namespace

StatusOr<std::unique_ptr<RepeatLoopVisitor>> CreateLoopVisitor(
    CompilerResources& res, const HloInstruction* inst,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const ReallocateInputsInfo& reallocate_inputs_info,
    const poplar::DebugNameAndId& debug_name_and_id) {
  // Check any opaque typed tensors are correctly connected to the root
  // instruction.
  TF_RETURN_IF_ERROR(CheckLoopOpaqueAliasing(res, inst));

  // If the repeat is only on a single IPU and has instructions on IO tiles,
  // then create an overlapping repeat visitor.
  TF_ASSIGN_OR_RETURN(bool has_io_tile_inst,
                      ComputationHasIoTileInstructions(inst->to_apply()));

  // We can't overlap the IO on a loop with too few iterations.
  const int64_t repeat_count = GetRepeatLoopCount(inst);

  // We also can't overlap small gradient accumulation factors.
  const bool small_gradient_accumulation =
      CouldHaveSmallGradientAccumulation(inst);

  if (SingleIPUComputation(inst->to_apply()) && has_io_tile_inst &&
      repeat_count > 1 && !small_gradient_accumulation) {
    return {absl::make_unique<RepeatLoopOverlapIOVisitor>(
        res, inputs, description, reallocate_inputs_info, debug_name_and_id)};
  } else {
    return absl::make_unique<RepeatLoopVisitor>(
        res, inputs, description, reallocate_inputs_info, debug_name_and_id);
  }
}
}  // namespace

StatusOr<DriverProgramSequence> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  CHECK_EQ(inputs.size(), inst->operand_count());

  ReallocateInputsInfo reallocate_input_info(inst->operand_count());

  auto inplace_description = GetInplaceDescription(inst);
  auto& inplace_operand_set = inplace_description.GetInplaceOperandSet();

  // Reallocate any inputs which are inplace and non parallel-writeable
  for (int64_t i = 0; i != inst->operand_count(); ++i) {
    const auto& op_inputs = inputs[i];
    reallocate_input_info[i].resize(op_inputs.size());

    if (!inplace_operand_set.contains(i)) {
      continue;
    }

    for (std::size_t j = 0; j < op_inputs.size(); ++j) {
      const auto& input = op_inputs[j];
      if (input && input->IsTensor() &&
          !input->AsTensor().isParallelWriteable()) {
        reallocate_input_info[i][j] = true;
      }
    }
  }

  const HloComputation* loop_body = inst->to_apply();
  auto order =
      loop_body->parent()->schedule().sequence(loop_body).instructions();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<RepeatLoopVisitor> visitor,
      CreateLoopVisitor(res, inst, inputs, GetInplaceDescription(inst),
                        reallocate_input_info, debug_name_and_id));

  // Evaluate the loop body in a order.
  TF_RETURN_IF_ERROR(loop_body->AcceptOrdered(visitor.get(), order));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(
      visitor->PropagateDeferredAllocations(inst, inputs, debug_name_and_id));

  const TensorOrRemoteBufferVector& loop_state = visitor->GetLoopState();

  DriverProgramSequence seq = visitor->GetRepeatLoopSequence(inst);

  for (uint64 i = 0; i < loop_state.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, loop_state[i]));
  }

  return seq;
}

namespace {
TensorOrRemoteBufferVectors GetAllInstructionInputs(
    CompilerResources& res, const HloInstruction* inst,
    DriverProgramSequence& seq, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TensorOrRemoteBufferVectors inputs(inst->operand_count());
  for (int64_t i = 0; i != inst->operand_count(); ++i) {
    inputs[i] =
        FindInstructionInputs(tensor_map, res, inst, i, seq, debug_name_and_id);
  }
  return inputs;
}
}  // namespace

StatusOr<DriverProgramSequence> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverProgramSequence seq(debug_name_and_id);

  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);
  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(DriverProgramSequence func_seq,
                      CreateFunctionOp(res, inst, deferred_inputs, output,
                                       tensor_map, debug_name_and_id));
  seq.add(func_seq);
  return seq;
}

StatusOr<DriverProgramSequence> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& deferred_inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  auto& graph = GetGraph(res, inst);
  DriverProgramSequence seq(debug_name_and_id);

  HloComputation* comp = inst->to_apply();

  bool keep_input_layouts = false;
  if (IsFunction(inst) || IsCall(inst)) {
    keep_input_layouts = GetFunctionKeepInputLayouts(inst);
  }

  // Get information about remote buffer inputs.
  const int64_t num_modified_remote_buffer_inputs =
      GetFunctionNumberModifiedRemoteBufferInputs(inst);
  const int64_t num_unmodified_remote_buffer_inputs =
      GetFunctionNumberUnmodifiedRemoteBufferInputs(inst);
  const int64_t num_remote_buffer_inputs =
      num_modified_remote_buffer_inputs + num_unmodified_remote_buffer_inputs;
  const bool partitioned_elementwise_cluster =
      GetFunctionPartitionedElementwiseCluster(inst);

  // This instruction needs to be lowered inplace on the remote buffers.
  if (num_remote_buffer_inputs) {
    if (!IsLoweredInplace(inst)) {
      return InternalErrorStrCat(
          "Found a function ", inst->ToString(),
          " with remote buffer inputs which is not lowered inplace.");
    }
  }

  TF_ASSIGN_OR_RETURN(auto subcomp_visitor,
                      res.subcomputation_cache.GetOrCompileSubcomputation(
                          res, deferred_inputs, comp, keep_input_layouts,
                          partitioned_elementwise_cluster));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(subcomp_visitor->PropagateDeferredAllocations(
      inst, deferred_inputs, debug_name_and_id));

  // Now that the all the inputs have been allocated and propagated, get them.
  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);
  for (int64_t o = 0; o < inst->operand_count(); o++) {
    auto& comp_inputs = subcomp_visitor->inputs()[o];
    auto& inst_inputs = inputs[o];
    if (comp_inputs.size() != inst_inputs.size()) {
      return xla::FailedPrecondition("Mismatched number of inputs.");
    }
    for (size_t i = 0; i < inst_inputs.size(); i++) {
      auto& inst_input = inst_inputs[i];
      auto& comp_input = comp_inputs[i];
      if (o < num_remote_buffer_inputs) {
        if (!inst_input.IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Unable to handle input to function call instruction %s at input "
              "index %d.",
              inst->name(), o);
        }
        CHECK(comp_input.IsRemoteBuffer());
        CHECK(comp_input == inst_input);
      } else {
        if (subcomp_visitor->InputIsUsed(o, i)) {
          if (comp_input.IsTensor()) {
            seq.add(poplar::program::Copy(inst_input.AsTensor(),
                                          comp_input.AsTensor(), false,
                                          debug_name_and_id));
          } else if (comp_input.IsOpaque()) {
            inst_input = comp_input.AsOpaque();
          } else if (comp_input.IsRemoteBuffer()) {
            return xla::FailedPrecondition(
                "Unable to handle used remote buffer tensor in function call "
                "instruction %s",
                inst->name());
          } else {
            return xla::FailedPrecondition(
                "Unknown output type in function call instruction `%s`, "
                "expected "
                "a tensor",
                inst->name());
          }
        }
      }
    }
  }

  // Add the function.
  seq.add(subcomp_visitor->GetFunctionCall());

  // Propagate the outputs.
  auto& outputs = subcomp_visitor->outputs();
  int64_t flat_tuple_index = 0;
  auto output_locations = ShapeUtil::GetLeafShapes(output);

  for (const auto& output_location : output_locations) {
    const ShapeIndex& shape_index = output_location.index;
    const int64_t tuple_index = shape_index.empty() ? 0 : shape_index[0];
    auto& output = outputs[flat_tuple_index];

    if (tuple_index < num_modified_remote_buffer_inputs) {
      if (!output.IsRemoteBuffer()) {
        return InternalErrorStrCat("Expected output at index ",
                                   shape_index.ToString(),
                                   " to be a RemoteBuffer object");
      }
      TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, flat_tuple_index, output));
    } else {
      if (output.IsOpaque()) {
        TF_RETURN_IF_ERROR(
            AddOutput(tensor_map, inst, flat_tuple_index, output));
      } else if (!output.IsTensor()) {
        return InternalErrorStrCat("Expected output at index ",
                                   shape_index.ToString(),
                                   " to be a Tensor object");
      } else {
        auto name = absl::StrCat("out/", flat_tuple_index);
        poplar::Tensor cloned_output = poputil::duplicate(
            graph, output.AsTensor(), seq, {debug_name_and_id, name},
            poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
        TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, flat_tuple_index,
                                           DriverTensor(cloned_output)));
      }
    }
    flat_tuple_index++;
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  DriverProgramSequence seq(debug_name_and_id);

  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);

  TF_ASSIGN_OR_RETURN(DriverProgramSequence pipeline_seq,
                      CreatePipelineOp(res, inst, deferred_inputs, output,
                                       tensor_map, debug_name_and_id));
  seq.add(pipeline_seq);

  return seq;
}

static DriverTensor GetGradientAccumulationCountTensor(
    const HloInstruction* inst, DeferredArgRBVectors& inputs) {
  int64_t index = GetAccumulationCountOperandIndex(inst);
  const auto& input = inputs[index];
  CHECK_EQ(input.size(), 1);
  CHECK(static_cast<bool>(input[0]));
  return input[0]->AsTensor();
}

static PipelineVisitor::IterationsType GetIterationsArgument(
    const HloInstruction* inst, DriverGraph& graph,
    DeferredArgRBVectors& inputs) {
  auto count = GetGradientAccumulationCount(inst);
  if (count) {
    return *count;
  }
  return PipelineVisitor::CountAndGraph(
      graph, GetGradientAccumulationCountTensor(inst, inputs));
}

StatusOr<DriverProgramSequence> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  auto& graph = GetGraph(res, inst);
  DriverProgramSequence seq(debug_name_and_id);
  HloComputation* pipeline_computation = inst->to_apply();
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                      inst->backend_config<PoplarBackendConfig>());

  auto gradient_accumulation_count = GetGradientAccumulationCount(inst);

  int64_t repeat_count = cfg.call_config().pipeline_config().repeat_count();

  CHECK_EQ(inputs.size(), inst->operand_count());

  // Check any opaque typed tensors are correctly connected to the root
  // instruction.
  TF_RETURN_IF_ERROR(CheckLoopOpaqueAliasing(res, inst));

  // Compile the pipeline.
  TF_ASSIGN_OR_RETURN(
      auto visitor,
      GetPipelineVisitor(inst, res, inputs, GetInplaceDescription(inst),
                         debug_name_and_id));

  if (gradient_accumulation_count) {
    // If known at compile time want run error checking before construction
    // of the pipeline graph. Provide dummy tensor as won't be used.
    // If not known at compile time we can't create the sequence yet as
    // need to call PropagateDeferredAllocations first
    auto dummy = graph.addVariable(poplar::FLOAT, {});
    poputil::mapTensorLinearly(graph, dummy);

    TF_RETURN_IF_ERROR(
        visitor
            ->VerifyPipelineArguments(
                GetGradientAccumulationCountInstruction(inst), dummy, graph)
            .status());
  }

  auto order = pipeline_computation->parent()
                   ->schedule()
                   .sequence(pipeline_computation)
                   .instructions();
  TF_RETURN_IF_ERROR(pipeline_computation->AcceptOrdered(visitor.get(), order));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(
      visitor->PropagateDeferredAllocations(inst, inputs, debug_name_and_id));

  // Make sure that inputs/outputs alias each other.
  TF_ASSIGN_OR_RETURN(auto pipeline_state,
                      visitor->AddLoopInputOutputAliasingCopies(
                          graph, pipeline_computation, debug_name_and_id));
  ExecutionCounters& execution_counters = visitor->GetExecutionCounters();
  // Initialize the counters.
  seq.add(execution_counters.SetInitialValuesToZero());

  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence verification_prog,
      visitor->VerifyPipelineArguments(
          GetGradientAccumulationCountInstruction(inst),
          GetGradientAccumulationCountTensor(inst, inputs), graph));

  // Get the pipeline sequence.
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence pipeline_prog,
      visitor->GetPipelineSequence(GetIterationsArgument(inst, graph, inputs)));
  // Increase the counters at the end of each pipeline execution.
  pipeline_prog.add(execution_counters.IncrementLiveCounters());

  seq.add(verification_prog);
  seq.add(
      poplar::program::Repeat(repeat_count, pipeline_prog, debug_name_and_id));

  for (size_t i = 0; i < pipeline_state.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, pipeline_state[i]));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  DriverProgramSequence seq(debug_name_and_id);
  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);

  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateConditionalOp(res, inst, deferred_inputs, output,
                                          tensor_map, debug_name_and_id));
  seq.add(prog);

  return seq;
}

StatusOr<DriverProgramSequence> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& deferred_inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  auto& graph = GetGraph(res, inst);

  DriverProgramSequence seq(debug_name_and_id);

  bool is_switch;
  if (inst->operand(0)->shape().element_type() == PRED) {
    is_switch = false;
  } else if (inst->operand(0)->shape().element_type() == S32) {
    is_switch = true;
  } else {
    return xla::FailedPrecondition(
        "Conditional %s has unsupported condition input type %s.",
        PrimitiveType_Name(inst->operand(0)->shape().element_type()));
  }

  int n_branches = inst->operand_count() - 1;
  if (n_branches == 0) {
    return xla::FailedPrecondition("Conditional %s has no branches.",
                                   inst->name().c_str());
  }

  auto conditional_start_sr_method = GetStochasticRoundingMethod(res);
  std::vector<StochasticRoundingMethod> bodies_end_sr_methods;

  std::vector<std::shared_ptr<DeferredVisitor>> bodies(n_branches);
  const auto& comps = inst->called_computations();

  // Compile each branch into a sequence
  for (auto b = 0; b < n_branches; b++) {
    // The seed changes of the branch that gets executed should be relative to
    // the start of the conditional.
    MaybeSetStochasticRoundingMethod(res, conditional_start_sr_method);

    CHECK_EQ(deferred_inputs[b + 1].size(),
             CountShapes(inst->operand(b + 1)->shape()));
    DeferredArgRBVectors body_inputs = {deferred_inputs[b + 1]};
    TF_ASSIGN_OR_RETURN(bodies[b],
                        res.subcomputation_cache.GetOrCompileSubcomputation(
                            res, body_inputs, comps[b], true));
    bodies_end_sr_methods.push_back(GetStochasticRoundingMethod(res));

    // Make sure any deferred inputs to the instruction are pushed up.
    TF_RETURN_IF_ERROR(bodies[b]->PropagateDeferredAllocationsOperand(
        inst, b + 1, 0, body_inputs[0], debug_name_and_id));
  }
  // We need a SR method that seed changes after this conditional should be
  // relative to, since the branches could end on different methods. For this we
  // choose the SR method of the last branch and make all branches end on that.
  auto conditional_end_method = bodies_end_sr_methods.back();

  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);

  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor pred = inputs[0][0].AsTensor();
  unsigned int output_count = bodies[0]->outputs().size();

  // Add final output tensors for the conditional op
  std::vector<poplar::Tensor> outputs;
  for (unsigned int i = 0; i < output_count; i++) {
    poplar::Tensor out =
        graph.clone(bodies[0]->outputs()[i].AsTensor(), debug_name_and_id);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
    outputs.push_back(out);
  }

  // Create the full program sequences for each branch
  // TODO(T32947) : Would be good if we could name each branch
  std::vector<poplar::program::Sequence> seqs(bodies.size(),
                                              {{}, debug_name_and_id});
  for (auto b = 0; b < n_branches; b++) {
    if (bodies[b]->inputs().size() != 1) {
      return xla::FailedPrecondition("Invalid input count on branch %d.", b);
    }
    if (bodies[b]->outputs().size() != output_count) {
      return xla::FailedPrecondition("Mismatched output size on branch %d.", b);
    }

    // Add input copies
    for (unsigned int i = 0; i < bodies[b]->inputs()[0].size(); i++) {
      if (bodies[b]->InputIsUsed(0, i)) {
        if (bodies[b]->inputs()[0][i].IsTensor()) {
          seqs[b].add(poplar::program::Copy(
              inputs[b + 1][i].AsTensor(), bodies[b]->inputs()[0][i].AsTensor(),
              false, debug_name_and_id));
        } else if (bodies[b]->inputs()[0][i].IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Unable to handle used remote buffer tensor in conditional "
              "instruction %s",
              inst->name());
        } else {
          return xla::FailedPrecondition(
              "Unknown output type in conditional instruction `%s`, expected a "
              "tensor.",
              inst->name());
        }
      }
    }

    // Add the actual body.
    seqs[b].add(bodies[b]->GetSequence());

    // Make sure all branches end on the same SR method.
    MaybeSetStochasticRoundingMethod(res, bodies_end_sr_methods[b]);
    const std::string change_name =
        absl::StrCat(inst->name(), "_body", std::to_string(b), "_end");
    MaybeChangeStochasticRoundingMethod(res, change_name,
                                        conditional_end_method, seqs[b]);

    // Add output copies
    for (unsigned int i = 0; i < output_count; i++) {
      if (bodies[b]->outputs()[i].IsTensor()) {
        seqs[b].add(poplar::program::Copy(bodies[b]->outputs()[i].AsTensor(),
                                          outputs[i], false,
                                          debug_name_and_id));
      } else if (bodies[b]->outputs()[i].IsRemoteBuffer()) {
        return xla::FailedPrecondition(
            "Unable to output remote buffer tensor in conditional "
            "instruction %s",
            inst->name());
      } else {
        return xla::FailedPrecondition(
            "Unknown output type in conditional instruction `%s`, expected "
            "a tensor",
            inst->name());
      }
    }
  }

  if (!is_switch) {
    poplar::Tensor scalar_pred =
        popops::allTrue(graph, pred, seq, {debug_name_and_id});

    seq.add(poplar::program::If(scalar_pred, seqs[0], seqs[1],
                                {debug_name_and_id}));
  } else {
    std::vector<std::pair<int32, poplar::program::Program>> cases;
    for (int64_t c = 0; c < static_cast<int64_t>(seqs.size()) - 1; c++) {
      cases.push_back(std::make_pair(c, seqs[c]));
    }
    seq.add(
        poplar::program::Switch(pred, cases, seqs.back(), {debug_name_and_id}));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreateResourceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  HloComputation* resource_update_comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : "
          << resource_update_comp->name() << " as a resource update.";
  auto& graph = GetGraph(res, inst);
  DriverProgramSequence seq(debug_name_and_id);
  // Create a visitor for the resource update.
  InplaceDeferredVisitor visitor(res, inputs, GetInplaceDescription(inst),
                                 debug_name_and_id);
  auto order = resource_update_comp->parent()
                   ->schedule()
                   .sequence(resource_update_comp)
                   .instructions();
  TF_RETURN_IF_ERROR(resource_update_comp->AcceptOrdered(&visitor, order));
  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(
      visitor.PropagateDeferredAllocations(inst, inputs, debug_name_and_id));
  // Add to the sequence.
  seq.add(visitor.GetSequence());
  seq.add(visitor.GetExecutionCounters().IncrementLiveCounters());
  // Set up the outputs.
  const TensorOrRemoteBufferVector& outputs = visitor.outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i]));
  }

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
