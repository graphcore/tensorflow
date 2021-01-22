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
      VLOG(1) << "Map didn't have a parallel computation " << inst->name();
      _is_ok = false;
      return Status::OK();
    }
  }

  bool _is_ok;
};

StatusOr<bool> IsParallelMap(const HloInstruction* inst,
                             const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

StatusOr<poplar::program::Program> CreateParallelMap(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Sequence seq({}, debug_name_and_id);
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
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

StatusOr<poplar::program::Program> CreateCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Sequence seq({}, debug_name_and_id);

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

StatusOr<poplar::program::Program> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::Graph& graph = GetGraph(res, inst);
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

StatusOr<poplar::program::Program> CreateFusionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  HloComputation* comp = inst->fused_instructions_computation();
  poplar::program::Sequence seq({}, debug_name_and_id);
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }

  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  InplaceDeferredVisitor inplace_visitor(
      res, deferred_inputs, HloInstructionDescription(inst), debug_name_and_id);
  TF_RETURN_IF_ERROR(comp->Accept(&inplace_visitor));

  seq.add(inplace_visitor.GetSequence());
  const TensorOrRemoteBufferVector& outputs = inplace_visitor.outputs();

  for (size_t i = 0; i < outputs.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i]));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));
  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence while_seq,
                      CreateWhileOp(res, inst, deferred_inputs, output,
                                    tensor_map, debug_name_and_id));
  seq.add(while_seq);
  return seq;
}

StatusOr<poplar::program::Program> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  CHECK_EQ(inputs.size(), 1);
  const HloInstruction* input_inst = inst->operand(0);

  const bool reallocate_all_inputs =
      !(IsLoweredInplace(inst) && IsLoweredInplace(input_inst));
  ReallocateInputsInfo reallocate_input_info(1);
  reallocate_input_info[0] =
      std::vector<bool>(inputs[0].size(), reallocate_all_inputs);

  // Reallocate any inputs which are copies.
  if (input_inst->opcode() == HloOpcode::kTuple) {
    for (int64 i = 0, flat_index = 0; i != input_inst->operand_count(); ++i) {
      const HloInstruction* operand = input_inst->operand(i);
      const int64 num_tensors = CountShapes(operand->shape());
      if (operand->opcode() == HloOpcode::kCopy) {
        std::fill_n(reallocate_input_info[0].begin() + flat_index, num_tensors,
                    true);
      }
      flat_index += num_tensors;
    }
  }

  poplar::Graph& graph = GetGraph(res, inst);

  // Create a visitor for the condition computation.
  // Conditional should not change the inputs - therefore it's not inplace.
  // Note that the visitor explicitly doesn't allocate all the input tensors for
  // the conditional computation in order to allow the body to visitor to create
  // the tensors.
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

  // Create an inplace visitor for the loop body.
  InplaceDeferredVisitor body_visitor(
      res, inputs, HloInstructionDescription(inst), {debug_name_and_id, "Body"},
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

  poplar::program::Sequence main_seq({}, {debug_name_and_id, "main"});
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
  poplar::program::Sequence cond_seq({}, {debug_name_and_id, "condition"});
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
    cond_seq.add(
        condition_visitor.GetSequence(/*copy_execution_counters*/ false));

    predicate =
        popops::allTrue(graph, cond_outputs[0], cond_seq, {debug_name_and_id});
    // Increase the local execution counters at the end of each iteration.
    cond_seq.add(cond_counters.IncrementLiveCounters());
  }
  // Add the aliasing copies for the loop so that the outputs of one iteration
  // are aliased to the inputs of the next one.
  TF_ASSIGN_OR_RETURN(const TensorOrRemoteBufferVector loop_state,
                      body_visitor.AddLoopInputOutputAliasingCopies(
                          graph, body_comp, {debug_name_and_id}));
  // Create a sequence for the body.
  poplar::program::Sequence body_seq({}, {debug_name_and_id, "body"});
  {
    body_seq.add(body_visitor.GetSequence(/*copy_execution_counters*/ false));
    // Increase the local execution counters at the end of each iteration.
    body_seq.add(body_counters.IncrementLiveCounters());
  }

  // Create the while loop.
  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, predicate, body_seq,
                                                {debug_name_and_id}));

  for (uint64 i = 0; i < param_count; i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, loop_state[i]));
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence loop_seq,
                      CreateRepeatOp(res, inst, deferred_inputs, output,
                                     tensor_map, debug_name_and_id));
  seq.add(loop_seq);
  return seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  CHECK_EQ(inputs.size(), inst->operand_count());

  const bool reallocate_all_inputs = !IsLoweredInplace(inst);
  ReallocateInputsInfo reallocate_input_info(inst->operand_count());

  // Reallocate any inputs which are copies.
  for (int64 i = 0; i != inst->operand_count(); ++i) {
    const bool reallocate_input =
        reallocate_all_inputs || inst->operand(i)->opcode() == HloOpcode::kCopy;
    reallocate_input_info[i] =
        std::vector<bool>(inputs[i].size(), reallocate_input);
  }

  const HloComputation* loop_body = inst->to_apply();
  auto order =
      loop_body->parent()->schedule().sequence(loop_body).instructions();

  // Create the visitor.
  RepeatLoopVisitor visitor(res, inputs, HloInstructionDescription(inst),
                            reallocate_input_info, debug_name_and_id);

  // Evaluate the loop body in a order.
  TF_RETURN_IF_ERROR(loop_body->AcceptOrdered(&visitor, order));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(
      visitor.PropagateDeferredAllocations(inst, inputs, debug_name_and_id));

  const TensorOrRemoteBufferVector& loop_state = visitor.GetLoopState();

  poplar::program::Sequence seq = visitor.GetRepeatLoopSequence(inst);

  for (uint64 i = 0; i < loop_state.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, loop_state[i]));
  }

  return seq;
}

namespace {
TensorOrRemoteBufferVectors GetAllInstructionInputs(
    CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  TensorOrRemoteBufferVectors inputs(inst->operand_count());
  for (int64 i = 0; i != inst->operand_count(); ++i) {
    inputs[i] =
        FindInstructionInputs(tensor_map, res, inst, i, seq, debug_name_and_id);
  }
  return inputs;
}
}  // namespace

StatusOr<poplar::program::Program> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);

  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);
  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence func_seq,
                      CreateFunctionOp(res, inst, deferred_inputs, output,
                                       tensor_map, debug_name_and_id));
  seq.add(func_seq);
  return seq;
}

StatusOr<poplar::program::Program> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& deferred_inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence seq({}, debug_name_and_id);

  HloComputation* comp = inst->to_apply();

  bool keep_input_layouts = false;
  if (IsFunction(inst)) {
    keep_input_layouts = GetFunctionKeepInputLayouts(inst);
  }

  // Get information about remote buffer inputs.
  const int64 num_modified_remote_buffer_inputs =
      GetFunctionNumberModifiedRemoteBufferInputs(inst);
  const int64 num_unmodified_remote_buffer_inputs =
      GetFunctionNumberUnmodifiedRemoteBufferInputs(inst);
  const int64 num_remote_buffer_inputs =
      num_modified_remote_buffer_inputs + num_unmodified_remote_buffer_inputs;

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
                          res, deferred_inputs, comp, keep_input_layouts));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(subcomp_visitor->PropagateDeferredAllocations(
      inst, deferred_inputs, debug_name_and_id));

  // Now that the all the inputs have been allocated and propagated, get them.
  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);
  for (int64 o = 0; o < inst->operand_count(); o++) {
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
  int64 flat_tuple_index = 0;
  auto output_locations = ShapeUtil::GetLeafShapes(output);

  for (const auto& output_location : output_locations) {
    const ShapeIndex& shape_index = output_location.index;
    const int64 tuple_index = shape_index.empty() ? 0 : shape_index[0];
    auto& output = outputs[flat_tuple_index];

    if (tuple_index < num_modified_remote_buffer_inputs) {
      if (!output.IsRemoteBuffer()) {
        return InternalErrorStrCat("Expected output at index ",
                                   shape_index.ToString(),
                                   " to be a RemoteBuffer object");
      }
      TF_RETURN_IF_ERROR(AddOutput(tensor_map, inst, flat_tuple_index, output));
    } else {
      if (!output.IsTensor()) {
        return InternalErrorStrCat("Expected output at index ",
                                   shape_index.ToString(),
                                   " to be a Tensor object");
      }

      auto name = absl::StrCat("out/", flat_tuple_index);
      poplar::Tensor cloned_output = poputil::duplicate(
          graph, output.AsTensor(), seq, {debug_name_and_id, name},
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
      TF_RETURN_IF_ERROR(
          AddOutputTensor(tensor_map, inst, flat_tuple_index, cloned_output));
    }

    flat_tuple_index++;
  }

  return seq;
}

StatusOr<poplar::program::Program> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::program::Sequence seq({}, debug_name_and_id);
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(
      TensorOrRemoteBufferVectors inputs,
      FindInplaceOutputs(tensor_map, res, inst, seq, debug_name_and_id));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);

  TF_ASSIGN_OR_RETURN(poplar::program::Sequence pipeline_seq,
                      CreatePipelineOp(res, inst, deferred_inputs, output,
                                       tensor_map, debug_name_and_id));
  seq.add(pipeline_seq);

  return seq;
}

StatusOr<poplar::program::Program> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence seq({}, debug_name_and_id);
  HloComputation* pipeline_computation = inst->to_apply();
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                      inst->backend_config<PoplarBackendConfig>());
  int64 gradient_accumulation_count =
      cfg.call_config().pipeline_config().gradient_accumulation_count();
  int64 repeat_count = cfg.call_config().pipeline_config().repeat_count();

  CHECK_EQ(inputs.size(), inst->operand_count());

  // Compile the pipeline.
  TF_ASSIGN_OR_RETURN(
      auto visitor,
      GetPipelineVisitor(inst, res, inputs, HloInstructionDescription(inst),
                         debug_name_and_id));
  TF_RETURN_IF_ERROR(
      visitor->VerifyPipelineArguments(gradient_accumulation_count));

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

  // Get the pipeline sequence.
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence pipeline_prog,
      visitor->GetPipelineSequence(gradient_accumulation_count));
  // Increase the counters at the end of each pipeline execution.
  pipeline_prog.add(execution_counters.IncrementLiveCounters());

  seq.add(
      poplar::program::Repeat(repeat_count, pipeline_prog, debug_name_and_id));

  for (size_t i = 0; i < pipeline_state.size(); i++) {
    TF_CHECK_OK(AddOutput(tensor_map, inst, i, pipeline_state[i]));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq({}, debug_name_and_id);

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

  TensorOrRemoteBufferVectors inputs =
      GetAllInstructionInputs(res, inst, seq, tensor_map, debug_name_and_id);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor pred = inputs[0][0].AsTensor();

  std::vector<std::shared_ptr<DeferredVisitor>> bodies(n_branches);
  const auto& comps = inst->called_computations();

  // Compile each branch into a sequence
  for (auto b = 0; b < n_branches; b++) {
    CHECK_EQ(inputs[b + 1].size(), CountShapes(inst->operand(b + 1)->shape()));
    TensorOrRemoteBufferVectors body_inputs = {inputs[b + 1]};
    TF_ASSIGN_OR_RETURN(bodies[b],
                        res.subcomputation_cache.GetOrCompileSubcomputation(
                            res, body_inputs, comps[b]));
  }

  unsigned int output_count = bodies[0]->outputs().size();

  // Add final output tensors for the conditional op
  std::vector<poplar::Tensor> outputs;
  for (unsigned int i = 0; i < output_count; i++) {
    poplar::Tensor out =
        graph.clone(bodies[0]->outputs()[i], debug_name_and_id);
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
    for (int64 c = 0; c < static_cast<int64>(seqs.size()) - 1; c++) {
      cases.push_back(std::make_pair(c, seqs[c]));
    }
    seq.add(
        poplar::program::Switch(pred, cases, seqs.back(), {debug_name_and_id}));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateResourceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id) {
  HloComputation* resource_update_comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : "
          << resource_update_comp->name() << " as a resource update.";
  poplar::program::Sequence seq({}, debug_name_and_id);
  // Create a visitor for the resource update.
  InplaceDeferredVisitor visitor(res, inputs, HloInstructionDescription(inst),
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
