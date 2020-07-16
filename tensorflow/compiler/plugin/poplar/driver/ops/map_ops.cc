/* Copyright 2017-2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

using ::absl::StrCat;
using tensorflow::str_util::StartsWith;

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<TensorVectors> GetCallInputs(CompilerResources& res,
                                      const HloInstruction* inst,
                                      TensorMap& tensor_map,
                                      poplar::program::Sequence& seq,
                                      const bool expand_aliasing = true) {
  TensorVectors args;
  for (int64 i = 0; i < inst->operand_count(); i++) {
    TF_ASSIGN_OR_RETURN(TensorVector t,
                        FindInstructionInputTensors(tensor_map, res, inst, i,
                                                    seq, expand_aliasing));
    args.push_back(t);
  }
  return args;
}
}  // namespace

class ParallelMapTester : public DfsHloVisitorWithDefault {
 public:
  ParallelMapTester() : _is_ok(true) {}

  Status DefaultAction(HloInstruction* inst) override {
    if (inst->IsElementwise()) {
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

StatusOr<poplar::program::Program> CreateParallelMap(CompilerResources& res,
                                                     const HloInstruction* inst,
                                                     const xla::Shape& output,
                                                     TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  MapVisitor visitor(res, inputs, output, GetDebugName(inst));
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  seq.add(visitor.GetSequence());

  auto outputs = visitor.outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
    } else if (outputs[i].IsRemoteBuffer()) {
      TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, i, outputs[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in fusion instruction `%s`, expected a tensor "
          "or a remote buffer.",
          inst->name());
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateCallOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Sequence seq;

  if (IsRepeatLoop(inst)) {
    // Version of the repeat operation which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(seq, CreateRepeatOp(res, inst, output, tensor_map));
  } else if (IsPipelineOp(inst)) {
    // Version of the pipeline operation which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(seq, CreatePipelineOp(res, inst, output, tensor_map));
  } else {
    // Version of a function call which does not allow parameters to be
    // deferred.
    TF_ASSIGN_OR_RETURN(seq, CreateFunctionOp(res, inst, output, tensor_map));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);
  if (IsPoplibsHloCustomOp(inst)) {
    VLOG(1) << "Processing " << inst->custom_call_target()
            << " as Poplibs call";
    return CreatePoplarOp(graph, res, inst, output, tensor_map);
  } else {
    return xla::FailedPrecondition("Unrecognised kCustomCall %s.",
                                   inst->ToString().c_str());
  }
}

StatusOr<poplar::program::Program> CreateFusionOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  HloComputation* comp = inst->fused_instructions_computation();
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }

  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  InplaceDeferredVisitor inplace_visitor(res, deferred_inputs,
                                         GetDebugName(inst));
  TF_RETURN_IF_ERROR(comp->Accept(&inplace_visitor));

  seq.add(inplace_visitor.GetSequence());
  const TensorOrRemoteBufferVector& outputs = inplace_visitor.outputs();

  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
    } else if (outputs[i].IsRemoteBuffer()) {
      TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, i, outputs[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in fusion instruction `%s`, expected a tensor "
          "or a remote buffer.",
          inst->name());
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));
  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence while_seq,
      CreateWhileOp(res, inst, deferred_inputs, output, tensor_map));
  seq.add(while_seq);
  return seq;
}

StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 DeferredArgRBVectors& inputs,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map) {
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
  DeferredVisitor condition_visitor(res, inputs, GetDebugName(inst) + "/Cond",
                                    /*mark_all_input_tensors_as_used*/ false,
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
  InplaceDeferredVisitor body_visitor(res, inputs, GetDebugName(inst) + "/Body",
                                      {&condition_visitor},
                                      reallocate_input_info);
  const HloComputation* body_comp = inst->while_body();
  {
    auto order =
        body_comp->parent()->schedule().sequence(body_comp).instructions();
    TF_RETURN_IF_ERROR(body_comp->AcceptOrdered(&body_visitor, order));

    // Make sure any deferred inputs to the instruction are pushed up.
    TF_RETURN_IF_ERROR(body_visitor.PropagateDeferredAllocations(inst));
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

  poplar::program::Sequence main_seq;
  // Add copies for any inputs which were reallocated.
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence copy_seq,
                      body_visitor.GetPreambleCopies());
  main_seq.add(copy_seq);

  // Only add the copies for the execution counters in the condition and body
  // visitors once before the execution of the loop so that they are not reset
  // at the beginning of each iteration.
  TF_RETURN_IF_ERROR(
      CopyExecutionCountersFromScope(res, cond_counters, main_seq));
  TF_RETURN_IF_ERROR(
      CopyExecutionCountersFromScope(res, body_counters, main_seq));

  // Create a sequence and predicate for the condition.
  poplar::program::Sequence cond_seq;
  poplar::Tensor predicate;
  {
    // Before executing the condition, copy inputs which are required by
    // the condition to cond_inputs.
    for (uint64 i = 0; i < param_count; i++) {
      if (condition_visitor.InputIsUsed(0, i) && body_inputs[i].IsTensor()) {
        cond_seq.add(poplar::program::Copy(body_inputs[i].AsTensor(),
                                           cond_inputs[i].AsTensor()));
      }
    }
    cond_seq.add(
        condition_visitor.GetSequence(/*copy_execution_counters*/ false));

    predicate =
        popops::allTrue(graph, cond_outputs[0], cond_seq, GetDebugName(inst));
    // Increase the local execution counters at the end of each iteration.
    cond_seq.add(cond_counters.IncrementLiveCounters());
  }
  // Add the aliasing copies for the loop so that the outputs of one iteration
  // are aliased to the inputs of the next one.
  TF_ASSIGN_OR_RETURN(const TensorOrRemoteBufferVector loop_state,
                      body_visitor.AddLoopInputOutputAliasingCopies(
                          graph, body_comp, GetDebugName(inst)));
  // Create a sequence for the body.
  poplar::program::Sequence body_seq;
  {
    body_seq.add(body_visitor.GetSequence(/*copy_execution_counters*/ false));
    // Increase the local execution counters at the end of each iteration.
    body_seq.add(body_counters.IncrementLiveCounters());
  }

  // Create the while loop.
  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, predicate, body_seq));

  for (uint64 i = 0; i < param_count; i++) {
    if (loop_state[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, loop_state[i]));
    } else if (loop_state[i].IsRemoteBuffer()) {
      TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, i, loop_state[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in while instruction `%s`, expected a tensor "
          "or a remote buffer.",
          inst->name());
    }
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence loop_seq,
      CreateRepeatOp(res, inst, deferred_inputs, output, tensor_map));
  seq.add(loop_seq);
  return seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  DeferredArgRBVectors& inputs,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
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
  RepeatLoopVisitor visitor(res, inputs, reallocate_input_info,
                            GetDebugName(inst));

  // Evaluate the loop body in a order.
  TF_RETURN_IF_ERROR(loop_body->AcceptOrdered(&visitor, order));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(visitor.PropagateDeferredAllocations(inst));

  const TensorOrRemoteBufferVector& loop_state = visitor.GetLoopState();

  poplar::program::Sequence seq = visitor.GetRepeatLoopSequence(inst);

  for (uint64 i = 0; i < loop_state.size(); i++) {
    if (loop_state[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, loop_state[i]));
    } else if (loop_state[i].IsRemoteBuffer()) {
      TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, i, loop_state[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in repeat instruction `%s`, expected a tensor "
          "or a remote buffer.",
          inst->name());
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateFunctionOp(CompilerResources& res,
                                                    const HloInstruction* inst,
                                                    const xla::Shape& output,
                                                    TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                      GetCallInputs(res, inst, tensor_map, seq));

  HloComputation* comp = inst->to_apply();

  auto inputs_casted = CastTensorVectors(inputs);
  TF_ASSIGN_OR_RETURN(auto subcomp_visitor,
                      res.subcomputation_cache.GetOrCompileSubcomputation(
                          res, inputs_casted, comp));

  for (int64 o = 0; o < inst->operand_count(); o++) {
    auto& comp_inputs = subcomp_visitor->inputs()[o];
    auto& inst_inputs = inputs[o];
    if (comp_inputs.size() != inst_inputs.size()) {
      return xla::FailedPrecondition("Mismatched number of inputs.");
    }
    for (size_t i = 0; i < inst_inputs.size(); i++) {
      if (subcomp_visitor->InputIsUsed(o, i)) {
        if (comp_inputs[i].IsTensor()) {
          seq.add(
              poplar::program::Copy(inst_inputs[i], comp_inputs[i].AsTensor()));
        } else if (comp_inputs[i].IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Unable to handle used remote buffer tensor in function call "
              "instruction %s",
              inst->name());
        } else {
          return xla::FailedPrecondition(
              "Unknown output type in function call instruction `%s`, expected "
              "a tensor",
              inst->name());
        }
      }
    }
  }

  // Add the function.
  seq.add(subcomp_visitor->GetSequence());

  // Propagate the outputs.
  for (size_t i = 0; i < subcomp_visitor->outputs().size(); i++) {
    if (subcomp_visitor->outputs()[i].IsTensor()) {
      auto name = StrCat(GetDebugName(inst), "_out_", i);
      poplar::Tensor output = poputil::duplicate(
          graph, subcomp_visitor->outputs()[i].AsTensor(), seq, name,
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output));
    } else if (subcomp_visitor->outputs()[i].IsRemoteBuffer()) {
      return xla::FailedPrecondition(
          "Remote buffer output type in function instruction `%s`, expected a "
          "tensor.",
          inst->name());
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in function instruction `%s`, expected a "
          "tensor.",
          inst->name());
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreatePipelineOp(CompilerResources& res,
                                                    const HloInstruction* inst,
                                                    const xla::Shape& output,
                                                    TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  // Get all the inputs.
  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));

  // Call the create op with a deferred version of inputs.
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  TF_ASSIGN_OR_RETURN(
      poplar::program::Sequence pipeline_seq,
      CreatePipelineOp(res, inst, deferred_inputs, output, tensor_map));
  seq.add(pipeline_seq);

  return seq;
}

StatusOr<poplar::program::Program> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence seq;
  HloComputation* pipeline_computation = inst->to_apply();
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                      inst->backend_config<PoplarBackendConfig>());
  int64 pipeline_depth = cfg.call_config().pipeline_config().pipeline_depth();
  int64 repeat_count = cfg.call_config().pipeline_config().repeat_count();

  CHECK_EQ(inputs.size(), inst->operand_count());

  // Compile the pipeline.
  PipelineVisitor visitor(inst, res, inputs, GetDebugName(inst));
  auto order = pipeline_computation->parent()
                   ->schedule()
                   .sequence(pipeline_computation)
                   .instructions();

  TF_RETURN_IF_ERROR(pipeline_computation->AcceptOrdered(&visitor, order));

  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(visitor.PropagateDeferredAllocations(inst));

  // Make sure that inputs/outputs alias each other.
  TF_ASSIGN_OR_RETURN(auto pipeline_state,
                      visitor.AddLoopInputOutputAliasingCopies(
                          graph, pipeline_computation, GetDebugName(inst)));
  ExecutionCounters& execution_counters = visitor.GetExecutionCounters();
  // Initialize the counters.
  TF_RETURN_IF_ERROR(
      CopyExecutionCountersFromScope(res, execution_counters, seq));

  // Get the pipeline sequence.
  TF_ASSIGN_OR_RETURN(poplar::program::Sequence pipeline_prog,
                      visitor.GetPipelineSequence(pipeline_depth));
  // Increase the counters at the end of each pipeline execution.
  pipeline_prog.add(execution_counters.IncrementLiveCounters());

  seq.add(poplar::program::Repeat(repeat_count, pipeline_prog));

  for (size_t i = 0; i < pipeline_state.size(); i++) {
    if (pipeline_state[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, pipeline_state[i]));
    } else if (pipeline_state[i].IsRemoteBuffer()) {
      TF_CHECK_OK(
          AddOutputRemoteBuffer(tensor_map, inst, i, pipeline_state[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in pipeline instruction `%s`, expected a tensor "
          "or a remote buffer.",
          inst->name());
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

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

  TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                      FindInplaceOutputs(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
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
    poplar::Tensor out = graph.clone(bodies[0]->outputs()[i]);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
    outputs.push_back(out);
  }

  // Create the full program sequences for each branch
  std::vector<poplar::program::Sequence> seqs(bodies.size());
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
          seqs[b].add(
              poplar::program::Copy(inputs[b + 1][i].AsTensor(),
                                    bodies[b]->inputs()[0][i].AsTensor()));
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
                                          outputs[i]));
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
        popops::allTrue(graph, pred, seq, GetDebugName(inst));

    seq.add(poplar::program::If(scalar_pred, seqs[0], seqs[1]));
  } else {
    std::vector<std::pair<int32, poplar::program::Program>> cases;
    for (int64 c = 0; c < static_cast<int64>(seqs.size()) - 1; c++) {
      cases.push_back(std::make_pair(c, seqs[c]));
    }
    seq.add(poplar::program::Switch(pred, cases, seqs.back()));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateResourceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map) {
  HloComputation* resource_update_comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : "
          << resource_update_comp->name() << " as a resource update.";
  poplar::program::Sequence seq;
  // Create a visitor for the resource update.
  InplaceDeferredVisitor visitor(res, inputs, GetDebugName(inst));
  auto order = resource_update_comp->parent()
                   ->schedule()
                   .sequence(resource_update_comp)
                   .instructions();
  TF_RETURN_IF_ERROR(resource_update_comp->AcceptOrdered(&visitor, order));
  // Make sure any deferred inputs to the instruction are pushed up.
  TF_RETURN_IF_ERROR(visitor.PropagateDeferredAllocations(inst));
  // Add to the sequence.
  seq.add(visitor.GetSequence());
  seq.add(visitor.GetExecutionCounters().IncrementLiveCounters());
  // Set up the outputs.
  const TensorOrRemoteBufferVector& outputs = visitor.outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].IsTensor()) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
    } else if (outputs[i].IsRemoteBuffer()) {
      TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, i, outputs[i]));
    } else {
      return xla::FailedPrecondition(
          "Unknown output type in resource update instruction `%s`, expected a "
          "tensor or a remote buffer.",
          inst->name());
    }
  }

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
