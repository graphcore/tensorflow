/* Copyright 2017 Graphcore Ltd
 */

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
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"
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
  bool new_stochastic_rounding_enabled;
  switch (poplar_backend_config.stochastic_rounding()) {
    case NOT_SET:
      new_stochastic_rounding_enabled =
          resources_.global_floating_point_behaviour.esr();
      break;
    case FORCE_ON:
      new_stochastic_rounding_enabled = true;
      break;
    case FORCE_OFF:
      new_stochastic_rounding_enabled = false;
      break;
    default:
      return InvalidArgument(
          "Invalid value for PoplarBackendConfig.stochastic_rounding()");
  }
  if (new_stochastic_rounding_enabled != stochastic_rounding_enabled_) {
    poplar::setStochasticRounding(GetGraph(resources_, inst), sequence,
                                  new_stochastic_rounding_enabled,
                                  "Preprocess");
    stochastic_rounding_enabled_ = new_stochastic_rounding_enabled;
  }
  return Status::OK();
}

BaseVisitor::BaseVisitor(CompilerResources& res) : resources_(res) {
  stochastic_rounding_enabled_ = res.global_floating_point_behaviour.esr();
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
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCastOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleTupleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateTupleSelectOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleBitcastConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Type type, PoplarDataType(inst->shape()));
  out = out.reinterpret(type);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status BaseVisitor::HandleAllReduce(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  auto reduction = inst->to_apply();
  auto reduction_root = reduction->root_instruction();

  if (reduction_root->opcode() != HloOpcode::kAdd) {
    return xla::FailedPrecondition(
        "Unsupported all-reduce reduction computation.");
  }

  for (auto& reduction_operand : reduction_root->operands()) {
    if (reduction_operand->opcode() != HloOpcode::kParameter) {
      return xla::FailedPrecondition(
          "Unsupported all-reduce reduction computation.");
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto seq, CreateReplicatedAllReduce(resources_, inst,
                                          GetOutputShape(inst), tensor_map));

  sequence.add(seq);
  return Status::OK();
}

Status BaseVisitor::HandleConstant(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::Graph& graph = GetGraph(resources_, inst);
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor t,
      AddConstantTensor(graph, TensorLocation{inst, 0}, GetOutputShape(inst),
                        inst->literal(), resources_, tensor_map));

  // If this constant is used inplace then we need to add a copy and use that
  // instead so the original constant value is always preserved.
  bool is_inplace_read_write = IsOutputModifiedInplace(inst);
  if (is_inplace_read_write && t.numElements() != 0) {
    VLOG(1) << "Constant tensor is read/write inplace, adding copy";
    poplar::program::Sequence prog;
    poplar::Tensor clone = poputil::duplicate(
        graph, t, prog, GetDebugName(inst) + ".clone",
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

    sequence.add(prog);
    t = clone;
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors output_tensors,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(output_tensors.size(), 1);
  CHECK_EQ(output_tensors[0].size(), CountShapes(inst->shape()));
  for (size_t i = 0; i < output_tensors[0].size(); i++) {
    poplar::Tensor out;
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, output_tensors[0][i]));
  }
  return Status::OK();
}

namespace {
TensorVectors GetFusionInputs(CompilerResources& res,
                              const HloInstruction* inst, TensorMap& tensor_map,
                              poplar::program::Sequence& seq,
                              const bool expand_aliasing = true) {
  TensorVectors args;
  for (int64 i = 0; i < inst->operand_count(); i++) {
    TensorVector t =
        FindInstructionInputs(tensor_map, res, inst, i, seq, expand_aliasing);
    args.push_back(t);
  }
  return args;
}
}  // namespace

Status BaseVisitor::HandleFusion(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->ToString();
  poplar::program::Program prog;
  HloComputation* comp = inst->fused_instructions_computation();

  if (IsArithmeticExpressionFusion(inst)) {
    TensorVectors args =
        GetFusionInputs(resources_, inst, tensor_map, sequence);
    ArithmeticExprVisitor arithmetic_visitor(resources_, args);
    TF_RETURN_IF_ERROR(comp->Accept(&arithmetic_visitor));
    prog = arithmetic_visitor.GetSequence();

    for (size_t i = 0; i < arithmetic_visitor.outputs().size(); i++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  arithmetic_visitor.outputs()[i]));
    }
  } else if (IsPopOpsFusion(inst)) {
    // Fusions are handle as Poplar custom ops.
    poplar::Graph& graph = GetGraph(resources_, inst);
    const bool is_poplar_custom_op = GetPoplarCustomOp(inst).has_value();
    if (is_poplar_custom_op) {
      TF_ASSIGN_OR_RETURN(
          prog, CreatePoplarOp(graph, resources_, inst, GetOutputShape(inst),
                               tensor_map));
    } else {
      return xla::FailedPrecondition("Unrecognised fusion instruction %s.",
                                     inst->ToString().c_str());
    }
  } else {
    TF_ASSIGN_OR_RETURN(prog, CreateFusionOp(resources_, inst,
                                             GetOutputShape(inst), tensor_map));
  }

  sequence.add(prog);
  return Status::OK();
};

Status BaseVisitor::HandleCall(HloInstruction* inst) {
  HloComputation* comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : " << comp->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCallOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleCustomCall(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCustomCallOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);

  return Status::OK();
}

Status BaseVisitor::HandleTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateTuple(resources_, inst, tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleMap(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(bool simple_parallel,
                      IsParallelMap(inst, inst->to_apply()));
  if (simple_parallel) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateParallelMap(resources_, inst, GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConditional(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateConditionalOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);

  return Status::OK();
}

Status BaseVisitor::HandleReal(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor in,
      FindInstructionInput(tensor_map, resources_, inst, 0, sequence));

  poplar::Tensor out = GetGraph(resources_, inst).clone(in);
  sequence.add(poplar::program::Copy(in, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return Status::OK();
}

Status BaseVisitor::HandleAllToAll(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(
      auto seq, CreateReplicatedAllToAll(resources_, inst, GetOutputShape(inst),
                                         tensor_map));

  sequence.add(seq);
  return Status::OK();
}

Status BaseVisitor::HandleAddDependency(HloInstruction* inst) {
  std::vector<std::string> dep_names;
  GetAllDepNames(inst->operand(1), dep_names);

  VLOG(1) << "Processing " << inst->name() << " on "
          << absl::StrJoin(dep_names, ",");
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), CountShapes(inst->operand(0)->shape()));
  for (size_t idx = 0; idx < inputs[0].size(); idx++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, idx, inputs[0][idx]));
  }
  return Status::OK();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s (%s) not implemented", inst->name().c_str(),
                            HloOpcodeString(inst->opcode()).c_str());
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
}  // namespace poplarplugin
}  // namespace xla
