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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_full.h"

#include <stddef.h>
#include <string.h>

#include <map>
#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/rnn.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/stream_executor/lib/initialize.h"

using ::tensorflow::str_util::Join;

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

FullVisitor::FullVisitor(CompilerResources& res, const std::string& name)
    : BaseVisitor(res, name) {}

Status FullVisitor::HandleConcatenate(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  int64 dimension(inst->concatenate_dimension());
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), inst->operand_count());

  std::vector<poplar::Tensor> tensors(inputs.size());
  absl::c_transform(inputs, tensors.begin(), [](const TensorVector& ts) {
    CHECK_EQ(ts.size(), 1);
    return ts[0];
  });
  poplar::Tensor out = poplar::concat(tensors, dimension);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDot(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateMatMulForDotOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleCopy(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCopy(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleReverse(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor t = inputs[0][0];

  TF_ASSIGN_OR_RETURN(t, ReverseTensor(t, inst->dimensions()));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status FullVisitor::HandleReduce(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsReducibleArithmetic(inst->to_apply())) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateSimpleReduction(resources_, inst, GetOutputShape(inst),
                              tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleBroadcast(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  TF_ASSIGN_OR_RETURN(
      out, BroadcastTensor(out, GetOutputShape(inst), inst->dimensions()));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleReshape(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleTranspose(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  auto optional_permutation =
      convert_array<std::vector<unsigned>>(inst->dimensions());
  if (!optional_permutation) {
    return xla::FailedPrecondition(
        "HandleTranspose - cannot cast permutation.");
  }
  std::vector<unsigned> permutation = *optional_permutation;
  out = out.dimShuffle(permutation);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateSlice(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleDynamicSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateDynamicSliceOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleDynamicUpdateSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateDynamicUpdateSliceOp(
                          resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReduceWindow(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsPoplibsPool(inst, inst->to_apply())) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreatePoplibsWindowReduction(resources_, inst, GetOutputShape(inst),
                                     tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  if (IsReducibleArithmetic(inst->to_apply())) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateSimpleWindowReduction(resources_, inst, GetOutputShape(inst),
                                    tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  if (IsSimpleSelection(inst->select()) &&
      IsReducibleArithmetic(inst->scatter())) {
    VLOG(1) << "Processing " << inst->name();
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateSimpleSelectAndScatter(resources_, inst, GetOutputShape(inst),
                                     tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleWhile(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // Version of the while operation which does not allow parameters to be
  // deferred.
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateWhileOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandlePad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      TensorVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(inputs[0].size(), 1);
  CHECK_EQ(inputs[1].size(), 1);
  poplar::Tensor out = inputs[0][0];
  poplar::Tensor pad = inputs[1][0];
  TF_ASSIGN_OR_RETURN(out, PadTensor(inst->padding_config(), out, pad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleIota(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog, CreateIota(resources_, inst,
                                            GetOutputShape(inst), tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleSort(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog, CreateSort(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleOutfeed(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(auto prog, CreateOutfeed(resources_, inst, tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::Postprocess(HloInstruction* inst) {
  std::size_t next_tuple_index = 0;
  for (auto indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
    const std::size_t tuple_index = next_tuple_index++;
    const Shape shape = indexed_shape.shape;
    const ShapeIndex index = indexed_shape.index;

    if (shape.IsToken()) {
      continue;
    }
    // If the current location is a deferred location, then skip this.
    if (DeferredAllocations::IsDeferredAllocationLocation(
            resources_, {inst, tuple_index})) {
      continue;
    }

    // Handle special cases which do not have outputs at certain locations.
    switch (inst->opcode()) {
      case HloOpcode::kRecv: {
        continue;
      }
      case HloOpcode::kSend: {
        if (tuple_index > 0) {
          continue;
        }
        break;
      }
      case HloOpcode::kCustomCall: {
        const int64 output_idx = index[0];
        // Inference RNNs don't output intermediates.
        const bool is_fwd_lstm =
            IsPoplarInstruction(PoplarOp::LstmLayerFwd)(inst);
        const bool is_fwd_gru =
            IsPoplarInstruction(PoplarOp::GRULayerFwd)(inst);
        const int64 intermediates_idx = is_fwd_gru ? 2 : 3;
        if (is_fwd_lstm || is_fwd_gru) {
          auto rnn_inst = Cast<HloRNNFwdInstruction>(inst);
          if (!rnn_inst->is_training() && output_idx == intermediates_idx) {
            continue;
          }
        }

        // Dummy operation doesn't output tensors.
        if (IsPoplarInstruction(PoplarOp::RemoteParameterDummyOutput)(inst)) {
          CHECK(IsInstructionInEntryComputation(inst));
          continue;
        }
        break;
      }
      case HloOpcode::kParameter: {
        if (IsRemoteParameter(inst, resources_)) {
          CHECK(IsInstructionInEntryComputation(inst));
          // Remote parameters have no outputs.
          CHECK_EQ(inst->user_count(), 0);
          return Status::OK();
        }
        break;
      }
      case HloOpcode::kTuple: {
        // Get the operand index to the tuple instruction given the shape index.
        const int64 operand_idx = index[0];
        // If the input at this tuple index is RemoteParameterDummyOutput then
        // we can skip it.
        if (IsPoplarInstruction(PoplarOp::RemoteParameterDummyOutput)(
                inst->operand(operand_idx))) {
          continue;
        }
        break;
      }
      default: { break; }
    }

    // Find the tensor for this output location.
    auto outs = FindInstructionOutputsInRange(tensor_map, resources_, inst,
                                              {tuple_index, tuple_index + 1});
    CHECK_EQ(outs.size(), 1);
    auto& out = outs[0];
    if (!PoplarShapeMatchesXLAShape(out, shape)) {
      return xla::InternalErrorStrCat(
          "Instruction ", inst->name(), " has mismatched Poplar (",
          Join(out.shape(), ","), ") and XLA (", Join(shape.dimensions(), ","),
          ") shapes.");
    }
    TF_ASSIGN_OR_RETURN(poplar::Type expected_type, PoplarDataType(shape));
    if (expected_type != out.elementType()) {
      return xla::InternalErrorStrCat(
          "Instruction ", inst->name(), " has mismatched Poplar (",
          out.elementType().toString().cloneAsString(), ") and XLA (",
          expected_type.toString().cloneAsString(), ") type.");
    }
  }
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
