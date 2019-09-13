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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>

using ::tensorflow::str_util::Join;

namespace se = ::stream_executor;
namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

FullVisitor::FullVisitor(CompilerResources& res) : BaseVisitor(res) {}

Status FullVisitor::HandleConcatenate(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  int64 dimension(inst->concatenate_dimension());
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), inst->operand_count());

  std::vector<poplar::Tensor> tensors(inputs.size());
  absl::c_transform(inputs, tensors.begin(), [](const ArgVector& ts) {
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

Status FullVisitor::HandleConvolution(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateConv2D(resources_, inst, GetOutputShape(inst), tensor_map));
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
      ArgVectors inputs,
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
  if (IsReducableArtithmetic(inst->to_apply())) {
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
      ArgVectors inputs,
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
      ArgVectors inputs,
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
      ArgVectors inputs,
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
      ArgVectors inputs,
      FindInplaceOutputTensors(tensor_map, resources_, inst, sequence, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];

  auto optional_begin =
      convert_array<std::vector<size_t>>(inst->slice_starts());
  if (!optional_begin) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice starts.");
  }
  std::vector<size_t> begin = *optional_begin;

  auto optional_end = convert_array<std::vector<size_t>>(inst->slice_limits());
  if (!optional_end) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice limits.");
  }
  std::vector<size_t> end = *optional_end;

  std::vector<int64> strides(inst->slice_strides());
  bool simple(true);
  for (std::size_t s : strides) {
    simple &= (s == 1);
  }
  if (simple) {
    out = out.slice(begin, end);
  } else {
    for (size_t d = 0; d < strides.size(); d++) {
      int64 s = strides[d];
      if (s > 0) {
        out = out.slice(begin[d], end[d], d);
        out = out.subSample(strides[d], d);
      } else {
        out = out.slice(end[d] + 1, begin[d] + 1, d);
        out = out.reverse(d);
        out = out.subSample(-strides[d], d);
      }
    }
  }
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
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
  if (IsReducableArtithmetic(inst->to_apply())) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateSimpleWindowReduction(resources_, inst, GetOutputShape(inst),
                                    tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleScatter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::program::Program prog;
  if (IsMultiUpdate(inst)) {
    TF_ASSIGN_OR_RETURN(
        prog, CreateMultiUpdate(resources_, Cast<HloScatterInstruction>(inst),
                                tensor_map));
  } else if (IsMultiUpdateAdd(inst)) {
    TF_ASSIGN_OR_RETURN(
        prog, CreateMultiUpdateAdd(
                  resources_, Cast<HloScatterInstruction>(inst), tensor_map));
  } else {
    TF_ASSIGN_OR_RETURN(
        prog, CreateScatter(resources_, Cast<HloScatterInstruction>(inst),
                            tensor_map));
  }

  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  if (IsSimpleSelection(inst->select()) &&
      IsReducableArtithmetic(inst->scatter())) {
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
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateWhileOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandlePad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
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

Status FullVisitor::HandleBatchNormInference(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormInf(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleBatchNormTraining(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormTraining(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleBatchNormGrad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormGrad(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::Postprocess(HloInstruction* inst) {
  if (!(inst->shape().IsTuple() || inst->shape().IsToken())) {
    auto outs = FindInstructionOutputs(tensor_map, inst);
    if (outs.size() == 1) {
      if (!PoplarShapeMatchesXLAShape(outs[0], inst->shape())) {
        return xla::InternalError(
            "Instruction %s has mismatched Poplar (%s) and XLA (%s) shapes",
            inst->name().c_str(), Join(outs[0].shape(), ",").c_str(),
            Join(inst->shape().dimensions(), ",").c_str());
      }
      TF_ASSIGN_OR_RETURN(poplar::Type expected_type,
                          PoplarDataType(inst->shape()));
      if (expected_type != outs[0].elementType()) {
        return xla::InternalError(
            "Instruction %s has mismatched Poplar (%s) and XLA (%s) type",
            inst->name().c_str(),
            outs[0].elementType().toString().cloneAsString().c_str(),
            expected_type.toString().cloneAsString().c_str());
      }
    }
  }
  return Status::OK();
}

Status FullVisitor::HandleGather(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(
      auto prog,
      CreateGather(resources_, Cast<HloGatherInstruction>(inst), tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleOutfeed(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (resources_.annotations.outfeed_infos.size()) {
    return InvalidArgument("Only one IPUOutfeedQueue supported per graph.");
  }

  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(resources_, inst);

  HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(inst);
  xla::poplarplugin::PoplarFeedConfig outfeed_config;
  outfeed_config.ParseFromString(outfeed->outfeed_config());

  size_t io_batch_size = std::max<size_t>(1, outfeed_config.io_batch_size());

  // Check that the replication factor matches.
  if (resources_.replication_factor != outfeed_config.replication_factor()) {
    return xla::FailedPrecondition(
        "Current program has been created with replication_factor %d, however "
        "the IPUOutfeedQueue has been configured with replication_factor %d. "
        "Either reduce the number of IPUs in your TensorFlow device, or set "
        "the `replication_factor` to %d when creating IPUOutfeedQueue.",
        resources_.replication_factor, outfeed_config.replication_factor(),
        resources_.replication_factor);
  }

  // operand 1 is the input
  // operand 2 is the token
  if (outfeed->operand_count() != 2) {
    return InvalidArgument("Expected operand_count() == 2 for outfeed ops");
  }

  if (UseSyntheticData()) {
    return Status::OK();
  }

  HloInstruction* operand = outfeed->operands()[0];
  const Shape& shape = operand->shape();
  if (ShapeUtil::IsNestedTuple(shape)) {
    return InvalidArgument("Nested tuple shapes are not supported for outfeed");
  }

  ArgVector input_tensors;
  const bool expand_constants = true;
  if (shape.IsTuple()) {
    input_tensors = FindInstructionInputs(tensor_map, resources_, inst, 0, seq,
                                          expand_constants);
  } else {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in,
        FindInstructionInput(tensor_map, resources_, inst, 0, seq));
    input_tensors.emplace_back(in);
  }

  for (unsigned i = 0; i < input_tensors.size(); ++i) {
    poplar::Tensor& in = input_tensors[i];

    if (io_batch_size == 1) {
      // Simply copy to the stream
      auto fifo =
          graph.addDeviceToHostFIFO(GetOutfeedCopyHandle(inst->name(), i),
                                    in.elementType(), in.numElements());

      seq.add(poplar::program::Copy(in, fifo, false));
    } else {
      // Batch multiple writes, and then write as a block

      // Extend the old shape to add a new dimension for the batches of memory
      std::vector<size_t> new_shape = in.shape();
      new_shape.insert(new_shape.begin(), io_batch_size);

      std::vector<poplar::Tensor> cloned_tensors(io_batch_size);
      for (int i = 0; i < io_batch_size; ++i) {
        if (resources_.always_rearrange_copies_on_host) {
          // When rearranging on the host it is better to have the slices of the
          // buffer laid out in the same form as the 'in' tensor so that there
          // is no cost of rearrangement.
          cloned_tensors[i] = graph.clone(in);
        } else {
          // When the data is rearranged on the device, it is beter to have the
          // slices arranged in the standard order of the host buffer, and then
          // to have the rearragement done only once, during the dynamicUpdate.
          cloned_tensors[i] =
              graph.addVariable(in.elementType(), in.shape(),
                                poplar::VariableMappingMethod::LINEAR);
        }
      }
      poplar::Tensor batched =
          poplar::concat(cloned_tensors).reshape(new_shape);

      //  A counter for counting slots
      poplar::Tensor counter = graph.addVariable(
          poplar::UNSIGNED_INT, {}, poplar::VariableMappingMethod::LINEAR,
          GetDebugName(inst) + "/OutfeedCtr/" + std::to_string(i));
      resources_.zeroed_tensors.push_back(counter);

      // Use dynamic slice update to put the slices into the buffer
      popops::dynamicUpdate(graph, batched, in.expand({0}),
                            counter.reshape({1}), {0}, {1}, seq,
                            GetDebugName(inst) + "/Slice" + std::to_string(i));

      // Increment the counter by one.
      popops::mapInPlace(
          graph,
          pe::Rem(pe::Add(pe::_1, pe::Const(1)), pe::Const(io_batch_size)),
          {counter}, seq,
          GetDebugName(inst) + "/OutfeedCtrInc/" + std::to_string(i));

      // The body for copying to host and zeroing the counter.
      poplar::program::Sequence true_body;

      // Copy the data to the host
      if (!UseSyntheticData()) {
        auto fifo = graph.addDeviceToHostFIFO(
            GetOutfeedCopyHandle(outfeed->name(), i), batched.elementType(),
            batched.numElements());
        true_body.add(poplar::program::Copy(batched, fifo, false));
      }

      // The NOP body.
      poplar::program::Sequence false_body;

      // Check the counter doesn't equal
      poplar::Tensor predicate = popops::map(
          graph, pe::Equal(pe::_1, pe::Const(0)), {counter}, seq,
          GetDebugName(inst) + "/OutfeedCtrCmp/" + std::to_string(i));

      // The main body which contains the control flow for copy from host and
      // the dynamic slice.
      seq.add(poplar::program::If(predicate, true_body, false_body));
    }
  }

  FeedInfo info(outfeed->name(), outfeed_config,
                outfeed->operands()[0]->shape());
  resources_.annotations.outfeed_infos.push_back(info);
  sequence.add(seq);

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
