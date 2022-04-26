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

#include <poplar/DebugContext.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/collective_reorder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/partitioned_elementwise_cluster_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include <poputil/Broadcast.hpp>

#include <gcl/CollectiveBalancedReorder.hpp>

#include "absl/strings/str_cat.h"

namespace xla {
namespace poplarplugin {
namespace {

class CollectiveReorderOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CollectiveReorderOp");
    poplar::DebugNameAndId dnai(debug_info);
    DriverProgramSequence seq(graph, debug_info);

    CHECK(res.current_cluster_visitor)
        << "collective-reorder instruction must be used only inside "
           "partitioned cluster.";

    TF_ASSIGN_OR_RETURN(
        auto* cbr_info,
        res.current_cluster_visitor->GetCollectiveBalancedReorder(inst));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info, false));

    if (!cbr_info) {
      VLOG(2) << "Creating CBR instance for instruction " << inst->ToString();
      TF_RETURN_IF_ERROR(
          res.current_cluster_visitor->SetCollectiveBalancedReorder(
              inst, absl::make_unique<gcl::CollectiveBalancedReorder>(
                        graph, input, res.partition_replication_factor,
                        GetDebugNameAndId(res, inst), true)));
      TF_ASSIGN_OR_RETURN(
          cbr_info,
          res.current_cluster_visitor->GetCollectiveBalancedReorder(inst));
      CHECK_NE(cbr_info, nullptr);
    }

    auto* cbr = cbr_info->host_rearrangement.get();
    poplar::Tensor output = cbr->createCollectivesTensor(
        input.elementType(), debug_info.getPathName());
    VLOG(2) << "Created collectives tensor for input " << input.shapeToString()
            << " with shape " << output.shapeToString();

    // CBR collectives tensor could be bigger than reference tensor.
    // Fill collectives tensor with zeroes to fill the gaps.
    poplar::Tensor zero =
        graph.addConstant(input.elementType(), {}, 0, debug_info);
    graph.setTileMapping(zero, 0);
    poputil::broadcastToMatch(zero, output.shape());
    seq.add(poplar::program::Copy(zero, output, false, debug_info));

    // Undo rearrangement for collectives tensor and get a "view" with the shape
    // and order of the reference tensor.
    auto ref = cbr->undoRearrangeForCollective(output);

    // Copy input to the view. View is just a collection of collectives tensor
    // slices, so the input data appear in collectives tensor too.
    seq.add(poplar::program::Copy(input.flatten(), ref.flatten(), false,
                                  debug_info));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0,
                                DriverTensor(output.flatten(), graph)));

    return seq;
  }
};

REGISTER_POPLAR_OP(CollectiveReorder, CollectiveReorderOp);

class UndoCollectiveReorderOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "UndoCollectiveReorderOp");
    DriverProgramSequence seq(graph, debug_info);

    CHECK(res.current_cluster_visitor)
        << "undo-collective-reorder instruction must be used only inside "
           "partitioned cluster.";

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info, false));

    TF_ASSIGN_OR_RETURN(
        auto* cbr_info,
        res.current_cluster_visitor->GetCollectiveBalancedReorder(inst));
    if (!cbr_info) {
      return InternalError("Collective reorder instance was not created.");
    }
    auto output =
        cbr_info->host_rearrangement->undoRearrangeForCollective(input);

    output = output.reshape(PoplarShapeFromXlaShape(inst->shape()));

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(output, graph)));

    return seq;
  }
};

REGISTER_POPLAR_OP(UndoCollectiveReorder, UndoCollectiveReorderOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
