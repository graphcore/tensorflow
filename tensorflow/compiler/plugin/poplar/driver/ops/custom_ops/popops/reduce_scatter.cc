/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gcl/Collectives.hpp>
#include <poplar/DebugContext.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/reduce_scatter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/reduction_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class ReduceScatterOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "ReduceScatterOp");
    DriverProgramSequence seq(debug_info);

    const auto* reduce_scatter_inst = Cast<HloReduceScatterInstruction>(inst);
    const auto replica_groups = reduce_scatter_inst->GetPoplarReplicaGroups();

    // Collect all the inputs.
    const int64_t num_inputs = inst->operand_count();
    std::vector<poplar::Tensor> inputs(num_inputs);
    for (int64_t i = 0; i < num_inputs; ++i) {
      TF_ASSIGN_OR_RETURN(
          inputs[i],
          FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));
      // The op requires rank-1 input.
      CHECK_EQ(inputs[i].rank(), 1);
    }

    TF_ASSIGN_OR_RETURN(auto op,
                        ToPoplarCollectiveOperator(
                            reduce_scatter_inst->GetCollectiveOperator()));

    TF_ASSIGN_OR_RETURN(const auto gcl_comm_group,
                        ToGclCommGroup(replica_groups, res));

    // Call overload of reduce scatter which accepts a vector of inputs.
    std::vector<poplar::Tensor> outputs = gcl::reduceScatterCrossReplica(
        graph, inputs, op, seq, gcl_comm_group, {debug_info, "ReduceScatter"},
        GetReplicatedCollectiveOptions(res));

    for (int64_t i = 0; i != outputs.size(); ++i) {
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, i, DriverTensor(outputs[i])));
    }

    return seq;
  }
};

REGISTER_POPLAR_OP(ReduceScatter, ReduceScatterOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
