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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class RemoteParameterLoadOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    VLOG(1) << "Processing " << GetDebugName(inst);
    const auto* load_inst = Cast<HloRemoteParameterLoad>(inst);
    const int64 num_inputs = inst->operand_count();

    const auto shapes = output_shape.IsTuple()
                            ? output_shape.tuple_shapes()
                            : std::vector<xla::Shape>{output_shape};
    CHECK_EQ(shapes.size(), num_inputs);

    if (load_inst->GetReplicationFactor() != res.replication_factor) {
      return xla::FailedPrecondition(
          "RemoteBuffer load instruction replication factor doesn't match "
          "graph replication factor.");
    }

    poplar::program::Sequence seq;

    for (int64 i = 0; i < num_inputs; ++i) {
      poplar::Graph& shard_graph = GetGraphWithOutputIndex(res, inst, i);
      const Shape& shape = shapes[i];
      TF_ASSIGN_OR_RETURN(poplar::Tensor tensor,
                          AddTensor(shard_graph, TensorLocation{inst, i}, shape,
                                    res, tensor_map));

      if (!UseSyntheticData()) {
        TensorOrRemoteBufferVector inputs =
            FindInstructionInputs(tensor_map, res, inst, i, seq, true);

        CHECK_EQ(inputs.size(), 1);

        if (!inputs[0].IsRemoteBuffer()) {
          return xla::FailedPrecondition(
              "Expected a Poplar RemoteBuffer as operand %d to %s", i,
              GetDebugName(inst));
        }

        poplar::RemoteBuffer remote_buffer = inputs[0].AsRemoteBuffer();

        seq.add(poplar::program::Copy(remote_buffer, tensor));
      } else if (UseSyntheticData() && UseSyntheticDataInitializer()) {
        // Initialize the tensor to a constant value.
        auto& initializer = DataInitializer::GetSyntheticDataInitializer();
        TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(shape));
        TF_RETURN_IF_ERROR(SetInitialTensorValue(shard_graph, tensor, literal));
      }

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, tensor));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(RemoteParameterLoad, RemoteParameterLoadOp);

class RemoteParameterStoreOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    VLOG(1) << "Processing " << GetDebugName(inst);

    const int64 num_inputs = inst->operand_count();
    CHECK_EQ(num_inputs % 2, 0);
    const int64 num_outputs = num_inputs / 2;

    const auto shapes = output_shape.IsTuple()
                            ? output_shape.tuple_shapes()
                            : std::vector<xla::Shape>{output_shape};
    CHECK_EQ(shapes.size(), num_outputs);

    const auto* store_inst = Cast<HloRemoteParameterStore>(inst);

    if (store_inst->GetReplicationFactor() != res.replication_factor) {
      return xla::FailedPrecondition(
          "RemoteBuffer store instruction replication factor doesn't match "
          "graph replication factor.");
    }

    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors outputs,
                        FindInplaceOutputs(tensor_map, res, inst, seq));
    CHECK_EQ(outputs.size(), num_outputs);

    for (int64 i = 0; i < num_outputs; ++i) {
      CHECK_EQ(outputs[i].size(), 1);

      if (!outputs[i][0].IsRemoteBuffer()) {
        return xla::FailedPrecondition(
            "Expected a Poplar RemoteBuffer as operand %d to %s", i,
            GetDebugName(inst));
      }

      poplar::RemoteBuffer remote_buffer = outputs[i][0].AsRemoteBuffer();

      if (!UseSyntheticData()) {
        TF_ASSIGN_OR_RETURN(poplar::Tensor tensor,
                            FindInstructionInput(tensor_map, res, inst,
                                                 outputs.size() + i, seq));

        seq.add(poplar::program::Copy(tensor, remote_buffer));
      }
      TF_CHECK_OK(AddOutput(tensor_map, inst, i, outputs[i][0]));
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(RemoteParameterStore, RemoteParameterStoreOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
