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
    poplar::program::Sequence seq;
    const auto* load_inst = Cast<HloRemoteParameterLoad>(inst);

    TF_ASSIGN_OR_RETURN(poplar::Tensor tensor,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    if (!UseSyntheticData()) {
      TensorOrRemoteBufferVector inputs =
          FindInstructionInputs(tensor_map, res, inst, 0, seq, true);

      poplar::RemoteBuffer remote_buffer = inputs[0].AsRemoteBuffer();

      seq.add(poplar::program::Copy(remote_buffer, tensor));
    } else if (UseSyntheticData() && UseSyntheticDataInitializer()) {
      // Initialize the tensor to a constant value.
      auto& initializer = DataInitializer::GetSyntheticDataInitializer();
      TF_ASSIGN_OR_RETURN(auto literal, initializer.GetData(output_shape));
      TF_RETURN_IF_ERROR(SetInitialTensorValue(graph, tensor, literal));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, tensor));
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
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(TensorOrRemoteBufferVectors inputs,
                        FindInplaceOutputs(tensor_map, res, inst, seq));

    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::RemoteBuffer remote_buffer = inputs[0][0].AsRemoteBuffer();

    if (!UseSyntheticData()) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor tensor,
                          FindInstructionInput(tensor_map, res, inst, 1, seq));

      seq.add(poplar::program::Copy(tensor, remote_buffer));
    }
    TF_CHECK_OK(AddOutputRemoteBuffer(tensor_map, inst, 0, remote_buffer));
    return seq;
  }
};
REGISTER_POPLAR_OP(RemoteParameterStore, RemoteParameterStoreOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
