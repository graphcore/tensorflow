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

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class RemoteParameterLoadOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const int64 flat_tuple_index = 0;
    poplar::program::Sequence seq;
    const auto* load_inst = Cast<HloRemoteParameterLoad>(inst);

    TF_ASSIGN_OR_RETURN(poplar::Tensor tensor,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map));

    const auto& in_info =
        res.annotations.input_output_aliasing_map
            .GetEntryInputInfos()[load_inst->GetParameterNumber()];

    std::vector<Shape> module_shapes;

    const HloModule* module = inst->GetModule();
    const ComputationLayout layout = module->entry_computation_layout();
    if (layout.parameter_count() > load_inst->GetParameterNumber()) {
      const Shape& mod_shape =
          layout.parameter_shape(load_inst->GetParameterNumber());
      module_shapes = FlattenedXlaShape(mod_shape);
    }

    if (!UseSyntheticData()) {
      poplar::Tensor tensor_destination = tensor;
      if (!LayoutUtil::IsMonotonicWithDim0Major(
              module_shapes[flat_tuple_index].layout())) {
        // Host tensor needs to be host layout.
        tensor_destination =
            ConvertFromDeviceLayout(module_shapes[flat_tuple_index], tensor);
      }

      const std::string remote_buffer_name =
          res.annotations.input_output_aliasing_map
              .GetEntryInputInfos()[load_inst->GetParameterNumber()]
              .Handles()
              .at(flat_tuple_index);

      poplar::RemoteBuffer remote_buffer = graph.addRemoteBuffer(
          remote_buffer_name, tensor_destination.elementType(),
          tensor_destination.numElements(),
          /*repeats*/ 1, /*rearrangeOnHost*/ true);

      seq.add(poplar::program::Copy(remote_buffer, tensor_destination));

      CHECK(!res.remote_buffers.contains(remote_buffer_name));
      res.remote_buffers[remote_buffer_name] = remote_buffer;

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
    const int64 flat_tuple_index = 0;
    poplar::program::Sequence seq;
    const auto* store_inst = Cast<HloRemoteParameterStore>(inst);
    const uint64 output_idx = store_inst->GetOutputIndex();

    auto* layout = inst->GetModule()->mutable_entry_computation_layout();

    if (!UseSyntheticData()) {
      TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                          FindInplaceOutputTensors(tensor_map, res, inst, seq));
      CHECK_EQ(inputs.size(), 1);
      CHECK_EQ(inputs[0].size(), 1);
      poplar::Tensor tensor = inputs[0][0];

      const Shape output_shape =
          ShapeUtil::GetTupleElementShape(layout->result_shape(), output_idx);
      CHECK(!output_shape.IsTuple());

      poplar::Tensor out = ConvertFromDeviceLayout(output_shape, tensor);

      // Get the remote buffer.
      const auto& out_info = res.annotations.input_output_aliasing_map
                                 .GetEntryOutputInfos()[output_idx];

      const std::string remote_buffer_name =
          res.annotations.input_output_aliasing_map
              .GetEntryInputInfos()[out_info.GetInputIndex()]
              .Handles()
              .at(flat_tuple_index);

      poplar::RemoteBuffer& remote_buffer =
          res.remote_buffers.at(remote_buffer_name);

      seq.add(poplar::program::Copy(out, remote_buffer));
    }
    return seq;
  }
};
REGISTER_POPLAR_OP(RemoteParameterStore, RemoteParameterStoreOp);

class RemoteParameterDummyOutputOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // This op does not have an output.
    return poplar::program::Sequence();
  }
};
REGISTER_POPLAR_OP(RemoteParameterDummyOutput, RemoteParameterDummyOutputOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
