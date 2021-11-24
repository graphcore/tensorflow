/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <poputil/TileMapping.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/DebugContext.hpp>

namespace xla {
namespace poplarplugin {
namespace {
class UserOpImpl : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "UserOpImpl");
    const HloUserOpInstruction* user_op_inst = Cast<HloUserOpInstruction>(inst);

    const std::string& gp_path = user_op_inst->GetPath();

    const bool is_gradient = user_op_inst->IsGradient();

    // Add the codelet if needed.
    if (!gp_path.empty() && !res.custom_codelets_in_graph.contains(gp_path)) {
      graph.addCodelets(gp_path);
      res.custom_codelets_in_graph.insert(gp_path);
    }

    // Get the function pointer from the HLO.
    poplar::program::Program (*as_function_ptr)(
        poplar::Graph&, const std::vector<poplar::Tensor>&,
        std::vector<poplar::Tensor>& outputs, const std::string& attributes,
        const std::string& debugPrefix);

    // Get the function pointer from the HLO.
    poplar::program::Program (*as_function_ptr_gradient)(
        poplar::Graph&, int64 partial_derivative_index,
        const std::vector<poplar::Tensor>& gradients_in,
        const std::vector<poplar::Tensor>& fwd_outputs,
        const std::vector<poplar::Tensor>& fwd_inputs,
        std::vector<poplar::Tensor>& outputs, const std::string& attributes,
        const std::string& debugPrefix);

    // We have a special function pointer type for when the intent is to execute
    // the operation on the host by reading the raw bytes, executing the on the
    // host, then copying back.
    void (*as_function_host_rw_ptr)(
        const std::vector<void*>& data,
        const std::vector<std::uint32_t>& number_of_elements,
        std::vector<void*>& outputs, const std::string& attributes,
        const std::string& debugPrefix);

    // Convert the function pointer to each of the types of function we could
    // have.
    as_function_ptr = reinterpret_cast<decltype(as_function_ptr)>(
        user_op_inst->GetPointerToFunc());
    as_function_ptr_gradient =
        reinterpret_cast<decltype(as_function_ptr_gradient)>(
            user_op_inst->GetPointerToFunc());
    as_function_host_rw_ptr =
        reinterpret_cast<decltype(as_function_host_rw_ptr)>(
            user_op_inst->GetPointerToFunc());
    poplar::program::Sequence seq({}, debug_info);
    std::vector<poplar::Tensor> outputs;

    // Track the number of outputs/inputs this operation has.
    const std::uint32_t number_of_outputs = output_shape.tuple_shapes_size();
    const std::uint32_t number_of_inputs = user_op_inst->NumInputs();

    // If this is a user operation we have to copy over all the buffers.
    const bool is_user_read_write = user_op_inst->IsReadWrite();

    // If this op is a gradient and the gradients are separate, then this
    // is the index of the input for which this grad op is the partial
    // derivative.
    const int pdi = user_op_inst->PartialDerivativeIndex();

    // Pass in any attributes from the user.
    const std::string attributes = user_op_inst->GetAttributes();

    // We use the instruction name to keep track of which buffers are allocated
    // to which user op.
    const std::string instruction_name = GetDebugName(inst);

    // If this is a user op copy the buffers. At this stage we add the copy
    // operations to the graph and allocate the streams. To connect the actual
    // stream we create the StreamCopyInfo structures to communicate with
    // poplar_executor which does the actual linking of the streams.
    if (is_user_read_write) {
      // A wrapper around the user functor which we finally call down to.
      auto functor = [=](std::vector<void*>& data,
                         std::vector<std::uint32_t>& number_of_elements,
                         std::vector<void*>& outputs) {
        as_function_host_rw_ptr(data, number_of_elements, outputs, attributes,
                                instruction_name);
      };

      // We add a map of user ops to their owned streams.
      res.annotations.stream_infos.insert({instruction_name, {}});
      std::list<StreamCopyInfo>& info_list =
          res.annotations.stream_infos[instruction_name];

      // Allocate a stream info
      res.annotations.stream_meta_infos[instruction_name] = {inst,
                                                             number_of_inputs};
      StreamCopyMetaInfo& meta_info =
          res.annotations.stream_meta_infos[instruction_name];

      for (std::uint32_t i = 0; i < user_op_inst->NumInputs(); ++i) {
        // Get the poplar tensor.
        TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                            FindInstructionInput(tensor_map, res, inst, i, seq,
                                                 {debug_info}, false));

        // Give each input a stream identifier based on the instruction name.
        const std::string stream_name =
            instruction_name + "_read_" + std::to_string(i);

        // Create a datastream for the input tensor.
        poplar::DataStream stream = graph.addDeviceToHostFIFO(
            stream_name, in.elementType(), in.numElements());

        // Allocate this structure to communicate to the executor, which
        // callbacks to register to which input tensors.
        const uint32_t num_elements = static_cast<uint32_t>(in.numElements());
        const uint32_t type_size = static_cast<uint32_t>(
            graph.getTarget().getTypeSize(in.elementType()));
        StreamCopyInfo info{inst,      stream_name, num_elements,
                            type_size, i,           functor};
        info_list.push_back(info);

        // Copy from the tensor into the host stream. We will later attach a
        // callback to this.
        seq.add(poplar::program::Copy(in, stream, false, {debug_info}));
      }

      // Add an ontile sync to stop the copies from host being merged with the
      // above as there is an invisble dependency in the callback.
      seq.add(poplar::program::Sync(poplar::SyncType::INTERNAL, {debug_info}));

      outputs.resize(number_of_outputs);

      // Now go over and add a copy from the device back to the host for each
      // output.
      for (std::uint32_t output_index = 0; output_index != number_of_outputs;
           output_index++) {
        xla::Shape shape = output_shape.tuple_shapes()[output_index];
        // Create a new tensor using "AddTensor" to get a good layout.
        TF_ASSIGN_OR_RETURN(
            poplar::Tensor output_tensor,
            AddTensor(graph, TensorLocation{inst, output_index}, shape, res,
                      tensor_map, {debug_info, "output"}));

        // Add stream ID for each output tensor.
        const std::string stream_name =
            instruction_name + "_write_" + std::to_string(output_index);

        // Copy from the host into these new tensors.
        poplar::DataStream stream =
            graph.addHostToDeviceFIFO(stream_name, output_tensor.elementType(),
                                      output_tensor.numElements());

        // Allocate this structure to communicate to the executor so the
        // executor knows how much memory to allocate for the callback to write
        // into.
        const uint32_t num_elements =
            static_cast<uint32_t>(output_tensor.numElements());
        const uint32_t type_size = static_cast<uint32_t>(
            graph.getTarget().getTypeSize(output_tensor.elementType()));
        StreamCopyInfo info{inst, stream_name, num_elements, type_size,
                            output_index};
        info_list.push_back(std::move(info));

        // Store a reference to this stream copy info.
        StreamCopyInfo* ref = &info_list.back();
        meta_info.output_stream_info.push_back(ref);

        // Add the copy to the graph.
        seq.add(
            poplar::program::Copy(stream, output_tensor, false, {debug_info}));

        outputs[output_index] = output_tensor;
      }
    } else {
      if (!is_gradient) {
        std::vector<poplar::Tensor> inputs(user_op_inst->NumInputs());

        // Get the variadic inputs and store them in the vector.
        for (size_t i = 0; i < user_op_inst->NumInputs(); ++i) {
          TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                              FindInstructionInput(tensor_map, res, inst, i,
                                                   seq, {debug_info}, false));
          inputs[i] = in;
        }
        // Call the user operation and add it to the sequence.
        seq.add(as_function_ptr(graph, inputs, outputs, attributes,
                                instruction_name));
      } else {
        // There is a gradient for each output and if we are doing the backward
        // pass then the input will be packed like: | Gradients |
        // previous_outputs | previous_inputs |
        const ssize_t size_of_gradient = user_op_inst->GetGradientSize();
        const ssize_t outputs_index = size_of_gradient * 2;

        if (size_of_gradient <= 0) {
          return xla::InternalErrorStrCat("Instruction ", instruction_name,
                                          " has wrong gradient size",
                                          size_of_gradient);
        }

        std::vector<poplar::Tensor> gradients;
        std::vector<poplar::Tensor> previous_outputs;
        std::vector<poplar::Tensor> previous_inputs;

        // Get the gradients.s
        for (ssize_t i = 0; i < size_of_gradient; ++i) {
          TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                              FindInstructionInput(tensor_map, res, inst, i,
                                                   seq, {debug_info}, false));

          gradients.push_back(in);
        }

        // Get the previous inputs.
        for (ssize_t i = size_of_gradient; i < outputs_index; ++i) {
          TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                              FindInstructionInput(tensor_map, res, inst, i,
                                                   seq, {debug_info}, false));
          previous_outputs.push_back(in);
        }

        // Get the previous outputs.
        for (size_t i = outputs_index; i < user_op_inst->NumInputs(); ++i) {
          TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                              FindInstructionInput(tensor_map, res, inst, i,
                                                   seq, {debug_info}, false));
          previous_inputs.push_back(in);
        }

        // Call the user operation and add it to the sequence.
        seq.add(as_function_ptr_gradient(graph, pdi, gradients, previous_inputs,
                                         previous_outputs, outputs, attributes,
                                         instruction_name));
      }
    }

    // Register each of the returned tuple elements (if any) as outputs.
    if (outputs.size() < number_of_outputs) {
      return xla::InternalErrorStrCat(
          "Instruction ", instruction_name,
          " has mismatched outputs number, expected: ", number_of_outputs,
          ", returned: ", outputs.size());
    }
    for (uint32_t i = 0; i < number_of_outputs; ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
    }
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      poplar::Graph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    const HloInstruction* inst = tensor_target.tgt;
    const int64 input_index = tensor_target.input_index;

    const HloUserOpInstruction* user_op = Cast<HloUserOpInstruction>(inst);

    // It is valid to not have an allocator function.
    void* allocator_func = user_op->GetAllocatorFunc();
    if (!allocator_func) {
      return Status::OK();
    }

    // Get the Poplar shape and type
    auto shape = user_op->operand(input_index)->shape();
    std::vector<std::size_t> poplar_shape = PoplarShapeFromXlaShape(shape);
    TF_ASSIGN_OR_RETURN(auto poplar_type, PoplarDataType(shape));

    // Convert into a function pointer.
    poplar::Tensor (*allocatorSig)(
        poplar::Graph&, std::int64_t, const std::vector<std::size_t>&,
        poplar::Type, const std::string&, const std::string&);
    allocatorSig = reinterpret_cast<decltype(allocatorSig)>(allocator_func);
    const std::string attributes = user_op->GetAttributes();

    // Return the tensor via user specified function.
    return allocatorSig(graph, input_index, poplar_shape, poplar_type,
                        attributes,
                        absl::StrCat(GetDebugName(inst), ":", input_index));
  }
};

REGISTER_POPLAR_OP(UserOp, UserOpImpl);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
