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
        const std::vector<const void*>& data,
        const std::vector<std::uint32_t>& number_of_elements,
        const std::vector<void*>& outputs, const std::string& attributes,
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
      auto& host_function_info =
          res.annotations.host_function_infos[instruction_name];
      host_function_info.parent_instruction = user_op_inst;
      host_function_info.handle = instruction_name;

      std::vector<poplar::Tensor> inputs;
      std::vector<poplar::Graph::HostFunctionArgument> in_args;
      std::vector<poplar::Graph::HostFunctionArgument> out_args;
      std::vector<std::uint32_t> in_args_elems;
      inputs.resize(user_op_inst->NumInputs());
      outputs.resize(number_of_outputs);

      // Collect the input tensors and input arg descriptions.
      for (std::uint32_t i = 0; i < user_op_inst->NumInputs(); ++i) {
        // Get the poplar tensor.
        TF_ASSIGN_OR_RETURN(poplar::Tensor in,
                            FindInstructionInput(tensor_map, res, inst, i, seq,
                                                 {debug_info}, false));

        in_args.emplace_back(in.elementType(), in.numElements());
        in_args_elems.push_back(in.numElements());
        host_function_info.input_shapes.push_back(inst->operand(i)->shape());

        inputs[i] = in;
      }

      // Collect the output tensors and output arg descriptions.
      for (std::uint32_t output_index = 0; output_index != number_of_outputs;
           output_index++) {
        xla::Shape shape = output_shape.tuple_shapes()[output_index];
        // Create a new tensor using "AddTensor" to get a good layout.
        TF_ASSIGN_OR_RETURN(
            poplar::Tensor output_tensor,
            AddTensor(graph, TensorLocation{inst, output_index}, shape, res,
                      tensor_map, {debug_info, "output"}));

        out_args.emplace_back(output_tensor.elementType(),
                              output_tensor.numElements());
        host_function_info.output_shapes.push_back(shape);
        outputs[output_index] = output_tensor;
      }

      host_function_info.function = [as_function_host_rw_ptr, in_args_elems,
                                     attributes, instruction_name](
                                        const std::vector<const void*>& input,
                                        const std::vector<void*>& outputs) {
        as_function_host_rw_ptr(input, in_args_elems, outputs, attributes,
                                instruction_name);
      };

      // Create the host function
      auto user_fn_device =
          graph.addHostFunction(instruction_name, in_args, out_args);

      // Add the device call to the host function.
      seq.add(
          poplar::program::Call(user_fn_device, inputs, outputs, debug_info));
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
