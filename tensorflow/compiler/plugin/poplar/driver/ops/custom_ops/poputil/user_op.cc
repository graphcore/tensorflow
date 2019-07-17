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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <poputil/TileMapping.hpp>
namespace xla {
namespace poplarplugin {
namespace {
class UserOpImpl : public PoplibsOpDef {
  static absl::flat_hash_set<std::string> previously_added_codelets;

  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const HloUserOpInstruction* user_op_inst =
        DynCast<HloUserOpInstruction>(inst);

    if (!user_op_inst) {
      return xla::FailedPrecondition(
          "Expected HLO instruction to be HloUserOpInstruction.");
    }

    const std::string& gp_path = user_op_inst->GetPath();

    // Add the codelet if needed.
    if (gp_path.empty() == false &&
        !previously_added_codelets.contains(gp_path)) {
      graph.addCodelets(gp_path);
      previously_added_codelets.insert(gp_path);
    }

    // Get the function pointer from the HLO.
    poplar::program::Program (*as_function_ptr)(
        poplar::Graph&, const std::vector<poplar::Tensor>&,
        std::vector<poplar::Tensor>& outputs);

    as_function_ptr = reinterpret_cast<decltype(as_function_ptr)>(
        user_op_inst->GetPointerToFunc());

    poplar::program::Sequence seq;
    std::vector<poplar::Tensor> inputs;
    std::vector<poplar::Tensor> outputs;

    // Get the variadic inputs and store them in the vector.
    for (size_t i = 0; i < user_op_inst->NumInputs(); ++i) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor in,
          FindInstructionInput(tensor_map, res, inst, i, seq, false));
      inputs.push_back(in);
    }

    // Call the user operation and add it to the sequence.s
    seq.add(as_function_ptr(graph, inputs, outputs));

    // Register each of the returned tuple elements (if any) as outputs.
    for (int i = 0; i < outputs.size(); ++i) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
    }
    return seq;
  }
};

absl::flat_hash_set<std::string> UserOpImpl::previously_added_codelets{};

REGISTER_POPLIBS_OP(Poputil, UserOp, UserOpImpl);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
