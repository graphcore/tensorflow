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

#include <popnn/NonLinearity.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

template <popnn::NonLinearityType NLType>
class NonLinearityOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    poplar::Tensor t;
    const bool is_inplace =
        AreInplaceOutputTensorsWritable(tensor_map, res, inst);

    if (is_inplace) {
      TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                          FindInplaceOutputTensors(tensor_map, res, inst, seq));
      CHECK_EQ(inputs.size(), 1);
      CHECK_EQ(inputs[0].size(), 1);
      t = inputs[0][0];
      popnn::nonLinearityInPlace(graph, NLType, t, seq, GetDebugName(inst));
    } else {
      TF_ASSIGN_OR_RETURN(
          t, FindInstructionInput(tensor_map, res, inst, 0, seq, false));

      t = popnn::nonLinearity(graph, NLType, t, seq, GetDebugName(inst));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));

    return seq;
  }
};
REGISTER_POPLAR_OP(Relu, NonLinearityOp<popnn::NonLinearityType::RELU>);
REGISTER_POPLAR_OP(Gelu, NonLinearityOp<popnn::NonLinearityType::GELU>);
REGISTER_POPLAR_OP(Sigmoid, NonLinearityOp<popnn::NonLinearityType::SIGMOID>);

template <popnn::NonLinearityType NLType>
class NonLinearityGradOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor out,
        FindInstructionInput(tensor_map, res, inst, 0, seq, false));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor outgrad,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    poplar::Tensor t = popnn::nonLinearityInputGradient(
        graph, NLType, out, outgrad, seq, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));

    return seq;
  }
};
REGISTER_POPLAR_OP(ReluGrad, NonLinearityGradOp<popnn::NonLinearityType::RELU>);
REGISTER_POPLAR_OP(GeluGrad, NonLinearityGradOp<popnn::NonLinearityType::GELU>);
REGISTER_POPLAR_OP(SigmoidGrad,
                   NonLinearityGradOp<popnn::NonLinearityType::SIGMOID>);
REGISTER_POPLAR_OP(TanhGrad, NonLinearityGradOp<popnn::NonLinearityType::TANH>);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
