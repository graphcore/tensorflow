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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"

#include <gcl/Collectives.hpp>
#include <popops/DynamicSlice.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class AllGatherOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    // Replicated sum the concatenated tensor
    poplar::Tensor output =
        gcl::allGather(graph, input, seq, GetDebugName(inst),
                       GetReplicatedCollectiveOptions(res));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }
};

REGISTER_POPLAR_OP(AllGather, AllGatherOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
