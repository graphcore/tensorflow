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
#include <popops/NormaliseImage.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/normalise_image.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace poplarplugin {
namespace {

class NormaliseImageOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "NormaliseImageOp");
    poplar::program::Sequence seq({}, debug_info);

    const HloNormaliseImage* as_norm_image = Cast<HloNormaliseImage>(inst);
    const float scale = as_norm_image->Scale();

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor image,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor channel_offsets,
        FindInstructionInput(tensor_map, res, inst, 1, seq, {debug_info}));
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor channel_scales,
        FindInstructionInput(tensor_map, res, inst, 2, seq, {debug_info}));

    poplar::DebugNameAndId debug_name_and_id(debug_info);
    poplar::Tensor out =
        popops::normaliseImage(graph, seq, image, scale, channel_offsets,
                               channel_scales, {debug_name_and_id});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "NormaliseImageAllocator");

    const HloInstruction* inst = tensor_target.tgt;
    const Shape& xla_shape = inst->operand(0)->shape();
    TF_ASSIGN_OR_RETURN(auto type, PoplarDataType(xla_shape));
    auto shape = PoplarShapeFromXlaShape(xla_shape);

    auto out = popops::createNormaliseImageInput(
        graph, type, shape, {debug_context, "CreateInput"});
    return out;
  }
};

REGISTER_POPLAR_OP(NormaliseImage, NormaliseImageOp)

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
