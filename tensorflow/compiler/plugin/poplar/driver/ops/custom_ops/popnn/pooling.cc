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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
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

#include <poplar/DebugContext.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>

#include <string>

namespace xla {
namespace poplarplugin {
namespace {
class MaxPoolOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MaxPoolOp");
    auto pool_inst = Cast<HloPoolingInstruction>(inst);

    return CreatePoplibsPooling(res, inst, tensor_map, popnn::PoolingType::MAX,
                                pool_inst->window(), {debug_info});
  }
};
REGISTER_POPLAR_OP(MaxPool, MaxPoolOp);

class AvgPoolOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AvgPoolOp");
    auto pool_inst = Cast<HloPoolingInstruction>(inst);

    return CreatePoplibsPooling(res, inst, tensor_map, popnn::PoolingType::AVG,
                                pool_inst->window(), {debug_info});
  }
};
REGISTER_POPLAR_OP(AvgPool, AvgPoolOp);

class MaxPoolGradOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "MaxPoolGradOp");
    auto pool_inst = Cast<HloPoolingInstruction>(inst);

    return CreatePoplibsMaxPoolGrad(res, inst, tensor_map, pool_inst->window(),
                                    {debug_info});
  }
};
REGISTER_POPLAR_OP(MaxPoolGrad, MaxPoolGradOp);

class AvgPoolGradOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "AvgPoolGradOp");
    auto pool_inst = Cast<HloPoolingInstruction>(inst);

    return CreatePoplibsPoolingGrad(res, inst, tensor_map,
                                    popnn::PoolingType::AVG,
                                    pool_inst->window(), {debug_info});
  }
};
REGISTER_POPLAR_OP(AvgPoolGrad, AvgPoolGradOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
