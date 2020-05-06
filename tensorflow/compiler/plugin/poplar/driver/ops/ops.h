/* Copyright 2017 - 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_OPS_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */

#include <popfloat/experimental/CastToGfloat.hpp>
#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplin/Convolution.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Expr.hpp>
#include <poputil/exceptions.hpp>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;
class HloComputation;
class Literal;
class Shape;

namespace poplarplugin {

class PoplarBackendConfig;
struct CompilerResources;

StatusOr<popops::expr::UnaryOpType> LookupUnaryFn(const HloInstruction*);

StatusOr<popops::expr::BinaryOpType> LookupBinaryFn(const HloInstruction*);

StatusOr<popops::expr::TernaryOpType> LookupTernaryFn(const HloInstruction*);

StatusOr<popops::expr::BinaryOpType> LookupComparisonFn(
    const HloInstruction* inst);

std::set<unsigned int> GetPoolingReductionDims(const Window& window);
/* Ops */

StatusOr<poplar::program::Program> CreateComparisonOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

// Performs A = A z B * c (where z is + or -, depending on the op_type)
Status ScaledInplaceConstantOrTensor(poplar::Graph& graph, poplar::Tensor& lhs,
                                     poplar::Tensor& rhs, poplar::Tensor& scale,
                                     poplar::program::Sequence& prog,
                                     const HloOpcode op_type,
                                     const std::string& name);

Status ScaledInplaceConstantOrTensor(poplar::Graph& graph, poplar::Tensor& lhs,
                                     poplar::Tensor& rhs, const double scale,
                                     poplar::program::Sequence& prog,
                                     const HloOpcode op_type,
                                     const std::string& name);

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, const double scale_a,
    poplar::Tensor& tensor_b, const double scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const std::string& name);

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, poplar::Tensor& scale_a,
    poplar::Tensor& tensor_b, poplar::Tensor& scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const std::string& name);

StatusOr<poplar::program::Program> CreateMatMulForDotOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateTupleSelectOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCastOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

// Same as above, but allows the instruction for which we get the inputs for
// (`inst`) and the instruction from which we take the reduction parameters from
// (`reduce_inst`) to be different.
StatusOr<poplar::program::Program> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const HloInstruction* reduce_inst, const xla::Shape& output,
    TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreatePoplibsWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreatePoplibsGfloatParams(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    poplar::Type gf_calc_type, const unsigned gf_packed_cfg);

StatusOr<poplar::program::Program> CreatePoplibsCastNativeToGfloat(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& cast_op_cfg);

StatusOr<poplar::program::Program> CreatePoplibsCastGfloatToNative(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& cast_op_cfg);

StatusOr<poplar::program::Program> CreatePoplibsPooling(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    absl::optional<const HloInstruction*> optional_reduction_op =
        absl::nullopt);

StatusOr<poplar::program::Program> CreatePoplibsMaxPoolGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const Window& window);

StatusOr<poplar::program::Program> CreatePoplibsPoolingGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window);

StatusOr<poplar::program::Program> CreateParallelMap(CompilerResources& res,
                                                     const HloInstruction* inst,
                                                     const xla::Shape& output,
                                                     TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCallOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateFusionOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map);

// Version of While op which allows inputs to not have a layout.
StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 DeferredArgVectors& inputs,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

// Version of Repeat op which allows inputs to not have a layout.
StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  DeferredArgVectors& inputs,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateFunctionOp(CompilerResources& res,
                                                    const HloInstruction* inst,
                                                    const xla::Shape& output,
                                                    TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreatePipelineOp(CompilerResources& res,
                                                    const HloInstruction* inst,
                                                    const xla::Shape& output,
                                                    TensorMap& tensor_map);

// Version of Pipeline op which allows inputs to not have a layout.
StatusOr<poplar::program::Program> CreatePipelineOp(CompilerResources& res,
                                                    const HloInstruction* inst,
                                                    DeferredArgVectors& inputs,
                                                    const xla::Shape& output,
                                                    TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateConvBiasAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSimpleSelectAndScatter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateDynamicUpdateSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateDynamicSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateGeluOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateGeluGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReluOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReluGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSigmoidOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSigmoidGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateTanhOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateTanhGradOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateIota(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output_shape,
                                              TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateCopy(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output_shape,
                                              TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSlice(CompilerResources& res,
                                               const HloInstruction* inst,
                                               const xla::Shape& output_shape,
                                               TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateZeroPadOp(CompilerResources& res,
                                                   const HloInstruction* inst,
                                                   const xla::Shape& output,
                                                   TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReplicatedAllReduce(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateReplicatedAllToAll(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateSort(CompilerResources& res,
                                              const HloInstruction* inst,
                                              TensorMap& tensor_map);

poplar::Tensor ShuffleNormInputToPoplar(const poplar::Tensor& input,
                                        const unsigned feature_dimension);

poplar::Tensor ShuffleNormOutputToTensorflow(const poplar::Tensor& output,
                                             const unsigned feature_dimension);

StatusOr<poplar::program::Program> CreateSelectScalarFromRows(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateUpdateScalarInRows(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateTuple(CompilerResources& res,
                                               const HloInstruction* inst,
                                               TensorMap& tensor_map,
                                               bool expand_aliasing = true,
                                               bool preserve_aliases = false);

StatusOr<poplar::program::Program> CreateOutfeed(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 TensorMap& tensor_map);

StatusOr<poplar::program::Program> CreateInfeed(CompilerResources& res,
                                                const HloInstruction* inst,
                                                int64 tuple_index,
                                                const xla::Shape& output_shape,
                                                poplar::Tensor tensor);

/* Op Creation Helpers */

StatusOr<poplar::program::Sequence> CreateSort(
    poplar::Graph& graph, poplar::Tensor input, const int64 dimension,
    const std::string& debug_name = "");

StatusOr<poplar::program::Sequence> CreateSort(
    poplar::Graph& graph, poplar::Tensor key, poplar::Tensor value,
    const int64 dimension, const std::string& debug_name = "");

Status SetPartialsTypeIfPresent(const HloInstruction* inst,
                                poplar::OptionFlags& option_flags);
Status SetPartialsTypeIfPresent(
    const PoplarBackendConfig& poplar_backend_config,
    poplar::OptionFlags& option_flags);

}  // namespace poplarplugin
}  // namespace xla

#endif
