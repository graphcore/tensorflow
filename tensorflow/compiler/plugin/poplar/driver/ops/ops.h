/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <gcl/Collectives.hpp>
#include <popfloat/experimental/CastToGfloat.hpp>
#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplin/Convolution.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Expr.hpp>
#include <popops/Operation.hpp>
#include <poputil/exceptions.hpp>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
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

StatusOr<DriverProgramSequence> CreateComparisonOp(CompilerResources& res,
                                                   const HloInstruction* inst,
                                                   const xla::Shape& output,
                                                   TensorMap& tensor_map);

// Performs A = A z B * c (where z is + or -, depending on the op_type)
Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& lhs, poplar::Tensor& rhs,
    poplar::Tensor& scale, poplar::program::Sequence& prog,
    const HloOpcode op_type, const poplar::DebugNameAndId& debug_name_and_id);

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& lhs, poplar::Tensor& rhs,
    const double scale, poplar::program::Sequence& prog,
    const HloOpcode op_type, const poplar::DebugNameAndId& debug_name_and_id);

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, const double scale_a,
    poplar::Tensor& tensor_b, const double scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const poplar::DebugNameAndId& debug_name_and_id);

Status ScaledInplaceConstantOrTensor(
    poplar::Graph& graph, poplar::Tensor& tensor_a, poplar::Tensor& scale_a,
    poplar::Tensor& tensor_b, poplar::Tensor& scale_b,
    poplar::program::Sequence& prog, const HloOpcode op_type,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateTupleSelectOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateCastOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

// Same as above, but allows the instruction for which we get the inputs for
// (`inst`) and the instruction from which we take the reduction parameters from
// (`reduce_inst`) to be different.
StatusOr<DriverProgramSequence> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const HloInstruction* reduce_inst, const xla::Shape& output,
    TensorMap& tensor_map, bool with_scale,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateSimpleReduction(
    CompilerResources& res, popops::Operation reduction_operation,
    const HloInstruction* inst, const HloInstruction* reduce_inst,
    const xla::Shape& output, TensorMap& tensor_map, bool with_scale,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateSimpleWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsGfloatParams(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    poplar::Type gf_calc_type, const unsigned gf_packed_cfg,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsCastNativeToGfloat(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& cast_op_cfg,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsCastGfloatToNative(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& cast_op_cfg,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsPooling(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    const poplar::DebugNameAndId& debug_name_and_id,
    absl::optional<const HloInstruction*> optional_reduction_op =
        absl::nullopt);

StatusOr<DriverProgramSequence> CreatePoplibsMaxPoolGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const Window& window, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePoplibsPoolingGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateParallelMap(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateFusionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

// Version of While op which allows inputs to not have a layout.
StatusOr<DriverProgramSequence> CreateWhileOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

// Version of Repeat op which allows inputs to not have a layout.
StatusOr<DriverProgramSequence> CreateRepeatOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

// A ResourceUpdate op which allows inputs to not have a layout.
StatusOr<DriverProgramSequence> CreateResourceUpdateOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

// Version of Function op which allows inputs to not have a layout.
StatusOr<DriverProgramSequence> CreateFunctionOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

// Version of Pipeline op which allows inputs to not have a layout.
StatusOr<DriverProgramSequence> CreatePipelineOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateConvBiasAddOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map);

StatusOr<DriverProgramSequence> CreateSimpleSelectAndScatter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateDynamicUpdateSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateDynamicSliceOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateIota(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateSlice(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    DeferredArgRBVectors& deferred_inputs, const xla::Shape& output,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateReplicatedAllReduce(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const gcl::CollectiveOperator op,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateReplicatedAllToAll(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

poplar::Tensor ShuffleNormInputToPoplar(const poplar::Tensor& input,
                                        const unsigned feature_dimension);

poplar::Tensor ShuffleNormOutputToTensorflow(const poplar::Tensor& output,
                                             const unsigned feature_dimension);

StatusOr<DriverProgramSequence> CreateSelectScalarFromRows(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateUpdateScalarInRows(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map, const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<DriverProgramSequence> CreateTuple(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

struct ExternalAndLocalTransferSequence {
  // For infeeds/outfeeds, we want to place the local copies together in
  // a single sequence, and the host transfers together in a single
  // sequence. This is because we can then merge them together. To
  // do this, instead of returning the sequence return both so that
  // the calling layer can decide how to interleave these copies

  ExternalAndLocalTransferSequence() = default;
  ExternalAndLocalTransferSequence(DriverProgramSequence et,
                                   DriverProgramSequence lt)
      : external_transfer(et), local_transfer(lt) {}

  DriverProgramSequence external_transfer;
  DriverProgramSequence local_transfer;
};

StatusOr<ExternalAndLocalTransferSequence> CreateOutfeed(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id);

StatusOr<ExternalAndLocalTransferSequence> CreateInfeed(
    CompilerResources& res, const HloInstruction* inst, int64_t tuple_index,
    const xla::Shape& output_shape, DriverTensor tensor,
    const poplar::DebugNameAndId& debug_name_and_id);

/* Op Creation Helpers */

Status SetPartialsTypeIfPresent(const HloInstruction* inst,
                                poplar::OptionFlags& option_flags);
Status SetPartialsTypeIfPresent(
    const PoplarBackendConfig& poplar_backend_config,
    poplar::OptionFlags& option_flags);

}  // namespace poplarplugin
}  // namespace xla

#endif
