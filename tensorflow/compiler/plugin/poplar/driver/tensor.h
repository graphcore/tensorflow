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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TENSOR_H_

#include <string>
#include <utility>
#include <vector>

#include <poplar/TensorCloneMethod.hpp>
#include <popops/DynamicSlice.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace poplar {
class Tensor;
class Graph;
class Type;
class DebugNameAndId;
}  // namespace poplar

namespace xla {
namespace poplarplugin {

struct CompilerResources;

StatusOr<poplar::Type> PoplarDataType(const xla::PrimitiveType& element_type);

StatusOr<poplar::Type> PoplarDataType(const xla::Shape& shape);

std::vector<size_t> PoplarShapeFromXlaShape(const xla::Shape& xla_shape);

xla::Shape XlaShapeFromPoplarShape(PrimitiveType element_type,
                                   const std::vector<size_t>& poplar_shape);

DriverTensor ConvertToDeviceLayout(const Shape& shape,
                                   const DriverTensor& tensor);

DriverTensor ConvertFromDeviceLayout(const Shape& shape,
                                     const DriverTensor& tensor);

bool PoplarShapeMatchesXLAShape(const DriverTensor& tensor,
                                const xla::Shape& shape);

bool PoplarShapeMatchesXLAShape(poplar::RemoteBuffer remote_buffer,
                                const xla::Shape& shape);

bool PoplarShapeMatchesXLAShape(TensorOrRemoteBuffer torb,
                                const xla::Shape& shape,
                                CompilerResources& resources);

// Concatenate all tensors into a single tensor.
DriverTensor ConcatenateTensors(const std::vector<DriverTensor>& tensors,
                                int64_t dimension = 0);

// Concatenate all tensors into a single tensor.
DriverTensor FlattenAndConcatenateTensors(
    const std::vector<DriverTensor>& tensors);

// Given a tensor of shape [..., N, ...] where N is the `slice_dimension`,
// create an output tensor of size [..., output_size, ...]
DriverTensor CreateTensorFromSlice(
    DriverGraph& graph, const DriverTensor& slice, int64_t slice_dimension,
    int64_t output_size, CompilerResources& resources,
    const poplar::DebugNameAndId& debug_name_and_id);

// Clone the tensor and rebalance any aliasing across the tiles.
DriverTensor TensorCloneAndRebalanceAliasing(
    DriverGraph& graph, CompilerResources& res, const DriverTensor& tensor,
    const poplar::DebugNameAndId& debug_name_and_id);
StatusOr<DriverTensor> SliceTensor(
    DriverTensor tensor_to_slice,
    const HloInstruction::InstructionVector& slices, int64_t slice_index);

// Slice tensor into tensors with shapes like the tensors.
std::vector<DriverTensor> SliceTensorIntoTensorsLike(
    DriverTensor tensor_to_slice,
    const std::vector<DriverTensor>& like_tensors);

StatusOr<DriverTensor> AddDynamicSliceTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla);

StatusOr<DriverTensor> AddDynamicUpdateSliceTensor(
    DriverGraph& graph, const std::string& debug_name,
    const xla::Shape& input_shape_xla, const xla::Shape& update_shape_xla);

StatusOr<DriverTensor> AddDynamicSliceTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla,
    DriverTensor& physical_layout);

StatusOr<DriverTensor> AddScatterTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla);

StatusOr<DriverTensor> AddGatherTensor(DriverGraph& graph,
                                       const std::string& debug_name,
                                       const xla::Shape& shape_xla,
                                       std::vector<std::size_t> slice_sizes,
                                       std::vector<unsigned> start_index_map);

StatusOr<DriverTensor> AddPlainTensor(DriverGraph& graph,
                                      const poplar::DebugContext& debug_context,
                                      const xla::Shape& shape,
                                      CompilerResources& resources,
                                      bool offset = true);

// Add a tensor with layout optimised for host exchange.
StatusOr<DriverTensor> AddHostCopyTensor(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const xla::Shape& shape);

StatusOr<DriverTensor> CreateIndicesTensor(
    DriverGraph& graph, const popops::SlicePlan& plan,
    const xla::Shape& xla_indices_shape,
    const poplar::DebugNameAndId& debug_name_and_id);

// Returns true if the given tensor source has a special layout allocation
// target.
bool HasTensorAllocationTarget(const TensorLocation& src,
                               const CompilerResources& resources);

StatusOr<DriverTensor> AddTensorForTarget(
    DriverGraph& graph, const TensorLocation& source,
    const TensorTarget& tensor_target, CompilerResources& resources,
    const TensorMap& tensor_map, const poplar::DebugContext& debug_context);

StatusOr<DriverTensor> AddTensor(DriverGraph& graph, const TensorLocation& src,
                                 const xla::Shape& shape,
                                 CompilerResources& resources,
                                 const TensorMap& tensor_map,
                                 const poplar::DebugContext& debug_context);

StatusOr<DriverTensor> AddConstantTensor(
    DriverGraph& graph, const TensorLocation& src, const xla::Shape& shape,
    const xla::Literal& literal, CompilerResources& resources,
    const TensorMap& tensor_map, const poplar::DebugContext& debug_context);

// Creates a constant tensor.
StatusOr<DriverTensor> CreateConstantTensor(
    DriverGraph& graph, const xla::Literal& literal, const xla::Shape& shape,
    const poplar::Type& poplar_type,
    const poplar::DebugNameAndId& debug_name_and_id);

// Sets a value of a tensor to a constant.
Status SetInitialTensorValue(DriverGraph& graph, DriverTensor& tensor,
                             const xla::Literal& literal);

template <typename T>
DriverTensor TileTensor(const T& multiples, const DriverTensor& in);

StatusOr<DriverTensor> UnpadTensor(const PaddingConfig& cfg,
                                   const DriverTensor& in);

StatusOr<DriverTensor> PadTensor(const PaddingConfig& cfg,
                                 const DriverTensor& in,
                                 const DriverTensor& pad);

StatusOr<DriverTensor> ReverseTensor(const DriverTensor& in,
                                     const std::vector<int64_t>& dimensions);

StatusOr<DriverTensor> BroadcastTensor(
    const DriverTensor& in, const xla::Shape& out,
    const std::vector<int64_t>& dimensions = {});

Status AddOutput(TensorMap& map, const HloInstruction* inst, int64_t n,
                 const TensorOrRemoteBuffer& torb);

Status AddOutputTensor(TensorMap& map, const HloInstruction* inst, int64_t n,
                       const DriverTensor& tensor);

Status AddOutputOpaque(TensorMap& map, const HloInstruction* inst, int64_t n,
                       absl::any token);

/* This returns a [range) which correspond to the flat tuple indices of output
 * tensors.
 */
std::pair<int64_t, int64_t> FindGetTupleElementTupleIndices(
    const HloInstruction* inst);

/**
 * This returns a vector of all poplar tensors which are outputs of the inst
 * operand index `input` in range [range.first, range.second).
 *
 * \param map   The tensor map from which to find the poplar tensors or remote
 *              buffers.
 * \param res   The compiler resources.
 * \param inst  The instruction which we want the inputs tensors or remote
 *              buffers for.
 * \param input The operand input index.
 * \param range The flattened tuple index range to select.
 * \param seq   A poplar sequence control program that will be populated with
 *              any copies that are required to produce the output.
 * \param expand_aliasing When true, any tensors which have aliasing (and may
 *                        not be parallel writeable) are duplicated with their
 *                        aliased elements "expanded".
 *
 * \returns A Status on error, or the requested vector of tensors.
 *
 * \note This function should only be used if a remote buffer is expected, or
 *       can't be excluded as a possibility.
 */
StatusOr<TensorVector> FindInstructionInputTensorsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, std::pair<int64_t, int64_t> range,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/**
 * This returns a vector of all poplar tensors or remote buffers which are
 * outputs of the inst operand index `input` in range [range.first,
 * range.second).
 *
 * \param map   The tensor map from which to find the poplar tensors or remote
 *              buffers.
 * \param res   The compiler resources.
 * \param inst  The instruction which we want the inputs tensors or remote
 *              buffers for.
 * \param input The operand input index.
 * \param range The flattened tuple index range to select.
 * \param seq   A poplar sequence control program that will be populated with
 *              any copies that are required to produce the output.
 * \param expand_aliasing When true, any tensors which have aliasing (and may
 *                        not be parallel writeable) are duplicated with their
 *                        aliased elements "expanded".
 *
 * \returns The requested vector of tensors.
 *
 * \note This function should only be used if a remote buffer is expected, or
 *       can't be excluded as a possibility.
 */
TensorOrRemoteBufferVector FindInstructionInputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, std::pair<int64_t, int64_t> range,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/* This returns the single poplar tensor which is the non-tuple input to the
 * input to the instruction
 */
StatusOr<DriverTensor> FindInstructionInput(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

StatusOr<DriverTensor> FindF8InstructionInput(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/**
 * This returns a vector of the poplar tensors or remote buffers which are the
 * inputs to the instruction at the given index.
 *
 * \param map   The tensor map from which to find the poplar tensors or remote
 *              buffers.
 * \param res   The compiler resources.
 * \param inst  The instruction which we want the inputs tensors or remote
 *              buffers for.
 * \param input The operand input index.
 * \param seq   A poplar sequence control program that will be populated with
 *              any copies that are required to produce the output.
 * \param expand_aliasing When true, any tensors which have aliasing (and may
 *                        not be parallel writeable) are duplicated with their
 *                        aliased elements "expanded".
 *
 * \returns The requested vector of tensors or remote buffers.
 *
 * \note This function should only be used if a remote buffer is expected, or
 *       can't be excluded as a possibility. Usually this is not the case.
 */
TensorOrRemoteBufferVector FindInstructionInputs(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/**
 * This returns the poplar tensors which are the inputs to the instruction.
 *
 * \param map   The tensor map from which to find the poplar tensors.
 * \param res   The compiler resources.
 * \param inst  The instruction which we want the inputs tensors for.
 * \param input The input index.
 * \param seq   A poplar sequence control program that will be populated with
 *              any copies that are required to produce the output.
 * \param expand_aliasing When true, any tensors which have aliasing (and may
 *                        not be parallel writeable) are duplicated with their
 *                        aliased elements "expanded".
 *
 * \returns A Status on error, or the requested vector of tensors.
 *
 * \note This function should only be used if a remote buffer is expected, or
 *       can't be excluded as a possibility.
 */
StatusOr<TensorVector> FindInstructionInputTensors(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64_t input, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

bool AreInplaceOutputTensorsWritable(TensorMap& map, CompilerResources& res,
                                     const HloInstruction* inst);

/* This returns a vector of poplar tensors which are all of the outputs from
 * the given instruction.
 */
TensorOrRemoteBufferVector FindInstructionOutputs(const TensorMap& map,
                                                  CompilerResources& res,
                                                  const HloInstruction* inst);

StatusOr<TensorVector> FindInstructionOutputTensors(const TensorMap& map,
                                                    CompilerResources& res,
                                                    const HloInstruction* inst);

/* Sometimes an inplace op cannot be performed because the input/output tensor
 * is not parallel writable or because further analysis has shown that the op
 * can no longer be in place. If that's the case, this function will add an
 * extra tensor copy and use that tensor as the input/output tensor.
 *
 * The TensorVector contains only those inputs which are listed as inplace
 * inputs by HloPoplarInplaceDescription.
 */
StatusOr<TensorVectors> FindInplaceOutputTensors(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/**
 * Same as the above function, but has the option to also return remote
 * buffers.
 */
StatusOr<TensorOrRemoteBufferVectors> FindInplaceOutputs(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id,
    bool expand_aliasing = true);

/* This returns a vector of all poplar tensors which are outputs of the inst
 *   in range [range.first, range.second).
 */
StatusOr<TensorVector> FindInstructionOutputTensorsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64_t, int64_t> range);

StatusOr<TensorOrRemoteBufferVector> FindInstructionOutputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64_t, int64_t> range);

/* This returns a vector of all poplar tensors which are outputs of the inst
 * in range [range.first, range.second) - any aliasing is expanded - TODO
 * T5364
 */
StatusOr<TensorVector> FindExpandedInstructionOutputsInRange(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    std::pair<int64_t, int64_t> range, poplar::program::Sequence& seq,
    const poplar::DebugNameAndId& debug_name_and_id);
}  // namespace poplarplugin
}  // namespace xla

#endif
