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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_UTIL_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/verified_streams_indices.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplin/Convolution.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Expr.hpp>
#include <poputil/exceptions.hpp>

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace popops {
class SlicePlan;
}  // namespace popops

namespace xla {
class HloModule;
class HloInstruction;
class HloComputation;
class Literal;
class Shape;

namespace poplarplugin {

struct CompilerResources;
class PoplarExecutor;

Status SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                      const Literal& literal);

// Get the master graph
poplar::Graph& GetMasterGraph(CompilerResources&);

// Get the appropriate virtual graph, or the replicated/master graph if not
poplar::Graph& GetGraph(CompilerResources&, const HloInstruction*);

// Get the shard Id for a given output of the given instruction.
uint64 GetShardForOutputIndex(const HloInstruction* inst,
                              int flattened_output_tuple_index);

// Get the virtual graph for a particular output of an operation. Operations
// like Parameter, Infeed, Call, While, Tuple can have multiple tensor
// outputs on different IPUs.
poplar::Graph& GetGraphWithOutputIndex(CompilerResources&,
                                       const HloInstruction*,
                                       int flattened_output_tuple_index);

// Convert a poplar/poplibs exception to a Tensorflow error Status
Status PoplarExceptionToTensorflowStatus(const std::string& origin,
                                         const std::exception& e);

void SetFlagIfNotPresent(poplar::OptionFlags& opts, const std::string& key,
                         const std::string& value);

poplar::OptionFlags GetReplicateAllReduceOptions();

// Try and dump the profiler report to a file if a OOM exception occurs.
void DumpIfPoplarOutOfMemoryAllocationException(
    const PoplarExecutor*, const std::string& module_name,
    const poplar::graph_memory_allocation_error& p_e);

/* Optimization tests */

bool IsPoplibsPool(const HloInstruction*, const HloComputation*);

bool IsSimpleSelection(const HloComputation*);

bool IsReducibleArithmetic(const HloComputation*);

StatusOr<bool> IsParallelMap(const HloInstruction*, const HloComputation*);

StatusOr<poplar::OptionFlags> GetConvolutionOptionsForInst(
    const HloInstruction* inst, CompilerResources& res);

StatusOr<poplar::OptionFlags> GetConvolutionOptionsForInst(
    const HloInstruction* inst, CompilerResources& res, const MLType conv_type);

StatusOr<poplar::OptionFlags> GetMatMulOptionsForInst(
    const HloInstruction* inst, CompilerResources& res);

poplar::program::Sequence ZeroTensors(CompilerResources& res);

bool IsRemoteParameter(int64 parameter_number,
                       const RemoteParameterInfos& remote_parameter_infos);
bool IsRemoteParameter(int64 parameter_number, const CompilerResources& res);
bool IsRemoteParameter(HloInstruction* inst, const CompilerResources& res);

StatusOr<std::string> GetInstructionCompilationInfo(
    const std::unique_ptr<xla::HloModule>& module, CompilerResources& res);

// Add a copy between two tensors with compatbile aliasing Poplar Tensors.
poplar::program::Sequence TensorCopyWithAliasing(poplar::Graph& graph,
                                                 const poplar::Tensor& src,
                                                 const poplar::Tensor& dst);

// Modify the compiler resources to indicate the embedding associated with a
// slice plan has been allocated with the given plan.
void NotifySlicePlanAllocation(CompilerResources& res,
                               const popops::SlicePlan* plan);

// Test whether the given slice plan has been used to allocate the embedding
// input.
bool SlicePlanHasAllocation(CompilerResources& res,
                            const popops::SlicePlan* plan);

// Get a slice plan for an instruction.
StatusOr<const popops::SlicePlan*> GetSlicePlan(CompilerResources& res,
                                                const HloInstruction* inst);

// A helper function to convert inputs into deferred inputs.
using DeferredArgVectors =
    std::vector<std::vector<absl::optional<poplar::Tensor>>>;
DeferredArgVectors ConvertInputsToDeferredInputs(TensorVectors& inputs);

/* Generate a JSON struture describing the tensor mappings
 */
std::string GetTensorMappingJson(const std::string& module_name,
                                 const poplar::Graph& graph,
                                 const TensorMaps& tensor_map);

/* Save the inputs / outputs metadata from the compiler resources in a Json
 * file.
 */
StatusOr<std::string> CreateExecutableMetadataJson(
    const InputOutputAliasingMap& io_map, const InfeedInfos& infeed_infos,
    const OutfeedInfos& outfeed_infos, uint32 replication_count,
    const poplar::OptionFlags& opts, const poplar::Target& target,
    const VerifiedStreamsIndices::KeyIdMappings& indices,
    const std::vector<string>& checkpoint_feeds_order);

}  // namespace poplarplugin
}  // namespace xla

#endif
