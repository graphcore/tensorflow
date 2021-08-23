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
#include <gcl/Collectives.hpp>
#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplin/Convolution.hpp>
#include <popnn/CTCLoss.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Expr.hpp>
#include <poputil/exceptions.hpp>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "ipu/poplar_executable_data.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"
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

// Get the current process ID
int32 GetPID();

// Get the current time in ISO format %Y-%m-%d_%H-%M-%S localized
std::string GetCurrentTimeInISOFormat();

// Generate a directory name of format "{prefix}_{ISO date}_{PID}"
std::string GenerateDirectoryName(const std::string& prefix);

// Parses a JSON string in 'json_str' to a Json::Value object in 'attributes'
// Returns if it was successful
bool JsonParse(const std::string& json_str, Json::Value& attributes);

// Check if the string 'opt' is a key of the POPLAR_ENGINE_OPTIONS env var.
// Return its value if it exists.
absl::optional<std::string> GetPoplarEngineOption(const std::string& opt);

// Get the shard Id for a given output of the given instruction.
int64 GetShardForOutputIndex(const HloInstruction* inst,
                             int flattened_output_tuple_index);

// Get the virtual graph for a particular output of an operation. Operations
// like Parameter, Infeed, Call, While, Tuple can have multiple tensor
// outputs on different IPUs.
poplar::Graph& GetGraphWithOutputIndex(CompilerResources&,
                                       const HloInstruction*,
                                       int flattened_output_tuple_index);

// Convert a poplar/poplibs exception to a Tensorflow error Status
Status PoplarExceptionToTensorflowStatus(const std::string& origin,
                                         const std::exception& e,
                                         bool recoverable_reset = true);

void SetFlagIfNotPresent(poplar::OptionFlags& opts, const std::string& key,
                         const std::string& value);

poplar::OptionFlags GetReplicatedCollectiveOptions(
    const CompilerResources& res);

poplar::OptionFlags GetReplicateAllReduceOptions(const CompilerResources& res);

StatusOr<gcl::CommGroup> ToGclCommGroup(PoplarReplicaGroups replica_groups,
                                        const CompilerResources& res);

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

StatusOr<poplar::OptionFlags> GetCholeskyOptionsForInst(
    const HloInstruction* inst, CompilerResources& res);

StatusOr<poplar::OptionFlags> GetTriangularSolveOptionsForInst(
    const HloInstruction* inst, CompilerResources& res);

StatusOr<poplar::OptionFlags> GetSliceOptionsForInst(const HloInstruction* inst,
                                                     CompilerResources& res);

void AddZeroTensorToPreamble(CompilerResources& res, const poplar::Tensor& t,
                             const poplar::DebugNameAndId& debug_name_and_id);

const RemoteParameterInfo* FindRemoteParameterInfo(
    int64 parameter_number, const RemoteParameterInfos& remote_parameter_infos);
bool IsRemoteParameter(int64 parameter_number,
                       const RemoteParameterInfos& remote_parameter_infos);
bool IsRemoteParameter(int64 parameter_number, const CompilerResources& res);
bool IsRemoteParameter(const HloInstruction* inst,
                       const CompilerResources& res);

bool IsReplicaPartitioned(int64 parameter_number,
                          const RemoteParameterInfos& remote_parameter_infos);
bool IsReplicaPartitioned(int64 parameter_number, const CompilerResources& res);
bool IsReplicaPartitioned(const HloInstruction* inst,
                          const CompilerResources& res);

StatusOr<TensorOrRemoteBuffer> GetOrCreateRemoteBuffer(
    poplar::Graph& graph, CompilerResources& res,
    std::string remote_buffer_name, poplar::Type element_type,
    int64 element_count, int64 num_repeats, int64 num_merged,
    bool is_replica_partitioned = false);

StatusOr<TensorOrRemoteBuffer> GetOrCreateRemoteParameterBuffer(
    const HloInstruction* inst, CompilerResources& res);

bool IsInPipeline(const HloInstruction* inst, CompilerResources& res);

StatusOr<std::string> GetInstructionCompilationInfo(const HloModule* module,
                                                    CompilerResources& res);

// Add a copy between two tensors with compatbile aliasing Poplar Tensors.
poplar::program::Sequence TensorCopyWithAliasing(
    poplar::Graph& graph, const poplar::Tensor& src, const poplar::Tensor& dst,
    const poplar::DebugNameAndId& debug_name_and_id);

// Modify the compiler resources to indicate the embedding associated with a
// instruction has been allocated with the given plan.
void NotifySlicePlanAllocation(CompilerResources& res,
                               const TensorTarget& target);

// Test whether the given instruction has been used to allocate the embedding
// input.
StatusOr<bool> SlicePlanHasAllocation(CompilerResources& res,
                                      const HloInstruction* inst);

// Check if slice plan a could be used for allocation and plan b for slice
// Currently checks equivalence
StatusOr<bool> SlicePlansCompatible(CompilerResources& res,
                                    const HloInstruction* a,
                                    const HloInstruction* b);

// Get a slice plan for an instruction.
StatusOr<const popops::SlicePlan*> GetSlicePlan(CompilerResources& res,
                                                const HloInstruction* inst);

// Get a ctc plan for an instruction
StatusOr<const popnn::ctc::Plan*> GetCTCPlan(CompilerResources& res,
                                             const HloInstruction* inst);

// A helper function to convert inputs into deferred inputs.
using DeferredArgVectors =
    std::vector<std::vector<absl::optional<poplar::Tensor>>>;
DeferredArgVectors ConvertInputsToDeferredInputs(TensorVectors& inputs);
using DeferredArgRBVectors =
    std::vector<std::vector<absl::optional<TensorOrRemoteBuffer>>>;
DeferredArgRBVectors ConvertInputsToDeferredInputs(
    TensorOrRemoteBufferVectors& inputs);

/* Generate a JSON struture describing the tensor mappings
 */
std::string GetTensorMappingJson(const std::string& module_name,
                                 const poplar::Graph& graph,
                                 const TensorMaps& tensor_map);

/* Create an inputs / outputs metadata structure from the compiler resources
 */
StatusOr<ipu::Metadata> CreateExecutableMetadata(
    const InputOutputAliasingMap& io_map,
    const CanonicalInfeedInfos& infeed_infos,
    const CanonicalOutfeedInfos& outfeed_infos, uint32 replication_count,
    const poplar::OptionFlags& device_opts,
    const poplar::OptionFlags& engine_opts, const poplar::Target& target);

// Zero the given remote buffer at the given repeat offset.
void ZeroRemoteBuffer(CompilerResources& res, poplar::Graph& graph,
                      poplar::RemoteBuffer& remote_buffer, int64 offset,
                      poplar::program::Sequence& sequence,
                      const poplar::DebugNameAndId& debug_name_and_id);

// Zero the given tensors efficiently.
void ZeroTensors(CompilerResources& res, poplar::Graph& graph,
                 const std::vector<poplar::Tensor>& tensors,
                 poplar::program::Sequence& sequence,
                 const poplar::DebugNameAndId& debug_name_and_id);

// Functor to hash a poplar type.
struct PoplarTypeHasher {
  inline std::size_t operator()(const poplar::Type& t) const noexcept {
    return std::hash<std::string>()(t.toString());
  }
};

void SetRuntimeReplicaOptions(poplar::OptionFlags* option_flags,
                              int64 process_index, int64 process_count,
                              int64 global_replication_factor);

bool HasIOTiles(CompilerResources& res);

}  // namespace poplarplugin
}  // namespace xla

#endif
