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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

#include <poplar/Program.hpp>
#include <poplin/Convolution.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Expr.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"

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

struct CompilerResources;
class SubComputationVisitor;

class PoplarExecutor;

using TensorKey = std::pair<std::string, int64>;
using TensorMap = std::map<TensorKey, poplar::Tensor>;
using TensorMaps = std::map<std::string, TensorMap>;

using OutVector = std::vector<poplar::Tensor>;
using ArgVector = std::vector<poplar::Tensor>;
using ArgVectors = std::vector<ArgVector>;

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
Status PoplarExceptionToTensorflowStatus(const std::string& prefix,
                                         const std::exception& e);

void SetFlagIfNotPresent(poplar::OptionFlags& opts, const std::string& key,
                         const std::string& value);

poplar::OptionFlags GetReplicateAllReduceOptions();

// Try and dump the profiler report to a file if a OOM exception occurs.
void DumpIfPoplarOutOfMemoryAllocationException(const PoplarExecutor*);

/* Optimization tests */

bool IsPoplibsPool(const HloInstruction*, const HloComputation*);

bool IsSimpleSelection(const HloComputation*);

bool IsReducableArtithmetic(const HloComputation*);

StatusOr<bool> IsParallelMap(const HloInstruction*, const HloComputation*);

poplar::OptionFlags GetConvolutionOptionsForType(CompilerResources& res,
                                                 const MLType conv_type);

poplar::OptionFlags GetMatMulOptionsForType(CompilerResources& res,
                                            const MLType conv_type);
}  // namespace poplarplugin
}  // namespace xla

#endif
