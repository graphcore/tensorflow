#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_UTIL_H_

/*
 * These functions are related to poplar, and cannot be used within the
 * optimizers target in the BUILD file.
 */

#include <vector>

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

StatusOr<std::shared_ptr<SubComputationVisitor>> GetOrCompileSubComputation(
    CompilerResources& res, const ArgVectors& inputs,
    const HloComputation* comp,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations =
        {});

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

}  // namespace poplarplugin
}  // namespace xla

#endif
