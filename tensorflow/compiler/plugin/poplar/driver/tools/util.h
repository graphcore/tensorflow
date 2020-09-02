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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

class Shape;
class Literal;
class HloSharding;
class HloInstruction;
class HloComputation;
class HloModule;

namespace poplarplugin {
namespace {
template <typename To, typename From>
bool check_convert_ok(const To& to, const From& from) {
  To from_converted = static_cast<To>(from);
  From to_converted = static_cast<From>(to);
  return from_converted == to && from == to_converted;
}
}  // namespace

template <typename To, typename From>
absl::optional<To> convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
    if (!check_convert_ok(out.back(), e)) {
      return absl::nullopt;
    }
  }
  return out;
};

template <typename To, typename From>
absl::optional<To> convert_scalar(const From& from) {
  To to = static_cast<To>(from);
  return check_convert_ok(to, from) ? absl::optional<To>(to) : absl::nullopt;
};

// Strip all Layout information from the Shapes of HloInstructions.
void StripAllInstructionLayouts(const HloModule*);

// Calculate the number of resource variable parameters
int64 GetResourceVariableParameterCount(const HloModule*);

// Check if there are any operations in the computation which have sharding
// information
bool HaveSharding(HloComputation* comp);

// Check if there are any operations in the module which have sharding
// information
bool HaveSharding(HloModule* module);

// Check whether the sharding info contains only supported sharding
bool IsSupportedSharding(const HloSharding& sharding);

// Get the sharding for a particular input operand of an instruction
HloSharding GetShardingForOperand(const HloInstruction* inst, int operand);

// Get sharding information for the output of an instruction
const HloSharding& GetShardingOfOutputTensor(const HloInstruction* inst);

// Get the vector of maximal sharding ids from the leaf nodes of a sharding
// object
std::vector<int64> GetShardingDeviceIdVector(const HloSharding& sharding);

// Get the sharding Id of the output tensor, given the knowledge that the
// sharding must be a single value.
int64 GetSingleShardingDeviceId(const HloInstruction* inst);

// Returns whether the instruction is allowed to have tuple sharding.
bool IsAllowedTupleSharding(const HloInstruction* inst);

// If `from` has sharding, then copy it to `to`
void CopyShardingIfPresent(HloInstruction* const from,
                           HloInstruction* const to);

// Count the number of leaf shapes in a shape tuple
int64 CountShapes(const Shape& shape);

// Find the index when embedding a shape into a tuple. The tuple_index is the
// index of the shape in the new tuple, and the original_index is the index
// of the tensor in the original shape.
int64 InsertIntoTuple(const Shape& tuple, int64 tuple_index,
                      int64 original_index);

std::vector<Shape> FlattenedXlaShape(const Shape& shape);

template <typename NativeT>
StatusOr<NativeT> LiteralScalarToNativeType(const Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> LiteralVectorToNativeType(const Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> WideConstToNativeType(
    const HloInstruction* wide_const);

bool IsInstructionInEntryComputation(const HloInstruction*);
bool IsPopOpsFusion(const HloComputation*, const std::string& postfix = "");
bool IsPopOpsFusion(const HloInstruction*, const std::string& postfix = "");
bool IsArithmeticExpressionFusion(const HloComputation*);
bool IsArithmeticExpressionFusion(const HloInstruction*);
bool IsRepeatLoop(const HloInstruction*);
int64 GetRepeatLoopCount(const HloInstruction*);
bool GetRepeatLoopAllowFinerAliasAnalysis(const HloInstruction*);
bool IsPipelineStage(const HloInstruction*);
bool IsPipelineStageBackward(const HloInstruction*);
bool IsPipelineStageRecomputation(const HloInstruction*);
bool IsResourceUpdate(const HloInstruction*);
bool IsFunction(const HloInstruction*);
bool IsMultiConv(const HloInstruction*);
bool IsPipelineOp(const HloInstruction*);
int64 GetPipelineStageID(const HloInstruction*);
int64 GetResourceUpdateBatchesToAccumulate(const HloInstruction*);
bool GetResourceUpdateOffloadVariables(const HloInstruction*);

bool IsSupportedSharding(const HloSharding&);

// This function returns the operand of inst at index operand_idx and if the
// operand is an inter ipu copy then it returns the operand which is being
// copied.
const HloInstruction* GetOperandLookThroughInterIpuCopy(
    const HloInstruction* inst, const int64 operand_idx);

// This function returns true if the environment variable flag
// "use_synthetic_data" has been set. Using synthetic data means that *no data*
// ill be copied to/from the device.
bool UseSyntheticData();

// This function returns true if the environment variable flag
// "synthetic_data_initializer" has been set. Using this flag means that all the
// inputs to the graph will be initialized to some constant, meaning that all
// the tensors will be always live.
bool UseSyntheticDataInitializer();

std::string GetDebugName(const HloInstruction*);

void GetAllDeps(const HloInstruction* base, std::vector<HloInstruction*>& deps);

void GetAllDepNames(const HloInstruction* base,
                    std::vector<std::string>& names);

// Configure the backend config of the instruction to indicate this instruction
// is used inplace.
void MakeUsedInplace(HloInstruction* inst);
// Configure the backend config of the instruction to indicate this instruction
// is not used inplace.
void MakeUsedNotInplace(HloInstruction* inst);
// Check whether this instruction is configured to be used inplace.
bool IsLoweredInplace(const HloInstruction* inst);

// Get all the inplace instructions in a computation.
absl::flat_hash_set<const HloInstruction*> GetInplaceInstructions(
    const HloComputation* comp);
// Get all the inplace instructions in a module.
absl::flat_hash_set<const HloInstruction*> GetInplaceInstructions(
    const HloModule* module);

HloInstruction* ConvertInstruction(HloInstruction* inst,
                                   const PrimitiveType& new_type);

HloInstruction* OutlineExpressionFromComputationWithFusion(
    absl::Span<HloInstruction* const> instructions_to_outline,
    const string& outlined_computation_name, HloComputation* computation);

// Helper for storing slice dimensions.
struct SliceInfo {
  // Dimensions we slice in.
  std::vector<size_t> sliced_dims;
  // Corresponding slice sizes.
  std::vector<size_t> slice_sizes;
};

// Given a shape and a slice, work out which dimensions are sliced.
SliceInfo GetSliceInfo(const std::vector<size_t>& shape_to_slice,
                       const std::vector<size_t>& slice_shape);

// Same as above, but for XLA shapes.
SliceInfo GetSliceInfo(const Shape& shape_to_slice, const Shape& slice_shape);

Shape GetConcatenatedShape(std::vector<HloInstruction*> insts,
                           const int64 dimension);
// Poplar's dimShuffle does: return_value.dimensions[i] =
// argument.dimensions[permutations[i]] Whereas ShapeUtil::PermuteDimensions
// does: return_value.dimensions[permutation[i]] = argument.dimensions[i].
template <typename Output, typename Input>
std::vector<Output> InvertPermutations(const std::vector<Input>& permutations) {
  std::vector<Output> result(permutations.size());

  for (Output i = 0; i < static_cast<Output>(permutations.size()); ++i) {
    result[permutations[i]] = i;
  }
  return result;
}

template <typename Input>
std::vector<unsigned> ToUnsignedVector(const std::vector<Input>& input) {
  std::vector<unsigned> results(input.size());
  for (unsigned i = 0; i < input.size(); ++i) {
    results[i] = input[i];
  }
  return results;
}

// Helper structs for hashing and comparing computations.
struct HloComputationHash {
  size_t operator()(const HloComputation* comp) const;
};
struct HloComputationEquals {
  bool operator()(const HloComputation* a, const HloComputation* b) const;
};

Status CreateDirIfMissing(const std::string& path);

}  // namespace poplarplugin
}  // namespace xla

#endif
