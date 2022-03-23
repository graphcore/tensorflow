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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REDUCTION_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REDUCTION_UTIL_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"

#include <gcl/Collectives.hpp>

namespace popops {
namespace expr {
enum class BinaryOpType;
}
enum class CollectiveOperator;
enum class Operation;
}  // namespace popops

namespace xla {
namespace poplarplugin {

// Mirrors popops::Operation.
enum class ReductionOperation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND,
  LOGICAL_OR,
  SQUARE_ADD,
  LOG_ADD,
};

struct ReductionInfo {
  ReductionInfo() = default;
  ReductionInfo(const ReductionInfo&);
  ReductionInfo(ReductionInfo&&) = default;
  std::vector<std::size_t> reduction_dims;
  Literal identity_literal;
  ReductionOperation reduction_op;
  bool with_scale = false;
};

StatusOr<ReductionInfo> GetReductionInfo(const HloInstruction* inst,
                                         bool with_scale);
StatusOr<popops::Operation> ToPopopsReductionOp(const ReductionOperation&);
StatusOr<ReductionOperation> FromPopopsReductionOp(const popops::Operation&);
StatusOr<popops::expr::BinaryOpType> ToBinaryOpType(const ReductionOperation&);
StatusOr<const HloInstruction*> GetReduceInstruction(
    const HloInstruction* root_inst);
StatusOr<popops::Operation> GetPoplibsReductionOperation(
    const HloInstruction* inst);

// Returns a literal representing the identity value for op type of the given
// instruction and dtype.
Literal GetIdentityConstantLiteral(const HloInstruction* inst,
                                   const PrimitiveType& dtype);

// Convert a xla::poplarplugin::CollectiveOperator to a popops one.
StatusOr<gcl::CollectiveOperator> ToPoplarCollectiveOperator(
    CollectiveOperator op);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_REDUCTION_UTIL_H_
