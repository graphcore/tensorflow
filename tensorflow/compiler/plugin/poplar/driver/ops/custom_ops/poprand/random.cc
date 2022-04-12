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

#include <poplar/DebugContext.hpp>
#include <popops/ElementWise.hpp>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateless_random.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace poplarplugin {
namespace {

class TruncatedNormalOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TruncatedNormalOp");
    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map, {debug_info, "ref"}));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    poplar::program::Sequence seq({}, debug_info);
    auto out = poprand::truncatedNormal(graph, nullptr, 0, ref, dtype, 0.0, 1.0,
                                        1.0, seq, {debug_info});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(TruncatedNormal, TruncatedNormalOp);

class StatelessRandomUniformOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "StatelessRandomUniformOp");
    poplar::program::Sequence seq({}, debug_info);
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor seed,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    // The reference must be mapped linearly, this will define which tiles are
    // generating the random numbers. Different tiles won't generate the same
    // numbers from the same seed (as each tile mutates it by the modifier).
    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddPlainTensor(graph, {debug_info, "Reference"},
                                       output_shape, res, false));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    const HloStatelessRandomUniform* as_stateless_random =
        Cast<HloStatelessRandomUniform>(inst);

    assert(as_stateless_random &&
           "Expected operation to be an "
           "xla::poplarplugin::HloStatelessRandomUniform");

    double min_val = static_cast<double>(as_stateless_random->GetMin());
    double max_val = static_cast<double>(as_stateless_random->GetMax());

    auto out = poprand::uniform(graph, &seed, 0, ref, dtype, min_val, max_val,
                                seq, {debug_info});

    // If this operation has an allocation target allocate a tensor of that
    // layout and copy the result into it after the random numbers have been
    // generated.
    if (HasTensorAllocationTarget(TensorLocation{inst, 0}, res)) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor new_out,
          AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                    tensor_map, {debug_info, "out"}));
      seq.add(poplar::program::Copy(out, new_out, false, {debug_info}));
      out = new_out;
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(StatelessRandomUniform, StatelessRandomUniformOp);

class StatelessRandomUniformIntOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "StatelessRandomUniformIntOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor seed,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    const HloInstruction* lower = inst->operand(1);
    CHECK_EQ(lower->opcode(), HloOpcode::kConstant);
    const HloInstruction* upper = inst->operand(2);
    CHECK_EQ(upper->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(int lower_val,
                        LiteralScalarToNativeType<int>(lower->literal()));
    TF_ASSIGN_OR_RETURN(int upper_val,
                        LiteralScalarToNativeType<int>(upper->literal()));

    // The reference must be mapped linearly, this will define which tiles are
    // generating the random numbers. Different tiles won't generate the same
    // numbers from the same seed (as each tile mutates it by the modifier).
    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddPlainTensor(graph, {debug_info, "Reference"},
                                       output_shape, res, false));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::uniform(graph, &seed, 0, ref, dtype, lower_val,
                                upper_val, seq, {debug_info});

    // If this operation has an allocation target allocate a tensor of that
    // layout and copy the result into it after the random numbers have been
    // generated.
    if (HasTensorAllocationTarget(TensorLocation{inst, 0}, res)) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor new_out,
          AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                    tensor_map, {debug_info, "out"}));
      seq.add(poplar::program::Copy(out, new_out, false, {debug_info}));
      out = new_out;
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(StatelessRandomUniformInt, StatelessRandomUniformIntOp);

class StatelessRandomNormalOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "StatelessRandomNormalOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor seed,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    // The reference must be mapped linearly, this will define which tiles are
    // generating the random numbers. Different tiles won't generate the same
    // numbers from the same seed (as each tile mutates it by the modifier).
    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddPlainTensor(graph, {debug_info, "Reference"},
                                       output_shape, res, false));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::normal(graph, &seed, 0, ref, dtype, 0.0, 1.0, seq,
                               {debug_info});

    // If this operation has an allocation target allocate a tensor of that
    // layout and copy the result into it after the random numbers have been
    // generated.
    if (HasTensorAllocationTarget(TensorLocation{inst, 0}, res)) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor new_out,
          AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                    tensor_map, {debug_info, "out"}));
      seq.add(poplar::program::Copy(out, new_out, false, {debug_info}));
      out = new_out;
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(StatelessRandomNormal, StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "StatelessTruncatedNormalOp");
    poplar::program::Sequence seq({}, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor seed,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    // The reference must be mapped linearly, this will define which tiles are
    // generating the random numbers. Different tiles won't generate the same
    // numbers from the same seed (as each tile mutates it by the modifier).
    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddPlainTensor(graph, {debug_info, "Reference"},
                                       output_shape, res, false));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::truncatedNormal(graph, &seed, 0, ref, dtype, 0.0, 1.0,
                                        1.0, seq, {debug_info});

    // If this operation has an allocation target allocate a tensor of that
    // layout and copy the result into it after the random numbers have been
    // generated.
    if (HasTensorAllocationTarget(TensorLocation{inst, 0}, res)) {
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor new_out,
          AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                    tensor_map, {debug_info, "out"}));
      seq.add(poplar::program::Copy(out, new_out, false, {debug_info}));
      out = new_out;
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(StatelessTruncatedNormal, StatelessTruncatedNormalOp);

class SeedOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "SeedOp");
    poplar::program::Sequence seq({}, debug_info);
    TF_ASSIGN_OR_RETURN(poplar::Tensor seed_ref,
                        AddPlainTensor(graph, {debug_info, "SeedRef"},
                                       output_shape, res, false));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    if (dtype != poplar::INT) {
      return UnimplementedStrCat(
          "Only integer seeds are supported, but requested ",
          output_shape.ToString());
    }

    poplar::Tensor seed = poprand::uniform(
        graph, nullptr, 1U, seed_ref, dtype, std::numeric_limits<int32>::min(),
        std::numeric_limits<int32>::max(), seq, {debug_info, "GenerateSeed"});

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(seed, graph)));
    return seq;
  }
};
REGISTER_POPLAR_OP(Seed, SeedOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
