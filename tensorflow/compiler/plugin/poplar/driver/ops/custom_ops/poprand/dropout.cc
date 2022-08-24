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
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
struct ReferenceInfo {
  ReferenceInfo(poplar::Graph& g, poplar::Tensor t) : graph(g), tensor(t) {}

  poplar::Graph& graph;
  poplar::Tensor tensor;
};

namespace pe = popops::expr;

poplar::Tensor GetReferenceTensor(const HloDropout* dropout_instruction,
                                  poplar::Graph& graph,
                                  const poplar::Tensor& input,
                                  CompilerResources& res, TensorMap& tensor_map,
                                  poplar::program::Sequence& seq,
                                  PoplarOpDefDebugInfo& debug_info) {
  // If there's a third operand, then we have an input reference tensor
  if (dropout_instruction->operand_count() == 3) {
    auto t = FindInstructionInputs(tensor_map, res, dropout_instruction, 2, seq,
                                   {debug_info}, false)[0];

    // Unpack the reference tensor.
    ReferenceInfo reference_info = absl::any_cast<ReferenceInfo>(t.AsOpaque());

    // If there's sharding, we need to make sure the reference tensor is in the
    // right virtual graph.
    if (dropout_instruction->has_sharding()) {
      const auto& dst_sharding =
          GetShardingDeviceIdVector(dropout_instruction->sharding());
      const unsigned dst_device_id = res.shard_to_ipu_id[dst_sharding[0]];
      poplar::Graph& master_graph = GetMasterGraph(res);
      reference_info.tensor = poputil::cloneToGraph(
          reference_info.graph, graph, reference_info.tensor, {},
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    }

    return reference_info.tensor;
  }

  // The default reference is just the input tensor.
  poplar::Tensor reference = input;

  // If there's noise shape, slice the reference down to the right shape.
  if (dropout_instruction->HasNoiseShape()) {
    std::vector<unsigned> permutation(reference.rank(), 0);
    absl::c_iota(permutation, 0);
    absl::c_reverse(permutation);
    reference = reference.dimShuffle(permutation);

    auto in_shape = reference.shape();
    auto ns_shape = dropout_instruction->NoiseShape();
    absl::c_reverse(ns_shape);

    // For each dimension that doesn't match, slice that dimensions down to
    // size 1.
    for (std::size_t dim = 0u; dim < std::min(ns_shape.size(), in_shape.size());
         ++dim) {
      if (ns_shape[dim] != in_shape[dim]) {
        reference = reference.slice(0, 1, dim);
      }

      // The mismatched shape should've been 1.
      CHECK_EQ(ns_shape[dim], reference.dim(dim));
    }

    // Dimshuffle the reference back.
    reference = reference.dimShuffle(permutation);
  }

  return reference;
}

class DropoutOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "DropoutOp");
    const HloDropout* dropout_instruction = Cast<HloDropout>(inst);
    const float rate = dropout_instruction->Rate();
    const float scale = dropout_instruction->Scale();

    DriverProgramSequence seq(debug_info);
    TF_ASSIGN_OR_RETURN(poplar::Tensor input,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor seed,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    // Clone the seed as it is an output.
    seed = poputil::duplicate(graph, seed, seq, {debug_info, "Seed"});

    // Dropout expects an unsigned int but tensorflow takes in int32 when
    // targeting IPU.
    poplar::Tensor seed_unsigned = seed.reinterpret(poplar::UNSIGNED_INT);

    poplar::Tensor output;

    // Create a tensor that specifies the relationship between the input tensor
    // and the tile PRNGs. This is used to guarantee reproducable masks during
    // recomputation and backpropogation.
    poplar::Tensor reference = GetReferenceTensor(
        dropout_instruction, graph, input, res, tensor_map, seq, debug_info);
    if (dropout_instruction->HasNoiseShape()) {
      // Pull out noise_shape if it is set. At this point it is a std::vector,
      // but poprand::dropout expects a poplar::Tensor*
      // noise_shape becomes a constant tensor residing on tile 0.
      output =
          poprand::shapedDropout(graph, &seed_unsigned, 1U, input, reference,
                                 rate, scale, seq, {debug_info});
    } else {
      // Perform the actual dropout by calling into the poprand function.
      output = poprand::dropout(graph, &seed_unsigned, 1U, input, reference,
                                rate, scale, seq, {debug_info});
    }

    ReferenceInfo ref_info(graph, reference);

    // Mark that tensor as our output.
    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(output, graph)));
    TF_RETURN_IF_ERROR(
        AddOutputTensor(tensor_map, inst, 1, DriverTensor(seed, graph)));
    TF_RETURN_IF_ERROR(AddOutputOpaque(tensor_map, inst, 2, {ref_info}));

    return seq;
  }
};
REGISTER_POPLAR_OP(Dropout, DropoutOp);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
