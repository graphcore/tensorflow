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

#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
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
namespace pe = popops::expr;

int64 GlobalRandomValue() {
  static auto* mu = new tensorflow::mutex();
  static std::mt19937_64 rng{42};
  tensorflow::mutex_lock l(*mu);
  return rng();
}

// Popops version of boost::hash_combine.
poplar::Tensor HashCombine(poplar::Graph& graph, poplar::program::Sequence& seq,
                           const poplar::Tensor& hash,
                           const poplar::Tensor& combine,
                           const std::string& debug_name) {
  return popops::map(
      graph,
      pe::_1 ^ (pe::_2 + pe::Const(0x9e3779b9U) + (pe::_1 << pe::Const(6U)) +
                (pe::_1 >> pe::Const(2U))),
      {hash, combine}, seq, debug_name + "/HashCombine");
}

class DropoutOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const std::string debug_name = GetDebugName(inst);
    poplar::program::Sequence seq;
    // Get the "x" tensor, aka the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, false));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in_seed,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    const HloDropoutInstruction* dropout_instruction =
        Cast<HloDropoutInstruction>(inst);

    double rate = dropout_instruction->Rate();
    double scale = dropout_instruction->Scale();
    bool is_user_seed = dropout_instruction->IsUserSeed();
    int32_t seed_modifier = dropout_instruction->SeedModifier();

    // By default we will use any seed provided by the user.
    poplar::Tensor initial_seed = in_seed;

    // If we aren't using a user provided seed we need to create one.
    if (!is_user_seed) {
      initial_seed = graph.addVariable(poplar::INT, {2}, debug_name + "/Seed");
      graph.setTileMapping(initial_seed, 0);
      const int64 seed_value = GlobalRandomValue();
      int32 seed_array[2];
      std::memcpy(seed_array, &seed_value, sizeof(int64));
      graph.setInitialValue<int32>(initial_seed, seed_array);
    }

    // Dropout expects an unsigned int but tensorflow takes in int32 when
    // targeting IPU.
    poplar::Tensor initial_seed_unsigned =
        initial_seed.reinterpret(poplar::UNSIGNED_INT);

    // To make sure the seed value changes, get the execution counter and hash
    // the initial seed with the execution counter.
    TF_ASSIGN_OR_RETURN(poplar::Tensor execution_counter,
                        GetExecutionCounter(res, inst));
    poplar::Tensor seed =
        HashCombine(graph, seq, initial_seed_unsigned, execution_counter,
                    debug_name + "/ExecutionCounter");

    // Also hash in the IPU number so each replica has a different seed.
    poplar::Tensor replica_constant =
        graph.addReplicationIndexConstant().reshape({1});
    graph.setTileMapping(replica_constant, 0);
    seed = HashCombine(graph, seq, seed, replica_constant,
                       debug_name + "/ReplicationCounter");

    // Create an empty tensor for the dropout. This is internal to the poprand
    // implementation but is exposed anyway so we need to provide it.
    TF_ASSIGN_OR_RETURN(poplar::Tensor reference,
                        AddPlainTensor(graph, debug_name + "/Reference",
                                       inst->operand(0)->shape(), res, false));

    // Perform the actual dropout by calling into the poprand function.
    poplar::Tensor final_output =
        poprand::dropout(graph, &seed, seed_modifier, input, reference, rate,
                         scale, seq, debug_name);

    seed = seed.reinterpret(poplar::INT);
    // Mark that tensor as our output.
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, final_output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, seed));

    return seq;
  }
};

REGISTER_POPLAR_OP(Dropout, DropoutOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
