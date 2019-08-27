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
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/core/platform/protobuf.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"

namespace xla {
namespace poplarplugin {
namespace generic_graph_caching {

using namespace poputil::graphfn;

size_t GenericGraphCache::HloInstructionHash::operator()(
    const HloInstruction* inst) const {
  return inst->Hash();
}

// The instruction will produce the same graph if it is identical apart from:
// 1. Operand shapes - we just require to have the same shape.
// 2. We ignore the inplace field in backend config.
bool GenericGraphCache::HloInstructionEquals::operator()(
    const HloInstruction* a, const HloInstruction* b) const {
  auto compare_operands = [](const HloInstruction* operand_a,
                             const HloInstruction* operand_b) {
    return operand_a->shape() == operand_b->shape();
  };

  auto compare_comps = [](const HloComputation* comp_a,
                          const HloComputation* comp_b) {
    return *comp_a == *comp_b;
  };
  auto compare_backend_configs = [](const std::string& raw_backend_config_a,
                                    const std::string& raw_backend_config_b) {
    PoplarBackendConfig backend_config_a;
    auto parse_a_status = tensorflow::HumanReadableJsonToProto(
        raw_backend_config_a, &backend_config_a);
    PoplarBackendConfig backend_config_b;
    auto parse_b_status = tensorflow::HumanReadableJsonToProto(
        raw_backend_config_b, &backend_config_b);
    if (!parse_a_status.ok() || !parse_b_status.ok()) {
      LOG(FATAL) << "Could not parse PoplarBackendConfig.";
    }
    // Ignore inplace field.
    backend_config_a.set_is_inplace(false);
    backend_config_b.set_is_inplace(false);
    return protobuf_util::ProtobufEquals(backend_config_a, backend_config_b);
  };
  return a->Identical(*b, compare_operands, compare_comps, false,
                      compare_backend_configs) &&
         GetSingleShardingDeviceId(a) == GetSingleShardingDeviceId(b);
}

Status GenericGraphCache::ExecuteCached(const HloInstruction* inst,
                                        poplar::Graph& graph,
                                        poplar::program::Sequence& seq,
                                        PoplarFunction func,
                                        Signature signature,
                                        std::vector<poplar::Tensor>& args) {
  // Check if we have already executed this instruction.
  auto itr = table_.find(inst);
  if (itr != table_.end()) {
    // We have a cached graph for this dot operation.
    itr->second(args, seq);
  } else {
    auto void_func = VoidFunction(graph, signature, func);
    void_func(args, seq);
    table_.insert({inst, std::move(void_func)});
  }
  return Status::OK();
}
}  // namespace generic_graph_caching
}  // namespace poplarplugin
}  // namespace xla
