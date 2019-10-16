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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>

namespace xla {
namespace poplarplugin {

namespace {

StatusOr<string> FindAttribute(const FrontendAttributes& attributes,
                               const string& key) {
  const auto& map = attributes.map();
  const auto found = map.find(key);
  if (found == map.end()) {
    return tensorflow::errors::NotFound("Frontend attribute: ", key);
  }
  return found->second;
}

}  // namespace

StatusOr<poplar::program::Program> CreateSendDone(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(res, inst);

  const auto* send_done = Cast<HloSendDoneInstruction>(inst);
  const auto* send = Cast<HloSendInstruction>(send_done->operand(0));

  TF_ASSIGN_OR_RETURN(const poplar::Tensor tensor,
                      FindInstructionInput(tensor_map, res, send, 0, seq));

  TF_ASSIGN_OR_RETURN(
      const string rendezvous_key,
      FindAttribute(send_done->frontend_attributes(), "rendezvous_key"));

  // Use the rendezvous key also for the Poplar stream handle.
  const poplar::DataStream stream = graph.addDeviceToHostFIFO(
      rendezvous_key, tensor.elementType(), tensor.numElements());

  seq.add(poplar::program::Copy(tensor, stream));

  const Shape& shape = send->operand(0)->shape();

  res.annotations.send_infos.emplace_back(stream.handle(), rendezvous_key,
                                          shape);

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
