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

#include <poplar/Graph.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"

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

StatusOr<poplar::program::Program> CreateRecvDone(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(res, inst);

  const auto* recv_done = Cast<HloRecvDoneInstruction>(inst);

  TF_ASSIGN_OR_RETURN(
      const string rendezvous_key,
      FindAttribute(recv_done->frontend_attributes(), "rendezvous_key"));

  const xla::Shape recv_shape =
      ShapeUtil::GetTupleElementShape(recv_done->shape(), 0);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor tensor,
      AddTensor(graph, TensorLocation{inst, 0}, recv_shape, res, tensor_map));

  // Use the rendezvous key also for the Poplar stream handle.
  const poplar::DataStream stream = graph.addHostToDeviceFIFO(
      rendezvous_key, tensor.elementType(), tensor.numElements(),
      poplar::ReplicatedStreamMode::BROADCAST);

  seq.add(poplar::program::Copy(stream, tensor));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, tensor));

  res.annotations.recv_infos.emplace_back(stream.handle(), rendezvous_key,
                                          recv_shape);
  return seq;
}

StatusOr<poplar::program::Program> CreateSendDone(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(res, inst);

  const auto* send_done = Cast<HloSendDoneInstruction>(inst);
  const auto* send = Cast<HloSendInstruction>(send_done->operand(0));

  TF_ASSIGN_OR_RETURN(const poplar::Tensor tensor,
                      FindInstructionInput(tensor_map, res, send_done, 0, seq));

  TF_ASSIGN_OR_RETURN(
      const string rendezvous_key,
      FindAttribute(send_done->frontend_attributes(), "rendezvous_key"));

  const auto replica_handling =
      FindAttribute(send_done->frontend_attributes(), "replica_handling");
  const bool concat_replicas =
      replica_handling.ok() && replica_handling.ValueOrDie() == "Concat";

  // Use the rendezvous key also for the Poplar stream handle.
  const poplar::DataStream stream = graph.addDeviceToHostFIFO(
      rendezvous_key, tensor.elementType(), tensor.numElements());

  seq.add(poplar::program::Copy(tensor, stream));

  const Shape& shape = send->operand(0)->shape();

  res.annotations.send_infos.emplace_back(stream.handle(), rendezvous_key,
                                          shape, concat_replicas);

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
