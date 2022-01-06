/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/xla_util.h"

#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

xla::XlaOp CombineHashes(xla::XlaOp from, xla::XlaOp to) {
  xla::XlaBuilder* builder = from.builder();
  Shape from_shape = builder->GetShape(from).ConsumeValueOrDie();
  from = xla::BitcastConvertType(from, U32);
  to = xla::BitcastConvertType(to, U32);
  // Create constants
  XlaOp large_constant =
      xla::ConstantLiteral(builder, LiteralUtil::CreateR0<uint32>(0x9E3779B9U));
  XlaOp six = xla::ConstantLiteral(builder, LiteralUtil::CreateR0<uint32>(6U));
  XlaOp two = xla::ConstantLiteral(builder, LiteralUtil::CreateR0<uint32>(2U));
  auto from_dims = from_shape.dimensions();
  auto to_dims = builder->GetShape(to).ConsumeValueOrDie().dimensions();

  // to + 0x9E3779B9U
  XlaOp rhs = xla::Add(to, xla::Broadcast(large_constant, to_dims));
  // (to + 0x9E3779B9U) + (seed << 6)
  rhs = xla::Add(xla::Broadcast(rhs, from_dims),
                 xla::ShiftLeft(from, xla::Broadcast(six, from_dims)));
  // (to + 0x9E3779B9U) + (seed << 6) + (seed >> 2)
  rhs = xla::Add(rhs,
                 xla::ShiftRightLogical(from, xla::Broadcast(two, from_dims)));
  from = xla::Xor(from, rhs);
  return xla::BitcastConvertType(from, from_shape.element_type());
}

xla::XlaOp HashSeedWithReplicaIndex(xla::XlaOp seed) {
  xla::XlaBuilder* builder = seed.builder();
  xla::XlaOp replica_index = xla::ReplicaId(builder);
  xla::XlaOp casted = xla::BitcastConvertType(replica_index, S32);
  return CombineHashes(seed, casted);
}

}  // namespace poplarplugin
}  // namespace xla
