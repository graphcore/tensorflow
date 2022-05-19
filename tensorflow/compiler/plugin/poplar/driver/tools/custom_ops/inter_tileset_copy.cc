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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloInterTilesetCopy::HloInterTilesetCopy(HloInstruction* operand)
    : HloPoplarInstruction(operand->shape(), {operand},
                           PoplarOp::InterTilesetCopy) {}

absl::flat_hash_set<int64_t> HloInterTilesetCopy::AllocatingIndices() const {
  if (IsCopyToIoTiles()) {
    // A copy *to* the IO tiles has an input tensor placed on the compute tiles,
    // and we do not have a tensor layout preference there.
    return {};
  }

  // However, for copies *from* the IO tiles, the tensor is placed on IO tiles
  // and we would like allocate it with a layout suitable for IO.
  return {0};
}

bool HloInterTilesetCopy::AllocatingOutput() const { return true; }

absl::flat_hash_map<int64_t, int64_t> HloInterTilesetCopy::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloInterTilesetCopy::GetUseDescriptions() const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloInterTilesetCopy::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloInterTilesetCopy::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloInterTilesetCopy::AllowNonInplaceLowering() const { return false; }

bool HloInterTilesetCopy::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloInterTilesetCopy::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return CreateInterTilesetCopy(new_operands[0]);
}

std::vector<std::string> HloInterTilesetCopy::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

bool HloInterTilesetCopy::IsCopyToIoTiles() const {
  const auto dst_tileset = GetTileset(this);
  TF_CHECK_OK(dst_tileset.status());
  return dst_tileset.ValueOrDie() == TILESET_IO_TILES;
}

std::unique_ptr<HloInstruction> CreateInterTilesetCopy(
    HloInstruction* operand) {
  return absl::make_unique<HloInterTilesetCopy>(operand);
}

}  // namespace poplarplugin
}  // namespace xla
