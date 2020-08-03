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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/seed.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSeed::HloSeed(const Shape& shape)
    : HloPoplarInstruction(shape, {}, PoplarOp::Seed) {
  set_custom_call_has_side_effect(true);
}

std::vector<std::string> HloSeed::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

std::unique_ptr<HloInstruction> HloSeed::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const>,
    HloCloneContext*) const {
  return CreateSeed(shape);
}

std::unique_ptr<HloInstruction> CreateSeed(const Shape& shape) {
  return absl::make_unique<HloSeed>(shape);
}

namespace {

static HloPoplarInstructionFactory seed_factory(
    PoplarOp::Seed,
    [](HloCustomCallInstruction* call) { return CreateSeed(call->shape()); });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
