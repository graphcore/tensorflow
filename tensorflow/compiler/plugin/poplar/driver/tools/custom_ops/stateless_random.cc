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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateless_random.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloStatelessRandom::HloStatelessRandom(
    const Shape& shape, absl::Span<HloInstruction* const> operands, PoplarOp op)
    : HloPoplarInstruction(shape, operands, op) {
  set_custom_call_has_side_effect(true);
}

std::vector<std::string> HloStatelessRandom::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

HloStatelessRandomUniform::HloStatelessRandomUniform(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    float min_val, float max_val)
    : HloStatelessRandom(shape, operands, PoplarOp::StatelessRandomUniform),
      min_val_(min_val),
      max_val_(max_val) {}

HloStatelessRandomUniformInt::HloStatelessRandomUniformInt(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(shape, operands, PoplarOp::StatelessRandomUniformInt) {
}

HloStatelessRandomNormal::HloStatelessRandomNormal(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(shape, operands, PoplarOp::StatelessRandomNormal) {}

HloStatelessTruncatedNormal::HloStatelessTruncatedNormal(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(shape, operands, PoplarOp::StatelessTruncatedNormal) {}

namespace {

static HloPoplarInstructionFactory stateless_random_uniform_factory(
    PoplarOp::StatelessRandomUniform,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(float min_val,
                          attribute_map.GetAttributeAsFloat("min_val"));
      TF_ASSIGN_OR_RETURN(float max_val,
                          attribute_map.GetAttributeAsFloat("max_val"));

      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomUniform>(
              call->shape(), call->operands(), min_val, max_val);
      return inst;
    });

static HloPoplarInstructionFactory stateless_random_uniform_int_factory(
    PoplarOp::StatelessRandomUniformInt, [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomUniformInt>(call->shape(),
                                                          call->operands());
      return inst;
    });

static HloPoplarInstructionFactory stateless_random_normal_factory(
    PoplarOp::StatelessRandomNormal, [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomNormal>(call->shape(),
                                                      call->operands());
      return inst;
    });

static HloPoplarInstructionFactory stateless_truncated_normal_factory(
    PoplarOp::StatelessTruncatedNormal, [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessTruncatedNormal>(call->shape(),
                                                         call->operands());
      return inst;
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
