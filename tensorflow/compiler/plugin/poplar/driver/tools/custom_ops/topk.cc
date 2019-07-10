/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/topk.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloTopK::HloTopK(HloInstruction* input, const Shape shape, int64 K, bool sort)
    : HloPoplarInstruction(
          shape, {input},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::TopK), K,
          sort),
      num_k(K),
      sorted(sort) {}

absl::flat_hash_set<int64> HloTopK::AllocatingIndices() const { return {}; }

absl::flat_hash_map<int64, int64> HloTopK::LayoutDependencies() const {
  return {};
}

uint64 HloTopK::NumberOfInplaceOperands() const { return 0; }

bool HloTopK::IsPopOpsElementwise() const { return false; }

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateHloTopK(HloInstruction* input,
                                              const Shape& shape, int64 K,
                                              bool sorted) {
  return absl::make_unique<HloTopK>(input, shape, K, sorted);
}

std::unique_ptr<HloInstruction> HloTopK::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloTopK(operands[0], shape, NumK(), ShouldBeSorted());
}

namespace {

static HloPoplarInstructionFactory argmax_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::TopK),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      // Get the num_k and sorted attributes.
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(int64 num_k,
                          attribute_map.GetAttributeAsInt("num_k"));

      TF_ASSIGN_OR_RETURN(bool sorted,
                          attribute_map.GetAttributeAsBool("sorted"));

      return CreateHloTopK(call->mutable_operand(0), call->shape(), num_k,
                           sorted);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
