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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/user_op_hlo.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloUserOpInstruction::HloUserOpInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    std::string path, void* function_ptr)
    : HloPoplarInstruction(
          shape, inputs,
          GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::UserOp),
          function_ptr_),
      function_ptr_(function_ptr),
      gp_path(path) {
  set_custom_call_has_side_effect(true);

  num_inputs_ = inputs.size();
}

absl::flat_hash_set<int64> HloUserOpInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloUserOpInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloUserOpInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloUserOpInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloUserOpInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateUserOp(new_operands, shape, this->GetPath(),
                      this->GetPointerToFunc());
}

std::unique_ptr<HloInstruction> CreateUserOp(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    std::string gp_path, void* function_ptr) {
  return absl::make_unique<HloUserOpInstruction>(inputs, shape, gp_path,
                                                 function_ptr);
}

namespace {

static HloPoplarInstructionFactory user_op_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::UserOp),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<xla::HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(
          uint64 pointer_to_function_as_int,
          attribute_map.GetAttributeAsUInt64("pointer_to_function_as_int"));

      TF_ASSIGN_OR_RETURN(std::string gp_path,
                          attribute_map.GetAttributeAsString("gp_path"));

      void* function_ptr = reinterpret_cast<void*>(pointer_to_function_as_int);
      return CreateUserOp(call->operands(), call->shape(), gp_path,
                          function_ptr);
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
