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

#include <map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

namespace xla {
namespace poplarplugin {

HloUserOpInstruction::HloUserOpInstruction(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    const std::string& path, void* fn_ptr, void* metadata_fn_ptr,
    void* allocator_function_ptr, int64 gradient_size,
    int64 partial_derivative_index, bool is_user_read_write,
    const std::string& attributes)
    : HloPoplarInstruction(shape, inputs, PoplarOp::UserOp, fn_ptr,
                           metadata_fn_ptr, allocator_function_ptr, path,
                           gradient_size, partial_derivative_index,
                           is_user_read_write, attributes),
      function_ptr_(fn_ptr),
      metadata_function_ptr_(metadata_fn_ptr),
      allocator_function_ptr_(allocator_function_ptr),
      gp_path_(path),
      gradient_size_(gradient_size),
      partial_derivative_index_(partial_derivative_index),
      is_user_read_write_(is_user_read_write),
      attributes_(attributes) {
  num_inputs_ = inputs.size();
  CHECK(absl::c_all_of(inputs, [](const HloInstruction* inst) {
    return inst->shape().IsArray();
  }));

  // If there is a metadata function, call it to populate the metadata_ struct.
  bool stateless = false;

  if (metadata_function_ptr_ != nullptr) {
    void (*metadataSignature)(
        std::vector<std::int64_t> & allocating_indices,
        std::map<std::int64_t, std::int64_t> & input_to_output_tensor_aliasing,
        bool& is_elementwise, bool& is_stateless, bool& is_hashable,
        std::uint32_t num_inputs);

    metadataSignature =
        reinterpret_cast<decltype(metadataSignature)>(metadata_function_ptr_);

    metadataSignature(metadata_.allocating_indices_,
                      metadata_.input_to_output_tensor_aliasing_,
                      metadata_.is_elementwise_, stateless,
                      metadata_.is_hashable_, num_inputs_);
  }
  set_custom_call_has_side_effect(!stateless);
}

absl::flat_hash_set<int64> HloUserOpInstruction::AllocatingIndices() const {
  absl::flat_hash_set<int64> set;
  for (std::int64_t i : metadata_.allocating_indices_) {
    set.insert({i});
  }
  return set;
}

bool HloUserOpInstruction::AllocatingOutput() const { return IsReadWrite(); }

absl::flat_hash_map<int64, int64> HloUserOpInstruction::LayoutDependencies()
    const {
  return {};
}

HloPoplarUseDescriptions HloUserOpInstruction::GetUseDescriptions() const {
  HloPoplarUseDescriptions descriptions;
  for (auto pair : metadata_.input_to_output_tensor_aliasing_) {
    descriptions.push_back(HloPoplarUseDescription{
        pair.first, /*operand_index=*/ShapeIndex{}, ShapeIndex{pair.second},
        BufferUseKind::USE_ALIAS_READ_WRITE});
  }
  return descriptions;
}

HloPoplarBufferDescriptions HloUserOpInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllUnaliasedBuffers(this,
                                                        GetUseDescriptions());
}

const FindConsumersExtensionResults HloUserOpInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloUserOpInstruction::IsPopOpsElementwise() const {
  return metadata_.is_elementwise_;
}

std::vector<string> HloUserOpInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::stringstream ss;
  ss << function_ptr_;
  std::string function_ptr_address = ss.str();
  ss.clear();

  ss << metadata_function_ptr_;
  std::string metadata_ptr_address = ss.str();
  ss.clear();

  ss << allocator_function_ptr_;
  std::string allocator_ptr_address = ss.str();
  ss.clear();

  std::vector<string> attributes;
  attributes.push_back(absl::StrCat("function_ptr=", function_ptr_address));
  attributes.push_back(absl::StrCat("metadata_ptr=", metadata_ptr_address));
  attributes.push_back(absl::StrCat("allocator_ptr=", allocator_ptr_address));

  attributes.push_back(
      absl::StrCat("metadata_.is_elementwise_=", metadata_.is_elementwise_));
  attributes.push_back(
      absl::StrCat("metadata_.is_hashable_=", metadata_.is_hashable_));
  attributes.push_back(absl::StrCat(
      "metadata_.input_to_output_tensor_aliasing_=",
      absl::StrJoin(
          metadata_.input_to_output_tensor_aliasing_, ", ",
          [](std::string* result,
             const std::pair<std::int64_t, std::int64_t>& alias_info) {
            result->append(absl::StrCat("[from ", alias_info.first, " to ",
                                        alias_info.second, "]"));
          })));

  attributes.push_back(absl::StrCat("num_inputs_=", num_inputs_));
  attributes.push_back(absl::StrCat("gp_path=", gp_path_));
  attributes.push_back(
      absl::StrCat("partial_derivative_index=", partial_derivative_index_));
  attributes.push_back(
      absl::StrCat("is_user_read_write=", is_user_read_write_));
  attributes.push_back(absl::StrCat("attributes=", attributes_));

  return attributes;
}

std::unique_ptr<HloInstruction> HloUserOpInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateUserOp(new_operands, shape, GetPath(), function_ptr_,
                      metadata_function_ptr_, allocator_function_ptr_,
                      gradient_size_, partial_derivative_index_,
                      is_user_read_write_, attributes_);
}

std::unique_ptr<HloInstruction> CreateUserOp(
    absl::Span<HloInstruction* const> inputs, const Shape& shape,
    const std::string& gp_path, void* function_ptr, void* metadata_function_ptr,
    void* allocator_function_ptr, int64 gradient_size,
    int64 partial_derivative_index, bool is_user_read_write,
    const std::string& attributes) {
  return absl::make_unique<HloUserOpInstruction>(
      inputs, shape, gp_path, function_ptr, metadata_function_ptr,
      allocator_function_ptr, gradient_size, partial_derivative_index,
      is_user_read_write, attributes);
}

namespace {

static HloPoplarInstructionFactory user_op_factory(
    PoplarOp::UserOp,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<xla::HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(uint64 operation_fn,
                          attribute_map.GetAttributeAsUInt64("operation_fn"));
      void* operation_fn_ptr = reinterpret_cast<void*>(operation_fn);

      TF_ASSIGN_OR_RETURN(
          uint64 metadata_function,
          attribute_map.GetAttributeAsUInt64("metadata_function"));
      void* metadata_function_ptr = reinterpret_cast<void*>(metadata_function);

      TF_ASSIGN_OR_RETURN(
          uint64 allocator_function,
          attribute_map.GetAttributeAsUInt64("allocator_function"));
      void* allocator_function_ptr =
          reinterpret_cast<void*>(allocator_function);

      TF_ASSIGN_OR_RETURN(std::string gp_path,
                          attribute_map.GetAttributeAsString("gp_path"));

      TF_ASSIGN_OR_RETURN(int64 gradient_size,
                          attribute_map.GetAttributeAsInt64("gradient_size"));

      TF_ASSIGN_OR_RETURN(
          int64 partial_derivative_index,
          attribute_map.GetAttributeAsInt64("partial_derivative_index"));

      TF_ASSIGN_OR_RETURN(
          bool is_user_read_write,
          attribute_map.GetAttributeAsBool("is_user_read_write"));

      TF_ASSIGN_OR_RETURN(std::string attributes,
                          attribute_map.GetAttributeAsString("attributes"));

      return CreateUserOp(
          call->operands(), call->shape(), gp_path, operation_fn_ptr,
          metadata_function_ptr, allocator_function_ptr, gradient_size,
          partial_derivative_index, is_user_read_write, attributes);
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
