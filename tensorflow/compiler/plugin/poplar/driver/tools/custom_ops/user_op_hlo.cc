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
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace poplarplugin {
namespace {

#define TF_ASSIGN_OR_DEFAULT(lhs, rexpr, default_value)                        \
  TF_ASSIGN_OR_DEFAULT_IMPL(                                                   \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr, \
      default_value)

#define TF_ASSIGN_OR_DEFAULT_IMPL(statusor, lhs, rexpr, default_value) \
  auto statusor = (rexpr);                                             \
  lhs = (statusor.ok()) ? statusor.ValueOrDie() : (default_value)

static constexpr int32 kApiLevel = 5;

StatusOr<void*> GetSymbolAddress(void* handle, const std::string& symbol_name) {
  void* ptr = nullptr;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetSymbolFromLibrary(
      handle, symbol_name.c_str(), &ptr));
  return ptr;
}
}  // namespace

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
    CHECK(!is_user_read_write_);
    auto metadata = reinterpret_cast<MetadataFn>(metadata_function_ptr_);

    metadata(metadata_.allocating_indices_,
             metadata_.replica_identical_output_indices_,
             metadata_.input_to_output_tensor_aliasing_,
             metadata_.is_elementwise_, stateless, metadata_.is_hashable_,
             num_inputs_);
  }
  set_custom_call_has_side_effect(!stateless);
}

absl::flat_hash_set<int64> HloUserOpInstruction::AllocatingIndices() const {
  absl::flat_hash_set<int64> indices(metadata_.allocating_indices_.begin(),
                                     metadata_.allocating_indices_.end());
  return indices;
}

absl::flat_hash_set<int64> HloUserOpInstruction::ReplicaIdenticalOutputIndices()
    const {
  absl::flat_hash_set<int64> indices(
      metadata_.replica_identical_output_indices_.begin(),
      metadata_.replica_identical_output_indices_.end());
  return indices;
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

bool HloUserOpInstruction::AllowNonInplaceLowering() const { return false; }

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

      TF_ASSIGN_OR_RETURN(std::string op_name,
                          attribute_map.GetAttributeAsString("op_name"));

      TF_ASSIGN_OR_RETURN(std::string library_path,
                          attribute_map.GetAttributeAsString("library_path"));

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

      // Load the relevant symbols.
      void* handle;
      TF_RETURN_IF_ERROR(tensorflow::Env::Default()->LoadLibrary(
          library_path.c_str(), &handle));

      TF_ASSIGN_OR_DEFAULT(void* api_level_ptr,
                           GetSymbolAddress(handle, "custom_op_api_level"),
                           nullptr);

      const int32 api_level =
          api_level_ptr ? *reinterpret_cast<const int32*>(api_level_ptr) : 0;

      if (api_level != kApiLevel) {
        return InternalErrorStrCat("Api level of module ", library_path,
                                   ", op name ", op_name, " is ", api_level,
                                   ", expected ", kApiLevel,
                                   ". See section `API Level Versioning` in "
                                   "documentation for more details.");
      }

      TF_ASSIGN_OR_RETURN(void* operation_fn_ptr,
                          GetSymbolAddress(handle, op_name));

      void* metadata_function_ptr = nullptr;
      void* allocator_function_ptr = nullptr;
      if (!is_user_read_write) {
        TF_ASSIGN_OR_DEFAULT(metadata_function_ptr,
                             GetSymbolAddress(handle, op_name + "_metadata"),
                             nullptr);

        TF_ASSIGN_OR_DEFAULT(allocator_function_ptr,
                             GetSymbolAddress(handle, op_name + "_allocator"),
                             nullptr);
      }

      return CreateUserOp(
          call->operands(), call->shape(), gp_path, operation_fn_ptr,
          metadata_function_ptr, allocator_function_ptr, gradient_size,
          partial_derivative_index, is_user_read_write, attributes);
    });

#undef TF_ASSIGN_OR_DEFAULT
#undef TF_ASSIGN_OR_DEFAULT_IMPL
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
