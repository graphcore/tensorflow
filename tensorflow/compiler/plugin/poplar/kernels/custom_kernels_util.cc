/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include <stdlib.h>
#include <memory>
#include <sstream>
#include <string>
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"

#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/human_readable_json.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/any.h"

namespace xla {
namespace poplarplugin {

namespace {

static inline bool NameSpaceStartsWith(const std::string& name,
                                       const char* prefix) {
  return name.find(prefix) != std::string::npos;
}

static absl::optional<PoplarOp> PoplarOpFromFusionString(
    const std::string& name) {
  PoplarOp the_op;

  std::string tmp_name = name;

  if (name.find("_pop_op_") != std::string::npos) {
    tmp_name = name.substr(8);
  }
  *tmp_name.begin() = std::toupper(*tmp_name.begin());

  auto itr = tmp_name.find(".");
  if (itr != std::string::npos) {
    tmp_name = tmp_name.substr(0, itr);
  }

  bool found = PoplarOp_Parse(tmp_name, &the_op);

  if (found) {
    return the_op;
  }
  return absl::nullopt;
}

}  // namespace

absl::optional<PoplarOp> GetPoplarCustomOp(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCustomCall) {
    PoplarOp op;
    bool op_parsed = PoplarOp_Parse(inst->custom_call_target(), &op);
    if (!op_parsed) {
      return absl::nullopt;
    }
    return op;
  } else if (IsPopOpsFusion(inst)) {
    // Look up the name of the poplar operation based on the name of the
    // computation.
    HloComputation* comp = inst->fused_instructions_computation();
    return PoplarOpFromFusionString(comp->name());
  }

  return absl::nullopt;
}

namespace IPUCustomKernelsUtil {

AttributeMap::AttributeMap() {}

AttributeMap::AttributeMap(const HloInstruction* custom_call)
    : AttributeMap(custom_call->raw_backend_config_string()) {
  CHECK_EQ(custom_call->opcode(), HloOpcode::kCustomCall);
}

AttributeMap::AttributeMap(const std::string& attributes_json) {
  Json::CharReaderBuilder builder;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool parsed = reader->parse(attributes_json.c_str(),
                              attributes_json.c_str() + attributes_json.size(),
                              &attributes_, &errs);
  if (!parsed) {
    LOG(FATAL)
        << "Could not parse the call target for custom op as JSON. Errors: "
        << errs << " Attributes: " << attributes_json;
  }
}

namespace {
template <typename T>
Json::Value GetAsJsonValue(const T& val) {
  return Json::Value(val);
}
template <>
Json::Value GetAsJsonValue(const tensorflow::DataType& val) {
  return Json::Value(DataType_Name(val));
}
template <>
Json::Value GetAsJsonValue(const int64_t& val) {
  return Json::Value(Json::Value::Int64(val));
}
template <>
Json::Value GetAsJsonValue(const uint64& val) {
  return Json::Value(Json::Value::UInt64(val));
}
}  // namespace

void AttributeMap::AddAttribute(const std::string& field_name,
                                const absl::any& attr) {
  const std::type_info& tinfo = attr.type();
  if (tinfo == typeid(std::string)) {
    auto casted_val = absl::any_cast<std::string>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);
  } else if (tinfo == typeid(float)) {
    auto casted_val = absl::any_cast<float>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(int)) {
    auto casted_val = absl::any_cast<int>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(bool)) {
    auto casted_val = absl::any_cast<bool>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(uint64)) {
    auto casted_val = absl::any_cast<uint64>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(int64_t)) {
    auto casted_val = absl::any_cast<int64_t>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(tensorflow::DataType)) {
    auto casted_val = absl::any_cast<tensorflow::DataType>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(std::string)) {
    auto casted_val = absl::any_cast<std::string>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(std::vector<int64_t>)) {
    auto casted_vals = absl::any_cast<std::vector<int64_t>>(attr);
    // Always create the field.
    auto& values = attributes_[field_name];
    values = Json::arrayValue;
    for (auto val : casted_vals) {
      values.append(GetAsJsonValue(val));
    }
  } else if (tinfo == typeid(std::vector<unsigned int>)) {
    auto casted_vals = absl::any_cast<std::vector<unsigned int>>(attr);
    // Always create the field.
    auto& values = attributes_[field_name];
    values = Json::arrayValue;
    for (auto val : casted_vals) {
      values.append(GetAsJsonValue(val));
    }
  } else if (tinfo == typeid(absl::flat_hash_set<int64_t>)) {
    auto casted_vals = absl::any_cast<absl::flat_hash_set<int64_t>>(attr);
    // Always create the field.
    auto& values = attributes_[field_name];
    values = Json::arrayValue;
    for (auto val : casted_vals) {
      values.append(GetAsJsonValue(val));
    }

  } else if (tinfo == typeid(absl::flat_hash_map<int64_t, int64_t>)) {
    auto casted_vals =
        absl::any_cast<absl::flat_hash_map<int64_t, int64_t>>(attr);

    auto& keys = attributes_[field_name]["keys"];
    auto& values = attributes_[field_name]["values"];
    keys = Json::arrayValue;
    values = Json::arrayValue;
    for (auto pair : casted_vals) {
      keys.append(GetAsJsonValue(pair.first));
      values.append(GetAsJsonValue(pair.second));
    }
  } else if (tinfo == typeid(Window)) {
    auto casted_val = absl::any_cast<Window>(attr);
    std::string window_proto_str;
    if (!tensorflow::ProtoToHumanReadableJson(casted_val, &window_proto_str,
                                              true)
             .ok()) {
      LOG(FATAL) << "Could not parse the window.";
    }
    attributes_[field_name] = GetAsJsonValue(window_proto_str);
  } else if (tinfo == typeid(std::string)) {
    auto casted_val = absl::any_cast<std::string>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);
  } else if (tinfo == typeid(Shape)) {
    auto casted_shape = absl::any_cast<Shape>(attr);
    std::string shape_proto_str;
    if (!tensorflow::ProtoToHumanReadableJson(casted_shape.ToProto(),
                                              &shape_proto_str, true)
             .ok()) {
      LOG(FATAL) << "Could not parse the xla shape.";
    }
    attributes_[field_name] = GetAsJsonValue(shape_proto_str);
  } else {
    LOG(FATAL) << "Unsupported attribute value type " << tinfo.name();
  }
}

bool AttributeMap::HasAttribute(const std::string& field_name) const {
  return attributes_.isMember(field_name);
}

Status AttributeMap::CheckHasAttribute(const std::string& field_name) const {
  if (!HasAttribute(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  } else {
    return Status::OK();
  }
}

StatusOr<std::string> AttributeMap::GetAttributeAsString(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asString();
}

StatusOr<float> AttributeMap::GetAttributeAsFloat(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asFloat();
}

StatusOr<int> AttributeMap::GetAttributeAsInt(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asInt();
}

StatusOr<uint64> AttributeMap::GetAttributeAsUInt64(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asUInt64();
}

StatusOr<int64_t> AttributeMap::GetAttributeAsInt64(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asInt64();
}

StatusOr<bool> AttributeMap::GetAttributeAsBool(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  return attributes_[field_name].asBool();
}

StatusOr<tensorflow::DataType> AttributeMap::GetAttributeAsTFDataType(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  const std::string dtype_string = attributes_[field_name].asString();
  tensorflow::DataType data_type;
  if (!DataType_Parse(dtype_string, &data_type)) {
    return xla::FailedPrecondition("Could not parse the DataType %s.",
                                   dtype_string.c_str());
  }
  return data_type;
}

StatusOr<std::vector<int64_t>> AttributeMap::GetAttributeInt64Vector(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  if (!attributes_[field_name].isArray()) {
    return xla::FailedPrecondition("Custom op field %s is not an array.",
                                   field_name.c_str());
  }
  std::vector<int64_t> result;
  for (auto val : attributes_[field_name]) {
    result.push_back(val.asInt64());
  }
  return result;
}

StatusOr<absl::flat_hash_set<int64_t>> AttributeMap::GetAttributeFlatHashSet(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  if (!attributes_[field_name].isArray()) {
    return xla::FailedPrecondition("Custom op field %s is not an array.",
                                   field_name.c_str());
  }
  absl::flat_hash_set<int64_t> result;
  for (auto val : attributes_[field_name]) {
    result.insert(val.asInt64());
  }
  return result;
}

StatusOr<absl::flat_hash_map<int64_t, int64_t>>
AttributeMap::GetAttributeFlatHashMap(const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  if (!attributes_[field_name].isMember("keys") ||
      !attributes_[field_name].isMember("values")) {
    return xla::FailedPrecondition("Custom op field %s is not a map.",
                                   field_name.c_str());
  }
  auto keys = attributes_[field_name]["keys"];
  auto values = attributes_[field_name]["values"];
  if (keys.size() != values.size()) {
    return xla::FailedPrecondition("Corrupted hash map %s for the custom op.",
                                   field_name.c_str());
  }
  absl::flat_hash_map<int64_t, int64_t> result;
  // i must be an 'int' otherwise the call to the operator [] is ambiguous
  // between Json::Value and int
  for (int i = 0; i < static_cast<int>(keys.size()); i++) {
    int64_t key = keys[i].asInt64();
    int64_t value = values[i].asInt64();
    result[key] = value;
  }
  return result;
}

StatusOr<Window> AttributeMap::GetAttributeAsWindow(
    const std::string& field_name) const {
  TF_RETURN_IF_ERROR(CheckHasAttribute(field_name));
  std::string window_proto_str = attributes_[field_name].asString();
  Window window;
  TF_RETURN_IF_ERROR(
      tensorflow::HumanReadableJsonToProto(window_proto_str, &window));
  return window;
}

StatusOr<Shape> AttributeMap::GetAttributeAsShape(
    const std::string& field_name) const {
  if (!HasAttribute(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  std::string shape_proto_str = attributes_[field_name].asString();
  ShapeProto shape;
  TF_RETURN_IF_ERROR(
      tensorflow::HumanReadableJsonToProto(shape_proto_str, &shape));
  return Shape(shape);
}

const std::string AttributeMap::Serialise() {
  Json::StreamWriterBuilder builder;
  std::stringstream ss;
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  writer->write(attributes_, &ss);
  return ss.str();
}

}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla
