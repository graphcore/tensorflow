/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_

#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "include/json/json.h"
#include "tensorflow/core/framework/types.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/any.h"

namespace xla {
class HloInstruction;
class Window;
class Shape;

namespace poplarplugin {
// Tries to convert the string target for the kCustomCall
absl::optional<PoplarOp> GetPoplarCustomOp(const HloInstruction* inst);

namespace IPUCustomKernelsUtil {

class AttributeMap {
 public:
  AttributeMap();
  AttributeMap(const HloInstruction* custom_call);
  AttributeMap(const std::string& attributes_json);

  // We support:
  // * float, int, bool, uint64, int64, tensorflow::DataType
  // * absl::flat_hash_set of int64
  // * absl::flat_hash_map of int64 to int64
  void AddAttribute(const std::string& field_name, const absl::any& attr);

  bool HasAttribute(const std::string& field_name) const;
  Status CheckHasAttribute(const std::string& field_name) const;
  StatusOr<std::string> GetAttributeAsString(
      const std::string& field_name) const;
  StatusOr<float> GetAttributeAsFloat(const std::string& field_name) const;
  StatusOr<int> GetAttributeAsInt(const std::string& field_name) const;
  StatusOr<uint64> GetAttributeAsUInt64(const std::string& field_name) const;
  StatusOr<int64> GetAttributeAsInt64(const std::string& field_name) const;
  StatusOr<bool> GetAttributeAsBool(const std::string& field_name) const;
  StatusOr<tensorflow::DataType> GetAttributeAsTFDataType(
      const std::string& field_name) const;

  // These are included as absl::flat_hash_set<T> is unordered,
  // whereas this allows a list to be passed whilst preserving ordering.
  StatusOr<std::vector<int64>> GetAttributeInt64Vector(
      const std::string& field_name) const;

  StatusOr<absl::flat_hash_set<int64>> GetAttributeFlatHashSet(
      const std::string& field_name) const;
  StatusOr<absl::flat_hash_map<int64, int64>> GetAttributeFlatHashMap(
      const std::string& field_name) const;
  StatusOr<Window> GetAttributeAsWindow(const std::string& field_name) const;
  StatusOr<Shape> GetAttributeAsShape(const std::string& field_name) const;

  const std::string Serialise();

 private:
  Json::Value attributes_;
};
}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_CUSOM_KERNELS_UTIL_H_
