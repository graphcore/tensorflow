/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ATTRIBUTES_UTILS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ATTRIBUTES_UTILS_H_
#include <initializer_list>
#include <string>
#include <utility>
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace xla {
namespace poplarplugin {
namespace util {

struct AttrKey {
  std::string key;
  // If we didn't have to mark these constructors as explicit could
  // change the initaliser list to not have to call AttrMember(...)
  // each time and just do {...}. Think this is one of the cases where
  // we deliberately don't want explicit construction but cpplint will
  // warn
  explicit AttrKey(std::string key) : key(std::move(key)) {}
  explicit AttrKey(int key)
      : key(FrontendAttributeId_Name(static_cast<FrontendAttributeId>(key))) {}
};

struct AttrValue {
  std::string value;

  explicit AttrValue(std::string value) : value(std::move(value)) {}
  template <typename T>
  explicit AttrValue(T value) : AttrValue(std::to_string(value)) {}
};

struct AttrMember {
  AttrKey first;
  AttrValue second;

  template <typename K, typename V>
  AttrMember(K k, V v) : first(std::move(k)), second(std::move(v)) {}
};

/*
Helper function to try make setting large numbers of front end attributes
a little less repetitive. Takes an initializer_list and iterate through
applying each one. The point in the AttrKey and AttrValue structs is
to avoid having to repeat calls to common string conversion functions
when making the initializer_list by having them constructable from
several types
 */

using AttrMembers = std::initializer_list<AttrMember>;

void SetInstructionFrontEndAttributes(tensorflow::XlaOpKernelContext* ctx,
                                      xla::XlaBuilder* builder,
                                      const xla::XlaOp& outputs,
                                      AttrMembers&& attributes) {
  for (auto& member : attributes) {
    OP_REQUIRES_OK(ctx, builder->SetInstructionFrontendAttribute(
                            outputs, std::move(member.first.key),
                            std::move(member.second.value)));
  }
}

}  // namespace util
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_ATTRIBUTES_UTILS_H_
