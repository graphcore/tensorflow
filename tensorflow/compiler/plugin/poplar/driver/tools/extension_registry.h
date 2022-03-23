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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXTENSION_REGISTRY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXTENSION_REGISTRY_H_

#include <map>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

// Wrapper that provides the common interface for
// calling and assigning to an Extension. An Extension
// can be any type that has a std::function impl member.
template <typename Extension>
struct ExtensionWrapper : private Extension {
  using FunctionDecl = decltype(Extension::impl);
  using Result = typename FunctionDecl::result_type;

  template <typename... Args>
  Result operator()(Args&&... args) {
    return this->impl(std::forward<Args>(args)...);
  }

  template <typename Func>
  ExtensionWrapper& operator=(Func&& func) {
    this->impl = std::forward<Func>(func);
    return *this;
  }
};

template <typename Extension>
using ExtensionRegistry = std::map<HloOpcode, ExtensionWrapper<Extension>>;

// Common structure/storage for registering/calling an Extension.
template <typename Extension>
struct InstructionExtensionHelper {
  using Registry = ExtensionRegistry<Extension>;
  using Result = typename Registry::mapped_type::Result;

  template <typename Func>
  void Register(HloOpcode op_code, Func&& ext) {
    registry_[op_code] = std::forward<Func>(ext);
  }

  template <typename... TParams>
  Result Call(const HloInstruction* instruction, TParams&&... params) {
    return registry_[instruction->opcode()](instruction,
                                            std::forward<TParams>(params)...);
  }

 private:
  Registry registry_;
};

// Container for using multiple Extensions. Calls made to HloPoplarInstructions
// are forwarded to the member function pointed to by Ext::poplarHandle.
template <typename... Exts>
struct InstructionExtensions : private InstructionExtensionHelper<Exts>... {
  template <typename Ext, typename Func>
  void Register(HloOpcode op_code, Func&& ext) {
    InstructionExtensionHelper<Ext>::Register(op_code, std::forward<Func>(ext));
  }

  template <typename Ext, typename... TParams>
  typename InstructionExtensionHelper<Ext>::Result Call(
      const HloInstruction* instruction, TParams&&... params) {
    CHECK(instruction != nullptr);

    if (auto* poplar_instruction = DynCast<HloPoplarInstruction>(instruction)) {
      return (poplar_instruction->*Ext::poplar_handle)(
          std::forward<TParams>(params)...);
    }

    return InstructionExtensionHelper<Ext>::Call(
        instruction, std::forward<TParams>(params)...);
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_EXTENSION_REGISTRY_H_
