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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_INSTRUCTION_EXTENSIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_INSTRUCTION_EXTENSIONS_H_

#include <functional>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/extension_registry.h"

namespace xla {
namespace poplarplugin {

template <typename T>
struct PoplarExtension {};

// Only providing a const member function specialisation since
// the HloPoplarInstruction functions we care about are all const
template <typename ReturnT, typename... TParams>
struct PoplarExtension<ReturnT (HloPoplarInstruction::*)(TParams...) const> {
  using FuncionDecl = std::function<ReturnT(const HloInstruction*, TParams...)>;
  FuncionDecl impl = [](const HloInstruction*, TParams...) {
    return ReturnT();
  };
};

template <typename PoplarFn, PoplarFn fn>
struct MakeExtension : PoplarExtension<PoplarFn> {
  static constexpr auto poplar_handle = fn;
};

#define MAKE_EXTENSION(POPLAR_HANDLE) \
  MakeExtension<decltype(&POPLAR_HANDLE), &POPLAR_HANDLE>

using AllocatingIndicesExtension =
    MAKE_EXTENSION(HloPoplarInstruction::AllocatingIndices);
using AllocatingOutputExtension =
    MAKE_EXTENSION(HloPoplarInstruction::AllocatingOutput);
using LayoutDependenciesExtension =
    MAKE_EXTENSION(HloPoplarInstruction::LayoutDependencies);
using FindConsumersExtension =
    MAKE_EXTENSION(HloPoplarInstruction::FindConsumers);

#undef MAKE_EXTENSION

using HloInstructionExtensions =
    InstructionExtensions<AllocatingIndicesExtension, AllocatingOutputExtension,
                          LayoutDependenciesExtension, FindConsumersExtension>;

HloInstructionExtensions& GetHloInstructionExtensions();

template <typename Ext, typename Callable>
void RegisterHloInstructionExtension(HloOpcode op_code, Callable&& callable) {
  GetHloInstructionExtensions().Register<Ext>(op_code, callable);
}

template <typename Ext, typename... TParams>
typename std::result_of<
    decltype (&HloInstructionExtensions::Call<Ext, TParams&&...>)(  // NOLINT
        HloInstructionExtensions, const HloInstruction*, TParams&&...)>::type
CallHloInstructionExtension(const HloInstruction* instruction,
                            TParams&&... params) {
  return GetHloInstructionExtensions().Call<Ext>(
      instruction, std::forward<TParams>(params)...);
}

// Utility function/macro to help statically register extensions
inline bool RegistrationWrapper(
    HloOpcode op_code,
    const std::function<void(HloOpcode)>& registration_func) {
  registration_func(op_code);
  return true;
}
#define REGISTER_HLO_INST_EXTENSIONS(hlo_op_code, ext_cb) \
  static bool registered_##hlo_op_code##_extensions =     \
      RegistrationWrapper(HloOpcode::hlo_op_code, ext_cb);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_INSTRUCTION_EXTENSIONS_H_
