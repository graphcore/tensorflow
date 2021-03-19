/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_RNN_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_RNN_UTIL_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <poplin/FullyConnected.hpp>
#include <popnn/Gru.hpp>
#include <popnn/Lstm.hpp>

namespace xla {
namespace poplarplugin {

StatusOr<popnn::lstm::LstmParams> GetLstmParameters(const HloInstruction* inst);

StatusOr<poplar::OptionFlags> GetLstmOpts(const HloInstruction* inst,
                                          const CompilerResources& res);

StatusOr<popnn::gru::GruParams> GetGruParameters(const HloInstruction* inst);

StatusOr<poplar::OptionFlags> GetGruOpts(const HloInstruction* inst,
                                         const CompilerResources& res);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_RNN_UTIL_H_
