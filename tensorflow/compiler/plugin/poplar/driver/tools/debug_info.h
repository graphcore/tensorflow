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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_DEBUG_INFO_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_DEBUG_INFO_H_

#include <string>

#include <poplar/DebugContext.hpp>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

// This file contains the specialization of the poplar DebugInfo for use in
// tensorflow Each specialization provides a debug information for separate
// layers in tensorflow
//
// poplar_driver
//  - PoplarOpDefDebugInfo - Used by the PoplarOpDef Creator and Allocation
//  functions
// hlo_instruction
//  - HloInstructionDebugInfo - Used to represent an HloInstruction
// xla_op
//  - XlaOpDebugInfo - Represents the higher level xla operation.
//
// They capture debug information associated with the relevant layer.

namespace poplar {
class Tensor;
}

namespace xla {

namespace poplarplugin {

class TensorMap;
struct CompilerResources;

class PoplarOpDefDebugInfo : public poplar::DebugInfo {
 public:
  PoplarOpDefDebugInfo(const poplar::DebugContext& debug_context,
                       std::string op_name);
  PoplarOpDefDebugInfo& operator=(const PoplarOpDefDebugInfo&) = delete;
  PoplarOpDefDebugInfo(const PoplarOpDefDebugInfo&) = delete;
  virtual ~PoplarOpDefDebugInfo();

  using poplar::DebugInfo::setValue;
};

class HloInstructionDebugInfo : public poplar::DebugInfo {
 public:
  HloInstructionDebugInfo(poplar::DebugContext& debug_context,
                          const HloInstruction* instruction);

  HloInstructionDebugInfo& operator=(const HloInstructionDebugInfo&) = delete;
  HloInstructionDebugInfo(const HloInstructionDebugInfo&) = delete;
  virtual ~HloInstructionDebugInfo();

  using poplar::DebugInfo::setValue;
};

class XlaOpDebugInfo : public poplar::DebugInfo {
 public:
  XlaOpDebugInfo(poplar::DebugContext& debug_context,
                 const xla::OpMetadata& metadata);

  XlaOpDebugInfo& operator=(const XlaOpDebugInfo&) = delete;
  XlaOpDebugInfo(const XlaOpDebugInfo&) = delete;
  virtual ~XlaOpDebugInfo();

  using poplar::DebugInfo::setValue;
};

poplar::DebugNameAndId GetDebugNameAndId(const CompilerResources&,
                                         const HloInstruction*,
                                         const std::string = "");

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_DEBUG_INFO_H_
