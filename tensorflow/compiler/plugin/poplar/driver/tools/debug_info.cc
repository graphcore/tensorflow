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

#include <utility>

#include <poplar/DebugContext.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/DebugInfo.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

PoplarOpDefDebugInfo::PoplarOpDefDebugInfo(
    const poplar::DebugContext& debug_context, std::string op_name)
    : poplar::DebugInfo(debug_context, "poplar_driver") {
  setValue("category", poplar::ProfileValue{"op"});
  setValue("class", op_name);
}

PoplarOpDefDebugInfo::~PoplarOpDefDebugInfo() {}

HloInstructionDebugInfo::HloInstructionDebugInfo(
    poplar::DebugContext& debug_context, const HloInstruction* instruction)
    : poplar::DebugInfo(debug_context, "hloinstruction") {
  setValue("category", poplar::ProfileValue{"op"});
  setValue("hlo_name", poplar::ProfileValue{instruction->name()});
  setValue("hlo_id", poplar::ProfileValue{instruction->unique_id()});
  setValue("opcode",
           poplar::ProfileValue{HloOpcodeString(instruction->opcode())});
  setValue("signature", poplar::ProfileValue{instruction->SignatureString()});
  setValue("debug_string", poplar::ProfileValue{instruction->ToString()});

  setValue("operand_count", poplar::ProfileValue{instruction->operand_count()});

  // Record the parent computation information.
  if (instruction->parent()) {
    setValue("computation", instruction->parent()->name());
    setValue("computation_id", instruction->parent()->unique_id());
  }

  poplar::ProfileValue::Map operands;
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->operand(i);
    operands.insert({operand->name(), ""});
  }
  setValue("operands", operands);

  poplar::ProfileValue::Map users;
  for (auto* user : instruction->users()) {
    users.insert({instruction->name(), ""});
  }
  setValue("users", users);

  // As the backend config is part of the debug_string we will not
  // explicitly include it for now.
}

HloInstructionDebugInfo::~HloInstructionDebugInfo() {}

XlaOpDebugInfo::XlaOpDebugInfo(poplar::DebugContext& debug_context,
                               const xla::OpMetadata& metadata)
    : poplar::DebugInfo(debug_context, "xla_op") {
  setValue("category", poplar::ProfileValue{"op"});
  setValue("op_name", poplar::ProfileValue{metadata.op_name()});
  setValue("op_type", poplar::ProfileValue{metadata.op_type()});

  if (!metadata.source_file().empty()) {
    setValue("sourcefile", poplar::ProfileValue{metadata.source_file()});
    setValue("sourceline", poplar::ProfileValue{metadata.source_line()});
  }
}

XlaOpDebugInfo::~XlaOpDebugInfo() {}

uint64_t GetDebugId(const CompilerResources& resources,
                    const HloInstruction* inst) {
  auto it = resources.hlo_instruction_to_debug_id_mapping.find(inst);
  if (it != resources.hlo_instruction_to_debug_id_mapping.end()) {
    return it->second;
  } else {
    return 0;
  }
}

poplar::DebugNameAndId GetDebugNameAndId(const CompilerResources& res,
                                         const HloInstruction* inst,
                                         const std::string suffix) {
  return poplar::DebugNameAndId(suffix, GetDebugId(res, inst),
                                GetDebugName(inst));
}

}  // namespace poplarplugin
}  // namespace xla
