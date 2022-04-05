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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/driver_types.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
class HloInstruction;
class HloCustomCallInstruction;
struct TensorTarget;
namespace poplarplugin {

class PoplarOpDef {
 public:
  PoplarOpDef() = default;
  // By default the op is not allocating.
  virtual StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) {
    return xla::FailedPrecondition(
        "Non-allocating op should not be allocating - %s.", name);
  }

  virtual StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) = 0;
};

// The following singleton class is used to register and access custom poplar
// ops.
template <typename OpType>
class OpManager {
 public:
  // Registration method
  static void RegsiterOp(OpType op,
                         std::unique_ptr<PoplarOpDef> poplar_op_def) {
    auto& ops = GetInstance().ops;

    if (ops.contains(op)) {
      LOG(FATAL) << "Trying to register the same op twice.";
    }
    ops[op] = std::move(poplar_op_def);
  }

  static bool HasOp(OpType op) {
    auto& ops = GetInstance().ops;
    return ops.contains(op);
  }

 protected:
  OpManager<OpType>() = default;

  static absl::optional<PoplarOpDef*> GetOpImpl(OpType op) {
    auto& ops = GetInstance().ops;
    auto itr = ops.find(op);
    if (itr != ops.end()) {
      return itr->second.get();
    }
    return absl::nullopt;
  }

 private:
  static OpManager<OpType>& GetInstance() {
    static OpManager<OpType> instance;
    return instance;
  }

  absl::flat_hash_map<OpType, std::unique_ptr<PoplarOpDef>> ops;
};

// Manager for custom Poplar ops.
class PoplarOpManager : public OpManager<PoplarOp> {
 public:
  static StatusOr<PoplarOpDef*> GetOp(const HloInstruction* inst) {
    // Find the Poplar info given a CustomCall instruction.
    auto ret = GetPoplarCustomOp(inst);
    if (!ret) {
      return FailedPrecondition("Could not find Poplar op %s.",
                                inst->ToString().c_str());
    }
    auto def = GetOpImpl(*ret);
    if (def) {
      return *def;
    }
    return FailedPrecondition("Could not find definition for %s.",
                              PoplarOp_Name(*ret).c_str());
  }
};

class PoplarOpRegistrar {
 public:
  PoplarOpRegistrar(PoplarOp op, std::unique_ptr<PoplarOpDef> poplar_op_def) {
    PoplarOpManager::RegsiterOp(op, std::move(poplar_op_def));
  }

  PoplarOpRegistrar() = delete;
};

#define REGISTER_POPLAR_OP(poplar_op, poplar_op_def)                         \
  namespace {                                                                \
  static PoplarOpRegistrar registrar__poplar_op__##poplar_op##__object(      \
      PoplarOp::poplar_op, std::unique_ptr<PoplarOpDef>(new poplar_op_def)); \
  }

// Manager for Hlo ops.
class HloOpManager : public OpManager<HloOpcode> {
 public:
  static StatusOr<PoplarOpDef*> GetOp(const HloInstruction* inst) {
    auto def = GetOpImpl(inst->opcode());
    if (def) {
      return *def;
    }
    return FailedPrecondition("Could not find definition for %s.",
                              HloOpcodeString(inst->opcode()).c_str());
  }
};

class HloOpRegistrar {
 public:
  HloOpRegistrar(HloOpcode hlo_opcode,
                 std::unique_ptr<PoplarOpDef> poplar_op_def) {
    HloOpManager::RegsiterOp(hlo_opcode, std::move(poplar_op_def));
  }

  HloOpRegistrar() = delete;
};

#define REGISTER_HLO_OP(hlo_opcode, poplar_op_def)                             \
  namespace {                                                                  \
  static HloOpRegistrar registrar__hlo_opcode__##hlo_opcode##__object(         \
      HloOpcode::hlo_opcode, std::unique_ptr<PoplarOpDef>(new poplar_op_def)); \
  }

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_
