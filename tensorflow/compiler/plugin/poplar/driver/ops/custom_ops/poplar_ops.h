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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "absl/container/flat_hash_map.h"

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
  virtual StatusOr<poplar::Tensor> Allocator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const std::string& name,
                                             const TensorTarget& tensor_target,
                                             const TensorMap& tensor_map) {
    return xla::FailedPrecondition(
        "Non-allocating op should not be allocating. {}", name);
  }

  virtual StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map) = 0;
};

// The following singleton class is used to register and access custom poplibs
// ops.
class PoplarOpManager {
 public:
  // Registration method
  static void RegsiterOp(PoplarOp op,
                         std::unique_ptr<PoplarOpDef> poplibs_op_def);
  static StatusOr<PoplarOpDef*> GetOp(const HloCustomCallInstruction* inst);

 private:
  PoplarOpManager() = default;
  static PoplarOpManager& GetInstance();

  absl::flat_hash_map<PoplarOp, std::unique_ptr<PoplarOpDef>> ops;
};

class PoplarOpRegistrar {
 public:
  PoplarOpRegistrar(PoplarOp op, std::unique_ptr<PoplarOpDef> poplibs_op_def);

  PoplarOpRegistrar() = delete;
};

#define REGISTER_POPLAR_OP(poplibs_op, poplibs_op_def)                         \
  namespace {                                                                  \
  static PoplarOpRegistrar registrar__poplibs_op__##poplibs_op##__object(      \
      PoplarOp::poplibs_op, std::unique_ptr<PoplarOpDef>(new poplibs_op_def)); \
  }

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_CUSTOM_OPS_POPLIBS_OPS_H_
