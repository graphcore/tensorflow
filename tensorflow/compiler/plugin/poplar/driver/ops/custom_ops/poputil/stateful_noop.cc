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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Program.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {

class StatefulNoopOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph&, CompilerResources&,
                                             const HloInstruction*,
                                             const xla::Shape&,
                                             TensorMap&) override {
    poplar::program::Sequence seq;
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poputil, StatefulNoop, StatefulNoopOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
