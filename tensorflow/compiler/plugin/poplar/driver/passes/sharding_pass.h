/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SHARDING_PASS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_SHARDING_PASS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * This adds sharding information to operations which do not have any, but are
 * parents of operations which do.
 *
 * The sharding represents the IPU where the output tensor of an operation is
 * located. This will be in the same place as the operation vertices are
 * located.
 *
 * Some rules:
 *   - while/repeat bodies must have equivalent input and output sharding
 *   - while/repeat body and predicates must have the same input sharding
 *   - conditional computations must have the same output sharding
 *   - all callsites must match their called computation output sharding
 *   - all GTE instructions must match the sharding of the selected part of
 *     their operand
 *   - 'plumbing' ops (call, while, conditional, tuple, GTE) can have a 'single'
 *     or 'tuple' type sharding info, indicating where the tensors are that they
 *     produce.
 *   - operations which output a tuple of tensors (batchnorm training) will have
 *     a tuple of sharding values, although they will all be the same.
 */
class ShardingPass : public HloModulePass {
 public:
  absl::string_view name() const override { return "sharding-pass"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
