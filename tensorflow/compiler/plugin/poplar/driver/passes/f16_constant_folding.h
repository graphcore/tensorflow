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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_F16_CONSTANT_FOLDING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_F16_CONSTANT_FOLDING_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/*
 * Allow folding of fp16 in fp32 to try and prevent inf/nans.
 * The aim is to deal with the following situation, example: you want to
 * calculate 256*256/32 but instructions are fp16. If you multiply 256*256 you
 * get inf. for fp16. But if we convert the instructions to fp32 you would get
 * 256*256/32=65536/32 = 2048 and you can convert 2048 back to fp16.
 */

class F16ConstantFolding : public HloModulePass {
 public:
  absl::string_view name() const override { return "f16-constant-folding"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
