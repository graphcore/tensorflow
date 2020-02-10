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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_PREPLANNING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_PREPLANNING_H_

#include <list>
#include <set>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {

/**
 * Memoization of matmul parameters.
 */
class MatMulPreplanning {
 public:
  Status Plan(const HloModule* module, CompilerResources& resources);

 private:
  // Store matmul parameters.
  std::set<poplin::MatMulPlanParams> preplan_matmuls;

  // OptionsFlags storage location.
  std::list<poplar::OptionFlags> option_flags_store;

  Status StorePreplanMatMulsLSTM(const HloInstruction* inst,
                                 CompilerResources& resources);

  Status StorePreplanMatMulsGRU(const HloInstruction* inst,
                                CompilerResources& resources);

  Status StorePreplanMatMuls(const HloInstruction* inst,
                             CompilerResources& resources);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATMUL_PREPLANNING_H_
