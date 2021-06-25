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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_BASE_H_

#include <memory>

#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
class HloModule;

namespace poplarplugin {

class HloPoplarTestBase : public HloTestBase {
 public:
  // Returns the number of IPUs a test is allowed to use.
  static StatusOr<int32> GetMaxIpuCount();

 protected:
  using HloTestBase::HloTestBase;

  static std::unique_ptr<CompilerResources> GetMockResources(
      poplar::Device& device, HloModule* module, int32 replication_factor = 1);

  static StatusOr<poplar::Device> CreateIpuModel(int32 num_ipus = 0,
                                                 int32 num_tiles = 0);
  static StatusOr<poplar::Device> CreateIpuDevice(int32 num_ipus = 1,
                                                  int32 num_tiles = 0);

  StatusOr<poplar::Engine> Compile(CompilerResources& resources,
                                   HloModule* module);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_TEST_BASE_H_
