/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_PVA_TEST_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_PVA_TEST_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"

#include "tensorflow/core/platform/path.h"

namespace pva {
class Report;
}  // namespace pva

namespace xla {
class HloModule;

namespace poplarplugin {
class PoplarExecutor;
class PoplarPlatform;

struct ScopedSyntheticData {
  ScopedSyntheticData(bool enable);  // NOLINT
  ~ScopedSyntheticData();
  bool original_state_;
};

struct ScopedPoplarAutoReport {
  ScopedPoplarAutoReport(const std::string& profile_root);  // NOLINT
  ~ScopedPoplarAutoReport();
  std::string env_name_ = "POPLAR_ENGINE_OPTIONS";
  std::string original_value_;
};

// Test fixture for running a HloModule with pva profiling enabled.
struct PVAHloTest : HloTestBase {
  struct MockArguments {
    explicit MockArguments(const HloModule* module);
    std::vector<xla::Literal> argument_storage;
    std::vector<xla::Literal*> arguments;
  };

  PVAHloTest();
  void SetUp() override;
  void TearDown() override;

  // Execute the given module using the provided ipu config, using mock values
  // for any arguments. Returns the path to the pva profile.
  StatusOr<std::string> ExecuteWithConfig(std::unique_ptr<HloModule> module,
                                          const IpuOptions& config);

  PoplarPlatform* platform_ = nullptr;
  PoplarExecutor* executor_ = nullptr;

  const std::string report_root_;

  ScopedSyntheticData scoped_synthetic_data_ = {true};
  ScopedPoplarAutoReport scoped_auto_report_ = {report_root_};
};

IpuOptions CreateConfig();

uint64_t total_tile_memory(const pva::Report& report);
uint64_t max_tile_memory(const pva::Report& report);
uint64_t always_live_memory(const pva::Report& report);
uint64_t peak_memory(const pva::Report& report);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HLO_POPLAR_PVA_TEST_H_
