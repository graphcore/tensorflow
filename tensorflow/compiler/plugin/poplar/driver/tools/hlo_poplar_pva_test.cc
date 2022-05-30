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

#include <experimental/filesystem>
#include <sstream>

#include "tensorflow/core/platform/path.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_pva_test.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "poplar/OptionFlags.hpp"

#include "pva/pva.hpp"

namespace xla {
namespace poplarplugin {
namespace {
std::string GetTestName() {
  const auto* test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  return test_info->name();
}

bool IsInTempDir(const std::string& path) {
  const auto base_name = tensorflow::io::Basename(path);

  const auto dir_name =
      tensorflow::io::CleanPath(tensorflow::io::Dirname(path));
  const auto tmp_dir = tensorflow::io::CleanPath(::testing::TempDir());
  return tensorflow::str_util::StartsWith(dir_name, tmp_dir) &&
         !base_name.empty();
}

// Outfeeds with the same name need to have the same shape/replication factor,
// since there's no guarantee of that for our tests we append the module
// name onto the outfeed id.
Status FixOutfeedNames(HloModule* module) {
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kOutfeed) {
        PoplarFeedConfig config;
        config.ParseFromString(inst->outfeed_config());
        const auto fixed_name = config.feed_id() + module->name();
        config.set_feed_id(fixed_name);

        std::string config_str;
        if (!config.SerializeToString(&config_str)) {
          return InternalError("Could not serialize feed config");
        }
        inst->set_outfeed_config(config_str);
      }
    }
  }
  return Status::OK();
}
}  // namespace

ScopedSyntheticData::ScopedSyntheticData(bool enable)
    : original_state_(PoplarXlaFlags::Get().use_synthetic_data) {
  auto& writable_flags = const_cast<PoplarXlaFlags&>(PoplarXlaFlags::Get());
  writable_flags.use_synthetic_data = enable;
}
ScopedSyntheticData::~ScopedSyntheticData() {
  auto& writable_flags = const_cast<PoplarXlaFlags&>(PoplarXlaFlags::Get());
  writable_flags.use_synthetic_data = original_state_;
}

ScopedPoplarAutoReport::ScopedPoplarAutoReport(
    const std::string& profile_root) {
  auto options = poplar::OptionFlags();
  poplar::readJSONFromEnv(env_name_, options);

  std::stringstream env_out;
  env_out << options;
  original_value_ = env_out.str();

  options.set("autoReport.all", "true");
  options.set("autoReport.directory", profile_root);
  env_out.str("");
  env_out << options;
  setenv(env_name_.c_str(), env_out.str().c_str(), 1);
}
ScopedPoplarAutoReport::~ScopedPoplarAutoReport() {
  setenv(env_name_.c_str(), original_value_.c_str(), 1);
}

PVAHloTest::MockArguments::MockArguments(const HloModule* module) {
  auto* entry = module->entry_computation();
  for (auto* param : entry->parameter_instructions()) {
    const auto& shape = param->shape();
    const auto dimensions = shape.dimensions();

    if (shape.element_type() == S32) {
      int32 value = 1;
      auto argument =
          LiteralUtil::CreateFullWithDescendingLayout(dimensions, value);
      argument_storage.push_back(std::move(argument));
    } else if (shape.element_type() == S64) {
      int64_t value = 1;
      auto argument =
          LiteralUtil::CreateFullWithDescendingLayout(dimensions, value);
      argument_storage.push_back(std::move(argument));
    } else if (shape.element_type() == F16) {
      auto argument =
          LiteralUtil::CreateFullWithDescendingLayout(dimensions, half(1.0f));
      argument_storage.push_back(std::move(argument));
    } else if (shape.element_type() == F32) {
      auto argument =
          LiteralUtil::CreateFullWithDescendingLayout(dimensions, 1.0f);
      argument_storage.push_back(std::move(argument));
    } else {
      CHECK(false) << "Unknown element type '"
                   << PrimitiveType_Name(shape.element_type())
                   << "' for argument generation.";
    }
  }

  for (auto& argument : argument_storage) {
    arguments.push_back(&argument);
  }
}

PVAHloTest::PVAHloTest()
    : report_root_(tensorflow::io::JoinPath(::testing::TempDir(),
                                            "_tf_pva_tests", GetTestName())) {}

void PVAHloTest::SetUp() {
  TF_ASSERT_OK_AND_ASSIGN(auto platform,
                          se::MultiPlatformManager::PlatformWithName("Poplar"));
  platform_ = static_cast<PoplarPlatform*>(platform);

  TF_ASSERT_OK_AND_ASSIGN(auto* executor, platform_->ExecutorForDevice(0));
  executor_ = static_cast<PoplarExecutor*>(executor->implementation());
}

void PVAHloTest::TearDown() {
  CHECK(IsInTempDir(report_root_));
  std::experimental::filesystem::remove_all(report_root_);
}

StatusOr<std::string> PVAHloTest::ExecuteWithConfig(
    std::unique_ptr<HloModule> module, const IpuOptions& config) {
  TF_RETURN_IF_ERROR(platform_->ResetExecutors());
  TF_RETURN_IF_ERROR(platform_->ConfigurePoplarDevices(config));
  TF_RETURN_IF_ERROR(FixOutfeedNames(module.get()));

  const auto profile_dir = executor_->GetModuleReportDirectory(module->name());
  const auto profile_path =
      tensorflow::io::JoinPath(report_root_, profile_dir, "profile.pop");

  auto mock_args = MockArguments(module.get());
  TF_ASSIGN_OR_RETURN(auto result,
                      Execute(std::move(module), mock_args.arguments));

  return profile_path;
}

IpuOptions CreateConfig() {
  IpuOptions config;
  config.set_creator_id(IpuOptionsCreator::IPU_UTILS);
  config.set_device_connection_type(IpuDeviceConnectionType::ON_DEMAND);

  return config;
}

uint64_t total_tile_memory(const pva::Report& report) {
  uint64_t total = absl::c_accumulate(
      report.compilation().tiles(), 0, [](int64_t sum, const pva::Tile& tile) {
        return sum + tile.memory().total().excludingGaps();
      });
  return total;
}
uint64_t max_tile_memory(const pva::Report& report) {
  uint64_t max_memory = 0;
  for (auto tile : report.compilation().tiles()) {
    max_memory = std::max(tile.memory().total().excludingGaps(), max_memory);
  }
  return max_memory;
}
uint64_t always_live_memory(const pva::Report& report) {
  uint64_t always_live = absl::c_accumulate(
      report.compilation().tiles(), 0, [](int64_t sum, const pva::Tile& tile) {
        return sum + tile.memory().alwaysLiveBytes();
      });
  return always_live;
}
uint64_t peak_memory(const pva::Report& report) {
  uint64_t always_live = always_live_memory(report);
  uint64_t max_non_live = 0;
  for (auto& step : report.compilation().livenessProgramSteps()) {
    max_non_live = std::max(max_non_live, step.notAlwaysLiveMemory().bytes());
  }

  return always_live + max_non_live;
}

}  // namespace poplarplugin
}  // namespace xla
