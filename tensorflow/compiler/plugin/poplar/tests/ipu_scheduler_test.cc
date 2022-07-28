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
#include <gtest/gtest.h>

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/post_order_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_pva_test.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/plugin/poplar/tests/hlo_samples/all_samples.h"

#include "pva/pva.hpp"

namespace xla {
namespace poplarplugin {
namespace {

// PVA based test fixture for measuring how much memory is used when executing a
// particular module with a particular scheduling algorithm.
using SchedulerTestCase = std::tuple<HloTestCase, IpuSchedulingAlgorithm>;
struct IpuSchedulerPVAHloTest
    : PVAHloTest,
      ::testing::WithParamInterface<SchedulerTestCase> {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(auto ipu_count,
                            HloPoplarTestBase::GetMaxIpuCount());
    if (ipu_count == 0) {
      GTEST_SKIP() << "No IPUS available, skipping.";
    }

    PVAHloTest::SetUp();

    test_hlo_ = std::get<0>(GetParam()).hlo;
    ipu_count_ = std::get<0>(GetParam()).ipu_count;
    scheduler_ = std::get<1>(GetParam());
  }

  StatusOr<std::string> ExecuteWithScheduler(std::unique_ptr<HloModule> module,
                                             IpuSchedulingAlgorithm scheduler) {
    const auto scheduler_name = IpuSchedulingAlgorithm_Name(scheduler);
    module->set_name(module->name() + scheduler_name);

    auto config = CreateConfig();
    auto* speed_config = config.mutable_speed_size_config();
    speed_config->set_scheduler_selection(scheduler);

    auto* device_config = config.add_device_config();
    device_config->set_auto_count(ipu_count_);

    return ExecuteWithConfig(std::move(module), config);
  }

  std::string test_hlo_;
  int64_t ipu_count_;
  IpuSchedulingAlgorithm scheduler_;
};

TEST_P(IpuSchedulerPVAHloTest, ChoosesBestIntegration) {
  // Integration test for the CHOOSE_BEST algorithm. Which should choose a
  // schedule with memory characteristics that are not worse than the other
  // algorithm.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(test_hlo_));
  TF_ASSERT_OK_AND_ASSIGN(
      auto profile_path,
      ExecuteWithScheduler(std::move(module),
                           IpuSchedulingAlgorithm::CHOOSE_BEST));
  auto profile = pva::openReport(profile_path);
  const auto chosen_peak = peak_memory(profile);

  TF_ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(test_hlo_));
  TF_ASSERT_OK_AND_ASSIGN(profile_path,
                          ExecuteWithScheduler(std::move(module), scheduler_));
  profile = pva::openReport(profile_path);
  const auto other_peak = peak_memory(profile);

  // The CHOOSE_BEST algorithm can only estimate the memory usage of a
  // schedule, so there's no guarantee we'll get the lowest peak memory
  // from it. Additionally we don't really care if two schedules which
  // use 100mb of memory differ by 100kb, hence the small tolerance we use
  // for this check.
  ASSERT_LE(chosen_peak, other_peak * 1.005)
      << IpuSchedulingAlgorithm_Name(scheduler_)
      << " has better peak memory than 'CHOOSE_BEST'";
}

std::string SchedulerTestCaseName(
    const ::testing::TestParamInfo<SchedulerTestCase>& info) {
  auto name = std::get<0>(info.param).name;
  auto scheduler = std::get<1>(info.param);
  return name + "_" + IpuSchedulingAlgorithm_Name(scheduler);
}

INSTANTIATE_TEST_SUITE_P(
    IpuSchedulerHLO, IpuSchedulerPVAHloTest,
    ::testing::Combine(
        ::testing::Values(MAKE_HLO_TEST_CASE(T44634_hlo),
                          MAKE_HLO_TEST_CASE(linear_regression_hlo, 2),
                          MAKE_HLO_TEST_CASE(pipeline_grouped_recomputation_hlo,
                                             4),
                          MAKE_HLO_TEST_CASE(serialized_matmul_hlo),
                          MAKE_HLO_TEST_CASE(rnn_hlo)),
        ::testing::Values(IpuSchedulingAlgorithm::CLUSTERING,
                          IpuSchedulingAlgorithm::POST_ORDER,
                          IpuSchedulingAlgorithm::SHORTEST_PATH)),
    SchedulerTestCaseName);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
