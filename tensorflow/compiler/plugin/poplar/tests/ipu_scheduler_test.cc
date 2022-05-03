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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/post_order_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/shortest_path_scheduler.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_util.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

namespace xla {
namespace poplarplugin {
namespace {

MATCHER_P(IsScheduledWith, expected_schedule, "Module to have given schedule") {
  auto& schedule = arg->schedule();
  for (auto* comp : arg->computations()) {
    if (schedule.is_computation_scheduled(comp)) {
      auto sequence = schedule.sequence(comp);
      auto expected_sequence = expected_schedule.sequence(comp);
      if (sequence.instructions() != expected_sequence.instructions()) {
        *result_listener << "Schedule of computation '" << comp->name()
                         << "' doesn't match expected.";
        return false;
      }
    }
  }
  return true;
}

int64 SizeFunction(const BufferValue& buffer) {
  if (buffer.shape().IsOpaque()) {
    return 0;
  }

  return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
}

MemoryEstimator ScheduleUsesLeastMemory(const HloSchedule& schedule) {
  return [=](const HloComputation& computation,
             const HloInstructionSequence& sequence,
             const HloAliasAnalysis& alias_analysis,
             const LogicalBuffer::SizeFunction& size_function,
             const absl::flat_hash_map<const HloComputation*, int64>*
                 computations) {
    if (schedule.is_computation_scheduled(&computation)) {
      const auto& expected = schedule.sequence(&computation);
      if (sequence.instructions() == expected.instructions()) {
        return 10;
      }
    }

    return 100;
  };
}

using IpuSchedulerTest = HloTestFixture;
TEST_F(IpuSchedulerTest, BestScheduleUsesLeastMemory) {
  const std::string schedule_test_hlo =
      HloPoplarTestUtil::GetLambLikeHloString(5, 1);
  ASSERT_TRUE(SetUpHloModule(schedule_test_hlo));
  auto good_scheduler =
      NamedIpuSchedulerAlgorithm("GoodScheduler", CreatePostOrderScheduler());

  CustomOpReplacer op_replacer;
  TF_ASSERT_OK_AND_ASSIGN(auto result, op_replacer.Run(hlo_module_));

  IpuScheduler expected_scheduler(SizeFunction, good_scheduler.function);
  TF_ASSERT_OK_AND_ASSIGN(result, expected_scheduler.Run(hlo_module_));
  ASSERT_TRUE(result);

  HloSchedule good_schedule = hlo_module_->schedule();
  hlo_module_->clear_schedule();

  const std::vector<NamedIpuSchedulerAlgorithm> schedulers = {
      {"BadScheduler2", CreateShortestPathScheduler({})},
      good_scheduler,
      {"BadScheduler2", CreateClusteringMemoryScheduler({})}};

  auto mock_memory_estimator = ScheduleUsesLeastMemory(good_schedule);
  TF_ASSERT_OK_AND_ASSIGN(
      auto best_algorithm,
      BestIpuSchedule(SizeFunction, schedulers, mock_memory_estimator));

  IpuScheduler scheduler(SizeFunction, best_algorithm);
  TF_ASSERT_OK_AND_ASSIGN(result, scheduler.Run(hlo_module_));
  ASSERT_TRUE(result);

  ASSERT_THAT(hlo_module_, IsScheduledWith(good_schedule));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
