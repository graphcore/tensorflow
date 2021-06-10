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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor_utils.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "absl/memory/memory.h"

namespace xla {
namespace poplarplugin {
namespace {

namespace util = ::xla::poplarplugin::pipelinevisitorutils;

class PipelineVisitorUtilTest : public HloTestBase {};

/**
 * Check that the grouped schedule is simply the stages repeated in reverse
 * order.
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedOrder4) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::GroupedScheduler().ConstructSchedule(offsets, elements));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "dcbadcbadcbadcba");
}

/**
 * Check that taking the first elements from the grouped schedule is the same as
 * running each stage once. This helps us reduce control code in the grouped
 * case.
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedResize) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::GroupedScheduler().ConstructSchedule(offsets, elements);

  for (auto& step : schedule) {
    step.resize(1);
  }

  auto schedule_flat = util::FlattenSchedule(schedule);

  std::string result(schedule_flat.begin(), schedule_flat.end());
  EXPECT_EQ(result, "dcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedPadLeft) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::GroupedScheduler().ConstructSchedule(offsets, elements);
  schedule = util::LeftPadSchedule(schedule, '.');

  auto schedule_flat = util::FlattenSchedule(schedule);

  std::string result(schedule_flat.begin(), schedule_flat.end());
  EXPECT_EQ(result, "....dcbadcbadcbadcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedPadRight) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::GroupedScheduler().ConstructSchedule(offsets, elements);
  schedule = util::RightPadSchedule(schedule, '.');

  auto schedule_flat = util::FlattenSchedule(schedule);

  std::string result(schedule_flat.begin(), schedule_flat.end());
  EXPECT_EQ(result, "dcbadcbadcbadcba....");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedConcat) {
  std::vector<char> A = {'a', 'b', 'c', 'd'};
  std::vector<char> B = {'w', 'x', 'y', 'z'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto A_sched = util::GroupedScheduler().ConstructSchedule(offsets, A);
  auto B_sched = util::GroupedScheduler().ConstructSchedule(offsets, B);
  auto schedule = util::ConcatSchedule(A_sched, B_sched);
  auto schedule_flat = util::FlattenSchedule(schedule);

  std::string result(schedule_flat.begin(), schedule_flat.end());
  EXPECT_EQ(result, "dcbadcbadcbadcbazyxwzyxwzyxwzyxw");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampUp) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampUpSchedule(offsets, elements, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "a...ba..cba.dcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampUpRecompute) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRecomputationRampUpSchedule(offsets, elements, 2, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "....b....ba...ba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampUpOverlapIOIn) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampUpScheduleOverlapIO(offsets, elements, 0, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "a...ba..cba.dcbadcbadcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest,
       ConstructScheduleGroupedRampUpOverlapIOCompute) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampUpScheduleOverlapIO(offsets, elements, 1, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "....a...ba..cba.dcbadcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampUpOverlapIOOut) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampUpScheduleOverlapIO(offsets, elements, 2, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "........a...ba..cba.dcba");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampDown) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampDownSchedule(offsets, elements, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, ".dcb..dc...d....");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampDownRecompute) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRecomputationRampDownSchedule(offsets, elements, 2, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "a..b.a..........");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampDownOverlapIOIn) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampDownScheduleOverlapIO(offsets, elements, 0, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, ".dcb..dc...d............");
}

/**
 */
TEST_F(PipelineVisitorUtilTest,
       ConstructScheduleGroupedRampDownOverlapIOCompute) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampDownScheduleOverlapIO(offsets, elements, 1, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "dcba.dcb..dc...d........");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleGroupedRampDownOverlapIOOut) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructRampDownScheduleOverlapIO(offsets, elements, 2, '.'));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "dcbadcba.dcb..dc...d....");
}

/**
 */
TEST_F(PipelineVisitorUtilTest, ConstructScheduleOverlapIO) {
  std::vector<char> elements = {'a', 'b', 'c', 'd'};
  std::vector<int> offsets = {0, 1, 2, 3};

  auto schedule = util::FlattenSchedule(
      util::ConstructScheduleOverlapIO(offsets, elements));

  std::string result(schedule.begin(), schedule.end());
  EXPECT_EQ(result, "dcbadcbadcbadcbadcbadcba");
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
