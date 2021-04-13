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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"

#include <gmock/gmock-matchers.h>

#include "tensorflow/core/platform/test.h"

namespace xla {
namespace poplarplugin {
namespace {

TEST(PoplarReplicaGroupsTest, Default) {
  const auto groups = PoplarReplicaGroups();
  EXPECT_EQ(groups.GroupSizeOr(42), 42);
  EXPECT_EQ(groups.ToString(), "single(group_size=all)");

  const auto xla_groups = groups.ToXlaReplicaGroups();
  EXPECT_TRUE(xla_groups.empty());

  const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(xla_groups);
  EXPECT_OK(recovered);
  EXPECT_EQ(recovered.ValueOrDie(), groups);
}

TEST(PoplarReplicaGroupsTest, Consecutive) {
  const auto groups = PoplarReplicaGroups::Consecutive(4);
  EXPECT_EQ(groups.GroupSizeOr(42), 4);
  EXPECT_EQ(groups.ToString(), "consecutive(group_size=4)");

  const auto xla_groups = groups.ToXlaReplicaGroups();
  EXPECT_EQ(xla_groups.size(), 1);
  EXPECT_EQ(xla_groups[0].replica_ids_size(), 4);

  const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(xla_groups);
  EXPECT_OK(recovered);
  EXPECT_EQ(recovered.ValueOrDie(), groups);
}

TEST(PoplarReplicaGroupsTest, NonConsecutiveXlaReplicaGroup) {
  xla::ReplicaGroup group;
  group.add_replica_ids(0);
  const auto recovered1 = PoplarReplicaGroups::FromXlaReplicaGroups({group});
  EXPECT_OK(recovered1);

  group.add_replica_ids(2);
  const auto recovered2 = PoplarReplicaGroups::FromXlaReplicaGroups({group});
  EXPECT_EQ(recovered2.status().code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      recovered2.status().error_message(),
      ::testing::StartsWith("Unsupported non-consecutive replica group"));
}

TEST(PoplarReplicaGroupsTest, MultipleXlaReplicaGroups) {
  const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(
      {xla::ReplicaGroup(), xla::ReplicaGroup()});
  EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      recovered.status().error_message(),
      ::testing::StartsWith("Expected a single replica group, but got 2"));
}

TEST(PoplarReplicaGroupsTest, EqualsAndHash) {
  const auto consecutive1 = PoplarReplicaGroups::Consecutive(4);
  const auto consecutive2 = PoplarReplicaGroups::Consecutive(4);
  const auto consecutive3 = PoplarReplicaGroups::Consecutive(8);

  EXPECT_EQ(consecutive1, consecutive2);
  EXPECT_NE(consecutive1, consecutive3);
  EXPECT_EQ(consecutive1.Hash(), consecutive2.Hash());
  EXPECT_NE(consecutive1.Hash(), consecutive3.Hash());

  const auto single1 = PoplarReplicaGroups();
  const auto single2 = PoplarReplicaGroups();
  EXPECT_EQ(single1, single2);
  EXPECT_EQ(single1.Hash(), single2.Hash());
  EXPECT_NE(single1, consecutive1);
  EXPECT_NE(single1.Hash(), consecutive1.Hash());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
