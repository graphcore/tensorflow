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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
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
  EXPECT_THAT(xla_groups[0].replica_ids(), ::testing::ElementsAre(0, 1, 2, 3));

  const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(xla_groups);
  EXPECT_OK(recovered);
  EXPECT_EQ(recovered.ValueOrDie(), groups);
}

TEST(PoplarReplicaGroupsTest, Orthogonal) {
  const auto groups = PoplarReplicaGroups::Orthogonal(4);
  EXPECT_EQ(groups.GroupSizeOr(42), 4);
  EXPECT_EQ(groups.ToString(), "orthogonal(group_size=4)");

  const auto xla_groups = groups.ToXlaReplicaGroups();
  EXPECT_EQ(xla_groups.size(), 2);
  EXPECT_THAT(xla_groups[0].replica_ids(), ::testing::ElementsAre(0, 2, 4, 6));
  EXPECT_THAT(xla_groups[1].replica_ids(), ::testing::ElementsAre(1, 3, 5, 7));

  const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(xla_groups);
  EXPECT_OK(recovered);
  EXPECT_EQ(recovered.ValueOrDie(), groups);
}

TEST(PoplarReplicaGroupsTest, FromXlaReplicaGroups) {
  xla::ReplicaGroup group0;
  group0.add_replica_ids(0);

  {
    const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups({group0});
    EXPECT_OK(recovered);
    EXPECT_EQ(recovered.ValueOrDie().GroupType(),
              PoplarReplicaGroups::Type::Consecutive);
  }

  group0.add_replica_ids(2);

  {
    const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups({group0});
    EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_THAT(
        recovered.status().error_message(),
        ::testing::StartsWith(
            "Unsupported replica group: Expected 1 at index 1, actual 2"));
  }

  xla::ReplicaGroup group1;
  group1.add_replica_ids(1);

  {
    const auto recovered =
        PoplarReplicaGroups::FromXlaReplicaGroups({group0, group1});
    EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_THAT(recovered.status().error_message(),
                ::testing::StartsWith(
                    "Irregular replica group size: Expected 2, actual 1"));
  }

  group1.add_replica_ids(3);

  {
    const auto recovered =
        PoplarReplicaGroups::FromXlaReplicaGroups({group0, group1});
    EXPECT_OK(recovered);
    EXPECT_EQ(recovered.ValueOrDie(), PoplarReplicaGroups::Orthogonal(2));
  }
}

TEST(PoplarReplicaGroupsTest, UnuspportedNumberOfXlaGroups) {
  std::vector<xla::ReplicaGroup> groups(3);

  {
    const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(groups);
    EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_THAT(recovered.status().error_message(),
                ::testing::StartsWith("Unsupported empty replica group"));
  }

  groups[0].add_replica_ids(0);
  {
    const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(groups);
    EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_THAT(recovered.status().error_message(),
                ::testing::StartsWith(
                    "Irregular replica group size: Expected 1, actual 0"));
  }

  groups[1].add_replica_ids(1);
  groups[2].add_replica_ids(2);
  {
    const auto recovered = PoplarReplicaGroups::FromXlaReplicaGroups(groups);
    EXPECT_EQ(recovered.status().code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_THAT(
        recovered.status().error_message(),
        ::testing::StartsWith("Unsupported number of replica groups: 3"));
  }
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

TEST(PoplarReplicaGroupsTest, ConsecutiveOrthogonal) {
  const auto consecutive0 = PoplarReplicaGroups::Consecutive(0);
  const auto orthogonal0 = PoplarReplicaGroups::Orthogonal(0);
  const auto default_instance = PoplarReplicaGroups();
  EXPECT_EQ(consecutive0, orthogonal0);
  EXPECT_EQ(consecutive0, default_instance);
  EXPECT_EQ(consecutive0.Hash(), orthogonal0.Hash());
  EXPECT_EQ(consecutive0.Hash(), default_instance.Hash());
  EXPECT_EQ(consecutive0.GroupSizeOr(42), 42);
  EXPECT_EQ(orthogonal0.GroupSizeOr(42), 42);

  const auto consecutive1 = PoplarReplicaGroups::Consecutive(1);
  const auto orthogonal1 = PoplarReplicaGroups::Orthogonal(1);
  EXPECT_NE(consecutive1, orthogonal1);
  EXPECT_NE(consecutive1.Hash(), orthogonal1.Hash());
  EXPECT_EQ(consecutive1.GroupSizeOrDie(), 1);
  EXPECT_EQ(orthogonal1.GroupSizeOrDie(), 1);
}

std::unique_ptr<HloComputation> CreateSumReduction(const Shape& shape) {
  auto builder = HloComputation::Builder("sum");

  auto* lhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));

  auto* rhs =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));

  auto* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));

  return builder.Build(add);
}

struct PoplarReplicaGroupsHloTest : public HloTestBase {
  std::unique_ptr<VerifiedHloModule> CreateModuleWithReplicaGroups(
      const PoplarReplicaGroups& groups, int64 replica_count) {
    auto module = CreateNewVerifiedModule("test", replica_count);
    auto shape = ShapeUtil::MakeShape(xla::F32, {});

    auto builder = HloComputation::Builder("entry");
    auto* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "param"));
    auto* reduction = module->AddEmbeddedComputation(CreateSumReduction(shape));
    auto* all_reduce = builder.AddInstruction(HloInstruction::CreateAllReduce(
        shape, std::vector<HloInstruction*>{param}, reduction,
        groups.ToXlaReplicaGroups(),
        /*constrain_layout=*/false,
        /*channel_id=*/absl::nullopt, /*use_global_device_ids=*/false));
    module->AddEntryComputation(builder.Build(all_reduce));

    return module;
  }
};

TEST_F(PoplarReplicaGroupsHloTest, VerifyModulesWithReplicaGroups) {
  // Check that the HloVerifier accepts the XLA replica groups that we produce.

  auto consecutive = CreateModuleWithReplicaGroups(
      PoplarReplicaGroups::Consecutive(4), /*replica_count=*/4);
  consecutive->VerifyOrAddFailure("consecutive");

  auto orthogonal = CreateModuleWithReplicaGroups(
      PoplarReplicaGroups::Orthogonal(4), /*replica_count=*/8);
  orthogonal->VerifyOrAddFailure("orthogonal");
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
