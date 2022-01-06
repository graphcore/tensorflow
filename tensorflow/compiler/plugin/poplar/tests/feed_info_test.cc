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
#include "tensorflow/compiler/plugin/poplar/driver/tools/feed_info.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using FeedInfoTest = HloTestBase;

std::string GetInfeedName(const HloInstruction* inst) {
  PoplarFeedConfig config;
  config.ParseFromString(inst->infeed_config());
  return config.feed_id();
}

std::string GetOutfeeddName(const HloInstruction* inst) {
  PoplarFeedConfig config;
  config.ParseFromString(inst->outfeed_config());
  return config.feed_id();
}

TEST_F(FeedInfoTest, TestNoReuse) {
  std::string hlo_string = R"(
HloModule top

main {
  after-all = token[] after-all()
  infeed1 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed0\"\001\001(\001"
  infeed2 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed1\"\001\001(\001"
  constant = f32[] constant(1)
  outfeed1 = token[] outfeed(constant, after-all), outfeed_config="\010\001\022\005feed0\"\001\001(\001"
  outfeed2 = token[] outfeed(constant, after-all), outfeed_config="\010\001\022\005feed1\"\001\001(\001"
  ROOT t = () tuple()
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto mappings,
                          CanonicalizeFeedsInModule(module.get()));
  const HloInstruction* infeed1 = FindInstruction(module.get(), "infeed1");
  const HloInstruction* infeed2 = FindInstruction(module.get(), "infeed2");
  const HloInstruction* outfeed1 = FindInstruction(module.get(), "outfeed1");
  const HloInstruction* outfeed2 = FindInstruction(module.get(), "outfeed2");

  ASSERT_EQ(mappings.infeeds.size(), 2);
  ASSERT_EQ(mappings.outfeeds.size(), 2);

  EXPECT_EQ(mappings.infeeds.at("1"), "feed0");
  EXPECT_EQ(GetInfeedName(infeed1), "1");
  EXPECT_EQ(mappings.infeeds.at("2"), "feed1");
  EXPECT_EQ(GetInfeedName(infeed2), "2");

  EXPECT_EQ(mappings.outfeeds.at("3"), "feed0");
  EXPECT_EQ(GetOutfeeddName(outfeed1), "3");
  EXPECT_EQ(mappings.outfeeds.at("4"), "feed1");
  EXPECT_EQ(GetOutfeeddName(outfeed2), "4");
}

// In this test we check that feeds with the same config name are canonicalised
// to the same name.
TEST_F(FeedInfoTest, TestReuseFeedName) {
  std::string hlo_string = R"(
HloModule top

main {
  after-all = token[] after-all()
  infeed1 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed1\"\001\001(\001"
  infeed2 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed1\"\001\001(\001"
  constant = f32[] constant(1)
  outfeed1 = token[] outfeed(constant, after-all), outfeed_config="\010\001\022\005feed2\"\001\001(\001"
  outfeed2 = token[] outfeed(constant, after-all), outfeed_config="\010\001\022\005feed2\"\001\001(\001"
  ROOT t = () tuple()
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto mappings,
                          CanonicalizeFeedsInModule(module.get()));
  const HloInstruction* infeed1 = FindInstruction(module.get(), "infeed1");
  const HloInstruction* infeed2 = FindInstruction(module.get(), "infeed2");
  const HloInstruction* outfeed1 = FindInstruction(module.get(), "outfeed1");
  const HloInstruction* outfeed2 = FindInstruction(module.get(), "outfeed2");

  ASSERT_EQ(mappings.infeeds.size(), 1);
  ASSERT_EQ(mappings.outfeeds.size(), 1);

  EXPECT_EQ(mappings.infeeds.at("1"), "feed1");
  EXPECT_EQ(GetInfeedName(infeed1), "1");
  EXPECT_EQ(GetInfeedName(infeed2), "1");

  EXPECT_EQ(mappings.outfeeds.at("2"), "feed2");
  EXPECT_EQ(GetOutfeeddName(outfeed1), "2");
  EXPECT_EQ(GetOutfeeddName(outfeed2), "2");
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
