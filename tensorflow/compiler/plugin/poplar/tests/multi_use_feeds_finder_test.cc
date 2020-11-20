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
#include "tensorflow/compiler/plugin/poplar/driver/passes/multi_use_feeds_finder.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using MultiUseFeedsFinderTest = HloTestBase;

TEST_F(MultiUseFeedsFinderTest, TestNoReuse) {
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
  EXPECT_FALSE(MultiUseFeedsFinder().Run(module.get()).ValueOrDie());
}

TEST_F(MultiUseFeedsFinderTest, TestReuse) {
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
  EXPECT_TRUE(MultiUseFeedsFinder().Run(module.get()).ValueOrDie());
  HloInstruction* infeed1 = FindInstruction(module.get(), "infeed1");
  HloInstruction* infeed2 = FindInstruction(module.get(), "infeed2");
  HloInstruction* outfeed1 = FindInstruction(module.get(), "outfeed1");
  HloInstruction* outfeed2 = FindInstruction(module.get(), "outfeed2");

  PoplarFeedConfig config;
  config.ParseFromString(infeed1->infeed_config());
  EXPECT_TRUE(config.reusable());

  config.ParseFromString(infeed2->infeed_config());
  EXPECT_TRUE(config.reusable());

  config.ParseFromString(outfeed2->outfeed_config());
  EXPECT_TRUE(config.reusable());

  config.ParseFromString(outfeed2->outfeed_config());
  EXPECT_TRUE(config.reusable());
}

TEST_F(MultiUseFeedsFinderTest, TestShardingMissmatch) {
  std::string hlo_string = R"(
HloModule top

main {
  after-all = token[] after-all()
  infeed1 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed1\"\001\001(\001", sharding={{maximal device=0}, {maximal device=0}}
  infeed2 = ((f32[2,4,4,2]), token[]) infeed(after-all), infeed_config="\010\001\022\005feed1\"\001\001(\001", sharding={{maximal device=1}, {maximal device=1}}
  ROOT t = () tuple()
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto status = MultiUseFeedsFinder().Run(module.get());
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
