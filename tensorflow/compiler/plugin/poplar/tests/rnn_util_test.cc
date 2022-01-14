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

#include <string>

#include <poplar/OptionFlags.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tools/rnn_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using RnnUtilTest = HloTestBase;

TEST_F(RnnUtilTest, TestDeserialiseOptionsOntoOptionFlags) {
  poplar::OptionFlags option_flags;
  option_flags.set({{"a", "0"}});
  {
    auto status = DeserialiseOptionsOntoOptionFlags("{\"b\": 1}", option_flags);
    EXPECT_TRUE(status.ok());
  }
  {
    auto status = DeserialiseOptionsOntoOptionFlags("{}", option_flags);
    EXPECT_TRUE(status.ok());
  }
  EXPECT_EQ(option_flags.at("a").cloneAsString(), "0");
  EXPECT_EQ(option_flags.at("b").cloneAsString(), "1");
}

TEST_F(RnnUtilTest, TestDeserialiseOptionsOntoOptionFlagsInvalidJson) {
  poplar::OptionFlags option_flags;
  {
    auto status = DeserialiseOptionsOntoOptionFlags("[1, 2, 3]", option_flags);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(
        status.error_message(),
        ::testing::StartsWith("[Poplar][Graph construction] parse_error:"));
  }
  {
    auto status = DeserialiseOptionsOntoOptionFlags("foo", option_flags);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(
        status.error_message(),
        ::testing::StartsWith("[Poplar][Graph construction] parse_error:"));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
