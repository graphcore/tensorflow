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

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/types/span.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace hash_util {
namespace {

template <typename T>
class AbslSpanHashTest : public HloTestBase {
 public:
  void VerifyHasher() {
    std::vector<T> v0{};
    std::vector<T> v1{1};
    std::vector<T> v2{2};

    absl::Span<const T> s0{v0};
    absl::Span<const T> s1{v1};
    absl::Span<const T> s2{v2};

    auto hash_fun = std::hash<absl::Span<const T>>{};

    std::size_t h0 = hash_fun(s0);
    std::size_t h1 = hash_fun(s1);
    std::size_t h2 = hash_fun(s2);

    EXPECT_NE(h0, h1);
    EXPECT_NE(h0, h2);
    EXPECT_NE(h1, h2);
  }
};

TYPED_TEST_SUITE_P(AbslSpanHashTest);
TYPED_TEST_P(AbslSpanHashTest, VerifyHasherTest) { this->VerifyHasher(); }
REGISTER_TYPED_TEST_SUITE_P(AbslSpanHashTest, VerifyHasherTest);

typedef ::testing::Types<int, float> TestTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, AbslSpanHashTest, TestTypes);

}  // namespace
}  // namespace hash_util
}  // namespace poplarplugin
}  // namespace xla
