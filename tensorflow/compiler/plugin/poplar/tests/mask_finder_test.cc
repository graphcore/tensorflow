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

#include "tensorflow/compiler/plugin/poplar/driver/passes/mask_finder.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using SimpleMaskFinderTest = HloTestBase;

TEST_F(SimpleMaskFinderTest, TestInvalidInstruction) {
  static const char* hlo_string = R"(
  HloModule top

  ENTRY main {
    p.0 = f32[3,3] parameter(0)
    p.1 = f32[] parameter(1)
    iota.1 = s32[3,3] iota(), iota_dimension=0
    iota.2 = s32[3,3] iota(), iota_dimension=1
    compare = pred[3,3] compare(iota.1, iota.2), direction=LT
    const = f32[] constant(1)
    add = f32[] add(p.1, const)
    broadcast = f32[3,3] broadcast(add), dimensions={}
    ROOT select = f32[3,3] select(compare, p.0, broadcast)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MaskFinder().Run(module.get()));
  EXPECT_FALSE(changed);
}

struct MaskFinderTestSpec {
  const char* name;
  std::string hlo;
  std::function<void(const HloModule*)> verify;
};

std::ostream& operator<<(std::ostream& os, const MaskFinderTestSpec& spec) {
  return os << "{ name: " << spec.name << "}";
}

class MaskFinderTest
    : public HloPoplarTestBase,
      public ::testing::WithParamInterface<MaskFinderTestSpec> {};

static std::string GetSimpleCompareTestSource(const std::string& direction,
                                              bool inverted) {
  return absl::StrReplaceAll(
      R"(
  HloModule top

  ENTRY main {
    p.0 = f32[3,3] parameter(0)
    iota.1 = s32[3,3] iota(), iota_dimension=0
    iota.2 = s32[3,3] iota(), iota_dimension=1
    compare = pred[3,3] compare(iota.1, iota.2), direction=$DIR
    const = f32[] constant(0)
    broadcast = f32[3,3] broadcast(const), dimensions={}
    ROOT select = f32[3,3] select(compare, $ARGS)
  }
  )",
      {{"$DIR", direction},
       {"$ARGS", inverted ? "broadcast, p.0" : "p.0, broadcast"}});
}

static std::string GetNonIotaCompareTestSource() {
  return R"(
  HloModule top

  ENTRY main {
    p.0 = f32[3,3] parameter(0)
    iota.1 = s32[3,3] iota(), iota_dimension=0
    const = s32[] constant(1)
    broadcast.1 = s32[3,3] broadcast(const), dimensions={}
    compare = pred[3,3] compare(iota.1, broadcast.1), direction=LE
    const.1 = f32[] constant(1)
    broadcast = f32[3,3] broadcast(const.1), dimensions={}
    ROOT select = f32[3,3] select(compare, p.0, broadcast)
  }
  )";
}

static std::string GetDotCompareTestSource() {
  return R"(
  HloModule top
  sum.1 {
    p.0 = s32[] parameter(0)
    p.1 = s32[] parameter(1)
    ROOT sum = s32[] add(p.0, p.1)
  }

  ENTRY main {
    p.0 = f32[3,3] parameter(0)
    iota.1 = s32[3,1] iota(), iota_dimension=0
    iota.2 = s32[1,3] iota(), iota_dimension=1
    convert.1 = f32[3,1] convert(iota.1)
    convert.2 = f32[1,3] convert(iota.2)
    dot = f32[3,3] dot(convert.1, convert.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    convert.3 = s32[3,3] convert(dot)
    transpose = s32[3,3] transpose(convert.3), dimensions={1, 0}

    const.s0 = s32[] constant(0)
    reduce = s32[] reduce(transpose, const.s0), dimensions={0, 1}, to_apply=sum.1

    const.s4 = s32[] constant(8)
    mean = s32[] subtract(reduce, const.s4)
    mean.bcast = s32[3,3] broadcast(mean), dimensions={}

    compare = pred[3,3] compare(transpose, mean.bcast), direction=LE
    const.f1 = f32[] constant(1)
    broadcast = f32[3,3] broadcast(const.f1), dimensions={}
    ROOT select = f32[3,3] select(compare, p.0, broadcast)
  }
  )";
}

static std::string GetSharedInstructionsTestSource() {
  return R"(
  HloModule top

  ENTRY main {
    param.0 = f32[3,3] parameter(0)
    iota.1 = s32[3,3] iota(), iota_dimension=0
    iota.2 = s32[3,3] iota(), iota_dimension=1

    const.f0 = f32[] constant(0)
    broadcast.f0 = f32[3,3] broadcast(const.f0), dimensions={}
    const.f1 = f32[] constant(1)
    broadcast.f1 = f32[3,3] broadcast(const.f1), dimensions={}

    compare.d = pred[3,3] compare(iota.1, iota.2), direction=EQ
    select.d = f32[3,3] select(compare.d, broadcast.f1, param.0)

    compare.1 = pred[3,3] compare(iota.1, iota.2), direction=LE
    select.1 = f32[3,3] select(compare.1, select.d, broadcast.f0)

    compare.2 = pred[3,3] compare(iota.1, iota.2), direction=GE
    select.2 = f32[3,3] select(compare.2, select.d, broadcast.f1)

    compare.3 = pred[3,3] and(compare.1, compare.2)
    select.3 = f32[3,3] select(compare.3, param.0, broadcast.f0)

    add = f32[3,3]  add(select.1, select.2)
    ROOT add.2 = f32[3,3] add(add, select.3)
  }
  )";
}

void RootInstructionIsMaskFusion(const HloModule* module) {
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsPopOpsFusion(root, "mask"));
}

void EveryFusionIsMask(const HloModule* module) {
  // Expect at least one fusion.
  EXPECT_GT(module->computation_count(), 1);
  for (const HloComputation* comp : module->computations()) {
    if (comp == module->entry_computation()) {
      continue;
    }
    const HloInstruction* root = comp->root_instruction();
    EXPECT_TRUE(IsPopOpsFusion(root, "mask"));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MaskFinderTestCases, MaskFinderTest,
    ::testing::ValuesIn(std::vector<MaskFinderTestSpec>{
        {"CompareLtLhs", GetSimpleCompareTestSource("LT", false),
         RootInstructionIsMaskFusion},
        {"CompareGtLhs", GetSimpleCompareTestSource("GT", false),
         RootInstructionIsMaskFusion},
        {"CompareLeLhs", GetSimpleCompareTestSource("LE", false),
         RootInstructionIsMaskFusion},
        {"CompareGeLhs", GetSimpleCompareTestSource("GE", false),
         RootInstructionIsMaskFusion},
        {"CompareEqLhs", GetSimpleCompareTestSource("EQ", false),
         RootInstructionIsMaskFusion},
        {"CompareNeLhs", GetSimpleCompareTestSource("NE", false),
         RootInstructionIsMaskFusion},
        {"CompareLtRhs", GetSimpleCompareTestSource("LT", true),
         RootInstructionIsMaskFusion},
        {"CompareGtRhs", GetSimpleCompareTestSource("GT", true),
         RootInstructionIsMaskFusion},
        {"CompareLeRhs", GetSimpleCompareTestSource("LE", true),
         RootInstructionIsMaskFusion},
        {"CompareGeRhs", GetSimpleCompareTestSource("GE", true),
         RootInstructionIsMaskFusion},
        {"CompareEqRhs", GetSimpleCompareTestSource("EQ", true),
         RootInstructionIsMaskFusion},
        {"CompareNeRhs", GetSimpleCompareTestSource("NE", true),
         RootInstructionIsMaskFusion},
        {"CompareConst", GetNonIotaCompareTestSource()},
        {"CompareDot", GetDotCompareTestSource()},
        {"SharedInstructions", GetSharedInstructionsTestSource(),
         EveryFusionIsMask},
    }),
    [](const ::testing::TestParamInfo<MaskFinderTestSpec>& spec)
        -> std::string { return spec.param.name; });

TEST_P(MaskFinderTest, DoTest) {
  auto param = GetParam();
  Literal p0 = LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(param.hlo));

  TF_ASSERT_OK_AND_ASSIGN(auto expected,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {&p0}));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, MaskFinder().Run(module.get()));
  ASSERT_TRUE(changed);
  ASSERT_IS_OK(HloDCE().Run(module.get()).status());
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          ExecuteNoHloPassesOnIpuModel(module.get(), {&p0}));
  EXPECT_EQ(expected.size(), 1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected[0], result[0],
                                           ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
