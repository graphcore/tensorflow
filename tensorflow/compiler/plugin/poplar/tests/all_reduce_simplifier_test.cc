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

#include "tensorflow/compiler/plugin/poplar/driver/passes/all_reduce_simplifier.h"

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

namespace m = match;

struct AllReduceSimplifierTestSpec {
  PrimitiveType gradient_type;
};

void PrintTo(const AllReduceSimplifierTestSpec& spec, std::ostream* os) {
  *os << PrimitiveType_Name(spec.gradient_type);
}

struct AllReduceSimplifierTest
    : public HloTestBase,
      public ::testing::WithParamInterface<AllReduceSimplifierTestSpec> {};

INSTANTIATE_TEST_SUITE_P(AllReduceSimplifierTestCases, AllReduceSimplifierTest,
                         ::testing::Values(AllReduceSimplifierTestSpec{F16},
                                           AllReduceSimplifierTestSpec{F32},
                                           AllReduceSimplifierTestSpec{U8},
                                           AllReduceSimplifierTestSpec{U32},
                                           AllReduceSimplifierTestSpec{S32}));
// Other data types have been disabled due to missing data types in
// the implementation of Literal::GetIntegralAsS64() in TF1

string ReplaceParams(absl::string_view s,
                     const AllReduceSimplifierTestSpec& spec) {
  return absl::StrReplaceAll(
      s, {
             {"$GT",
              primitive_util::LowercasePrimitiveTypeName(spec.gradient_type)},
         });
}

TEST_P(AllReduceSimplifierTest, Simple) {
  const char* hlo_template = R"(
    HloModule m
    mean {
      y = $GT[] parameter(1)
      x = $GT[] parameter(0), control-predecessors={y}
      norm_y = $GT[] custom-call(y), custom_call_target="ReplicationNormalise"
      ROOT add = $GT[] add(x, norm_y), backend_config="{\"isInplace\":true}"
    }

    ENTRY main {
      arg0 = $GT[1000] parameter(0)
      ROOT all-reduce = $GT[1000] all-reduce(arg0), to_apply=mean
    }
  )";

  const auto param = GetParam();
  const auto hlo_string = ReplaceParams(hlo_template, param);

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_TRUE(CustomOpReplacer().Run(m.get()).ValueOrDie());

  auto* module_root_instruction = m->entry_computation()->root_instruction();
  auto element_type = module_root_instruction->shape().element_type();

  ASSERT_TRUE(AllReduceSimplifier(8).Run(m.get()).ValueOrDie());

  module_root_instruction = m->entry_computation()->root_instruction();

  EXPECT_THAT(module_root_instruction,
              GmockMatch(m::AllReduce(m::Parameter(0))));

  HloComputation* all_reduce_computation = module_root_instruction->to_apply();

  auto* all_reduce_root = all_reduce_computation->root_instruction();
  EXPECT_THAT(
      all_reduce_root,
      GmockMatch(m::Add(m::Parameter(0),
                        m::Divide(m::Parameter(1), m::ConstantScalar(8)))));

  ASSERT_TRUE(all_reduce_root->operand(1)->shape().element_type() ==
              element_type);
  ASSERT_TRUE(all_reduce_root->operand(1)->operand(0)->shape().element_type() ==
              element_type);
  ASSERT_TRUE(all_reduce_root->operand(1)->operand(1)->shape().element_type() ==
              element_type);
}
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
