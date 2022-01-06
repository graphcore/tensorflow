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
#include "tensorflow/compiler/plugin/poplar/driver/passes/redundant_triangular_mask_remover.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace m = match;
namespace poplarplugin {
namespace {

using RedundantTriangularMaskRemoverTest = HloTestBase;

int64 GetNumMultiUpdateAdds(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(),
                          IsPoplarInstruction(PoplarOp::MultiUpdateAdd));
}

TEST_F(RedundantTriangularMaskRemoverTest, TestCholesky) {
  std::string hlo_string = R"(
HloModule top
main {
  arg0 = f32[3,3]{1,0} parameter(0)
  iota0 = s32[3,3]{1,0} iota(), iota_dimension=0
  iota1 = s32[3,3]{1,0} iota(), iota_dimension=1
  compare = pred[3,3]{1,0} compare(s32[3,3]{1,0} iota0, s32[3,3]{1,0} iota1), direction=GE
  cholesky = f32[3,3]{1,0} cholesky(f32[3,3]{1,0} arg0), lower=true
  zero = f32[] constant(0)
  broadcast = f32[3,3]{1,0} broadcast(f32[] zero), dimensions={}
  ROOT select = f32[3,3]{1,0} select(pred[3,3]{1,0} compare, f32[3,3]{1,0} cholesky, f32[3,3]{1,0} broadcast)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Select(m::Compare(m::Iota(), m::Iota()),
                                    m::Op()
                                        .WithOpcode(HloOpcode::kCholesky)
                                        .WithOperand(0, m::Parameter(0)),
                                    m::Broadcast(m::ConstantScalar(0)))));

  CompilerAnnotations annotations(module);

  RedundantTriangularMaskRemover pass(annotations);
  EXPECT_TRUE(pass.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kCholesky)
                              .WithOperand(0, m::Parameter(0))));
}

TEST_F(RedundantTriangularMaskRemoverTest, TestCholeskyFlippedSelect) {
  std::string hlo_string = R"(
HloModule top
main {
  arg0 = f32[3,3]{1,0} parameter(0)
  iota0 = s32[3,3]{1,0} iota(), iota_dimension=0
  iota1 = s32[3,3]{1,0} iota(), iota_dimension=1
  compare = pred[3,3]{1,0} compare(s32[3,3]{1,0} iota0, s32[3,3]{1,0} iota1), direction=LE
  cholesky = f32[3,3]{1,0} cholesky(f32[3,3]{1,0} arg0), lower=true
  zero = f32[] constant(0)
  broadcast = f32[3,3]{1,0} broadcast(f32[] zero), dimensions={}
  ROOT select = f32[3,3]{1,0} select(pred[3,3]{1,0} compare, f32[3,3]{1,0} broadcast, f32[3,3]{1,0} cholesky)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Select(m::Compare(m::Iota(), m::Iota()),
                                    m::Broadcast(m::ConstantScalar(0)),
                                    m::Op()
                                        .WithOpcode(HloOpcode::kCholesky)
                                        .WithOperand(0, m::Parameter(0)))));

  CompilerAnnotations annotations(module);

  RedundantTriangularMaskRemover pass(annotations);
  EXPECT_TRUE(pass.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kCholesky)
                              .WithOperand(0, m::Parameter(0))));
}

TEST_F(RedundantTriangularMaskRemoverTest, TestTriangularSolve) {
  std::string hlo_string = R"(
HloModule top
main {
  arg0 = f32[3,3]{1,0} parameter(0)
  arg1 = f32[3,1]{1,0} parameter(1)
  iota0 = s32[3,1]{1,0} iota(), iota_dimension=0
  iota1 = s32[3,1]{1,0} iota(), iota_dimension=1
  compare = pred[3,1]{1,0} compare(s32[3,1]{1,0} iota0, s32[3,1]{1,0} iota1), direction=LT
  triangular-solve = f32[3,1]{1,0} triangular-solve(f32[3,3]{1,0} arg0, f32[3,1]{1,0} arg1), left_side=true, lower=false, transpose_a=NO_TRANSPOSE
  zero = f32[] constant(0)
  broadcast = f32[3,1]{1,0} broadcast(f32[] zero), dimensions={}
  ROOT select = f32[3,1]{1,0} select(pred[3,1]{1,0} compare, f32[3,1]{1,0} triangular-solve, f32[3,1]{1,0} broadcast)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Select(m::Compare(m::Iota(), m::Iota()),
                                    m::Op()
                                        .WithOpcode(HloOpcode::kTriangularSolve)
                                        .WithOperand(0, m::Parameter(0))
                                        .WithOperand(1, m::Parameter(1)),
                                    m::Broadcast(m::ConstantScalar(0)))));

  CompilerAnnotations annotations(module);

  RedundantTriangularMaskRemover pass(annotations);
  EXPECT_TRUE(pass.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kTriangularSolve)
                              .WithOperand(0, m::Parameter(0))
                              .WithOperand(1, m::Parameter(1))));
}

TEST_F(RedundantTriangularMaskRemoverTest, TestTriangularSolveFlippedSelect) {
  std::string hlo_string = R"(
HloModule top
main {
  arg0 = f32[3,3]{1,0} parameter(0)
  arg1 = f32[3,1]{1,0} parameter(1)
  iota0 = s32[3,1]{1,0} iota(), iota_dimension=0
  iota1 = s32[3,1]{1,0} iota(), iota_dimension=1
  compare = pred[3,1]{1,0} compare(s32[3,1]{1,0} iota0, s32[3,1]{1,0} iota1), direction=GT
  triangular-solve = f32[3,1]{1,0} triangular-solve(f32[3,3]{1,0} arg0, f32[3,1]{1,0} arg1), left_side=true, lower=false, transpose_a=NO_TRANSPOSE
  zero = f32[] constant(0)
  broadcast = f32[3,1]{1,0} broadcast(f32[] zero), dimensions={}
  ROOT select = f32[3,1]{1,0} select(pred[3,1]{1,0} compare, f32[3,1]{1,0} broadcast, f32[3,1]{1,0} triangular-solve)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(hlo_string, config).ValueOrDie();
  auto* module = module0.get();
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Select(m::Compare(m::Iota(), m::Iota()),
                                    m::Broadcast(m::ConstantScalar(0)),
                                    m::Op()
                                        .WithOpcode(HloOpcode::kTriangularSolve)
                                        .WithOperand(0, m::Parameter(0))
                                        .WithOperand(1, m::Parameter(1)))));

  CompilerAnnotations annotations(module);

  RedundantTriangularMaskRemover pass(annotations);
  EXPECT_TRUE(pass.Run(module).ValueOrDie());

  root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(Match(root, m::Op()
                              .WithOpcode(HloOpcode::kTriangularSolve)
                              .WithOperand(0, m::Parameter(0))
                              .WithOperand(1, m::Parameter(1))));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
