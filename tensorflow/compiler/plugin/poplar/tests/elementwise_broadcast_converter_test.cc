/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/elementwise_broadcast_converter.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

using ElementwiseBroadcastConvert = HloTestBase;

std::vector<HloPoplarUseDescription> GetInplaceDescriptions(
    const HloInstruction* inst) {
  auto fusion_config =
      inst->backend_config<PoplarBackendConfig>().ValueOrDie().fusion_config();
  std::vector<HloPoplarUseDescription> output;
  for (const auto& inplace_description : fusion_config.inplace_descriptions()) {
    output.push_back(HloPoplarUseDescription::FromProto(inplace_description));
  }
  return output;
}

std::unique_ptr<CompilerResources> GetMockResources(HloModule* module) {
  auto resources = CompilerResources::CreateTestDefault(module);
  resources->module_call_graph = CallGraph::Build(module);
  resources->main_graph = absl::make_unique<poplar::Graph>(
      poplar::Device::createCPUDevice(), poplar::replication_factor(1));
  poplin::addCodelets(*resources->main_graph);
  popnn::addCodelets(*resources->main_graph);
  popops::addCodelets(*resources->main_graph);
  poprand::addCodelets(*resources->main_graph);
  return std::move(resources);
}

TEST_F(ElementwiseBroadcastConvert, BinaryRHS) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,4] parameter(0)
  p1 = f16[4] parameter(1)
  bcast = f16[1,16,16,4] broadcast(p1), dimensions={3}
  ROOT %add = f16[1,16,16,4] add(p0, bcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* bcast = root->operand(1);
  const auto* p1 = bcast->operand(0);
  const auto* p0 = root->operand(0);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  EXPECT_EQ(root->operand(1), p1);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_binary_inplace");
  auto inplace_descriptions = GetInplaceDescriptions(root);
  EXPECT_EQ(inplace_descriptions.size(), 1);
  EXPECT_EQ(inplace_descriptions[0],
            HloPoplarUseDescription(0, ShapeIndex{}, ShapeIndex{},
                                    BufferUseKind::USE_ALIAS_READ_WRITE));
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op1->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op1->operand(0)->parameter_number(), 1);
}

TEST_F(ElementwiseBroadcastConvert, BinaryRHSConst) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,4] parameter(0)
  c1 = f16[] constant(1)
  bcast = f16[1,16,16,4] broadcast(c1), dimensions={}
  ROOT %divide = f16[1,16,16,4] divide(p0, bcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* p0 = root->operand(0);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_binary_inplace");
  auto inplace_descriptions = GetInplaceDescriptions(root);
  EXPECT_EQ(inplace_descriptions.size(), 1);
  EXPECT_EQ(inplace_descriptions[0],
            HloPoplarUseDescription(0, ShapeIndex{}, ShapeIndex{},
                                    BufferUseKind::USE_ALIAS_READ_WRITE));
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op1->operand(0)->opcode(), HloOpcode::kConstant);
}

TEST_F(ElementwiseBroadcastConvert, BinaryRHSScalarConst) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1,16,16,4] parameter(0)
  c1 = f16[] constant(1)
  bcast = f16[1,16,16,4] broadcast(c1), dimensions={}
  ROOT %divide = f16[1,16,16,4] divide(p0, bcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* p0 = root->operand(0);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_binary_inplace");
  auto inplace_descriptions = GetInplaceDescriptions(root);
  EXPECT_EQ(inplace_descriptions.size(), 1);
  EXPECT_EQ(inplace_descriptions[0],
            HloPoplarUseDescription(0, ShapeIndex{}, ShapeIndex{},
                                    BufferUseKind::USE_ALIAS_READ_WRITE));
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op1->operand(0)->opcode(), HloOpcode::kConstant);
}

TEST_F(ElementwiseBroadcastConvert, BinaryLHS) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[4] parameter(0)
  bcast = f16[1,16,16,4] broadcast(p0), dimensions={3}
  p1 = f16[1,16,16,4] parameter(1)
  ROOT %add = f16[1,16,16,4] add(bcast, p1)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* bcast = root->operand(0);
  const auto* p0 = bcast->operand(0);
  const auto* p1 = root->operand(1);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  EXPECT_EQ(root->operand(1), p1);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_binary");
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op0->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->operand(0)->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op1->parameter_number(), 1);
}

TEST_F(ElementwiseBroadcastConvert, BinaryLHSConst) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  c0 = f16[] constant(1)
  bcast = f16[1,16,16,4] broadcast(c0), dimensions={}
  p1 = f16[1,16,16,4] parameter(0)
  ROOT %add = f16[1,16,16,4] add(bcast, p1)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* p1 = root->operand(1);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p1);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_binary");
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op0->operand(0)->opcode(), HloOpcode::kConstant);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op1->parameter_number(), 0);
}

TEST_F(ElementwiseBroadcastConvert, TernaryArg0Bcast) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  c = f16[] constant(0)
  bcast = f16[1,16,16,4] broadcast(c), dimensions={}
  p0 = f16[1,16,16,4] parameter(0)
  p1 = f16[1,16,16,4] parameter(1)
  ROOT %clamp = f16[1,16,16,4] clamp(bcast, p0, p1)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* bcast = root->operand(0);
  const auto* c = bcast->operand(0);
  const auto* p0 = root->operand(1);
  const auto* p1 = root->operand(2);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  EXPECT_EQ(root->operand(1), p1);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_ternary");
  auto fusion_config =
      root->backend_config<PoplarBackendConfig>().ValueOrDie().fusion_config();
  EXPECT_EQ(fusion_config.inplace_descriptions_size(), 0);
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op0->operand(0)->opcode(), HloOpcode::kConstant);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op1->parameter_number(), 0);
  auto* fusion_op2 = fusion_root->operand(2);
  EXPECT_EQ(fusion_op2->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op2->parameter_number(), 1);
}

TEST_F(ElementwiseBroadcastConvert, TernaryArg2Bcast) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = f16[4] parameter(0)
  bcast = f16[1,16,16,4] broadcast(p0), dimensions={3}
  p1 = f16[1,16,16,4] parameter(1)
  p2 = f16[1,16,16,4] parameter(2)
  ROOT %clamp = f16[1,16,16,4] clamp(p1, p2, bcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* p0 = root->operand(0);
  const auto* p1 = root->operand(1);
  const auto* bcast = root->operand(2);
  const auto* p2 = bcast->operand(0);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  EXPECT_EQ(root->operand(1), p1);
  EXPECT_EQ(root->operand(2), p2);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_ternary");
  auto fusion_config =
      root->backend_config<PoplarBackendConfig>().ValueOrDie().fusion_config();
  EXPECT_EQ(fusion_config.inplace_descriptions_size(), 0);
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op1->parameter_number(), 1);
  auto* fusion_op2 = fusion_root->operand(2);
  EXPECT_EQ(fusion_op2->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op2->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op2->operand(0)->parameter_number(), 2);
}

TEST_F(ElementwiseBroadcastConvert, TernaryArg1ConstArg2Bcast) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = pred[1,16,16,4] parameter(0)
  c1 = f16[] constant(1)
  bcast1 = f16[1,16,16,4] broadcast(c1), dimensions={}
  p2 = f16[4] parameter(1)
  bcast2 = f16[1,16,16,4] broadcast(p2), dimensions={3}
  ROOT %clamp = f16[1,16,16,4] select(p0, bcast1, bcast2)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  auto* root = module0->entry_computation()->root_instruction();
  auto opcode = root->opcode();
  const auto* p0 = root->operand(0);
  const auto* bcast2 = root->operand(2);
  const auto* p2 = bcast2->operand(0);

  ElementwiseBroadcastConverter ebc;
  EXPECT_TRUE(ebc.Run(module0).ValueOrDie());

  root = module0->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), p0);
  EXPECT_EQ(root->operand(1), p2);
  auto* fusion_comp = root->fused_instructions_computation();
  EXPECT_EQ(fusion_comp->name(), "_pop_op_implicit_ternary");
  auto* fusion_root = fusion_comp->root_instruction();
  EXPECT_EQ(opcode, fusion_root->opcode());
  auto* fusion_op0 = fusion_root->operand(0);
  EXPECT_EQ(fusion_op0->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op0->parameter_number(), 0);
  auto* fusion_op1 = fusion_root->operand(1);
  EXPECT_EQ(fusion_op1->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op1->operand(0)->opcode(), HloOpcode::kConstant);
  auto* fusion_op2 = fusion_root->operand(2);
  EXPECT_EQ(fusion_op2->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(fusion_op2->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion_op2->operand(0)->parameter_number(), 1);
}

TEST_F(ElementwiseBroadcastConvert, AllInputsConstant) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  c0 = f16[] constant(0)
  bcast1 = f16[1,16,16,4] broadcast(c0), dimensions={}
  c1 = f16[] constant(0)
  bcast2 = f16[1,16,16,4] broadcast(c1), dimensions={}
  c2 = f16[] constant(0)
  bcast3 = f16[1,16,16,4] broadcast(c2), dimensions={}
  ROOT %clamp = f16[1,16,16,4] clamp(bcast1, bcast2, bcast3)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  ElementwiseBroadcastConverter ebc;
  EXPECT_FALSE(ebc.Run(module0).ValueOrDie());
}

class ElementwiseBroadcastConvertImplicit
    : public HloTestBase,
      public ::testing::WithParamInterface<std::string> {};

const char* binary_all_inputs_broadcasts = R"(
HloModule top

ENTRY c1 {
  p0 = f16[1] parameter(0)
  bcast1 = f16[1,16,16,4] broadcast(p0), dimensions={0}
  p1 = f16[] parameter(1)
  bcast2 = f16[1,16,16,4] broadcast(p1), dimensions={}
  ROOT %add = f16[1,16,16,4] add(bcast1, bcast2)
}
)";

const char* ternary_all_inputs_broadcasts = R"(
HloModule top

ENTRY c1 {
  p0 = f16[] parameter(0)
  bcast1 = f16[1,16,16,4] broadcast(p0), dimensions={}
  p1 = f16[] parameter(1)
  bcast2 = f16[1,16,16,4] broadcast(p1), dimensions={}
  p2 = f16[] parameter(2)
  bcast3 = f16[1,16,16,4] broadcast(p2), dimensions={}
  ROOT %clamp = f16[1,16,16,4] clamp(bcast1, bcast2, bcast3)
}
)";

TEST_P(ElementwiseBroadcastConvertImplicit, DoTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(GetParam()));
  EXPECT_TRUE(ElementwiseBroadcastConverter().Run(module.get()).ValueOrDie());

  // Make sure the module compiles correctly.
  auto resources = GetMockResources(module.get());
  auto entry_computation = module->entry_computation();
  EntryVisitor visitor(*resources.get(), entry_computation);
  TF_ASSERT_OK(entry_computation->Accept(&visitor));
}

INSTANTIATE_TEST_SUITE_P(ElementwiseBroadcastConvertImplicitCases,
                         ElementwiseBroadcastConvertImplicit,
                         ::testing::ValuesIn(std::vector<std::string>{
                             binary_all_inputs_broadcasts,
                             ternary_all_inputs_broadcasts}));

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
