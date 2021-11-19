/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"

#include <algorithm>
#include <iterator>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_fuser.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/host_compute_barrier_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/clustering_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

// Utility matchers..
MATCHER_P2(ContainsNOps, opcode, expected_count, "") {
  const auto& sequence = arg;
  const auto pred = [&](const HloInstruction* inst) {
    return inst->opcode() == opcode;
  };

  const auto count = absl::c_count_if(sequence, pred);
  const auto success = count == expected_count;
  if (!success) {
    *result_listener << "Sequence contains " << count << " "
                     << HloOpcodeString(opcode) << " but expected "
                     << expected_count;
  }

  return success;
}

MATCHER_P2(ContainsNPoplarOps, opcode, expected_count, "") {
  const auto& sequence = arg;

  const auto count = absl::c_count_if(sequence, IsPoplarInstruction(opcode));
  const auto success = count == expected_count;
  if (!success) {
    *result_listener << "Sequence contains " << count << " "
                     << PoplarOp_Name(opcode) << " but expected "
                     << expected_count;
  }

  return success;
}

MATCHER_P(AllSharded, expected_sharding, "") {
  const auto& instructions = arg;
  for (auto inst : instructions) {
    if (!inst->has_sharding()) {
      *result_listener << "Expected " << inst->name() << " to be sharded.";
      return false;
    }
    if (inst->sharding() != expected_sharding) {
      *result_listener << "Expected " << inst->name() << " to have sharding "
                       << expected_sharding.ToString();
      return false;
    }
  }

  return true;
}

struct CombineInstructionsTest : HloPoplarTestBase {
  static HloMemoryScheduler CreateHloScheduler(
      IpuSchedulerAlgorithm algorithm) {
    HloMemoryScheduler scheduler(
        [](const BufferValue& buffer) {
          return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
        },
        ComputationSchedulerToModuleScheduler(
            IpuToMemorySchedulerAlgorithm(algorithm)));

    return scheduler;
  }

  static HloMemoryScheduler CreateLookAheadScheduler(
      const CompilerInformation& information) {
    return CreateHloScheduler(CreateClusteringMemoryScheduler(information));
  }

  void SetUp() override { config_.set_debug_options(GetDebugOptionsForTest()); }

  HloModuleConfig config_;
};

const char* all_reduce_hlo = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";
TEST_F(CombineInstructionsTest, TestSyncScheduler) {
  auto module_or_status = ParseAndReturnVerifiedModule(all_reduce_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateHloScheduler(CreateSyncListMemoryScheduler(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 8);
  ASSERT_THAT(seq, ContainsNOps(HloOpcode::kAllReduce, 1));
}

TEST_F(CombineInstructionsTest, TestLookAheadScheduler) {
  auto module_or_status = ParseAndReturnVerifiedModule(all_reduce_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 8);
  ASSERT_THAT(seq, ContainsNOps(HloOpcode::kAllReduce, 1));
}

TEST_F(CombineInstructionsTest, TestMergeInterIpuCopiesLookAheadScheduler) {
  std::string hlo_string = R"(
HloModule top

loop_body (arg_tuple.0: (s32[], f32[2], s32[])) -> (s32[], f32[2], s32[]) {
  after-all.1 = token[] after-all(), sharding={maximal device=0}
  infeed = ((f32[2]), token[]) infeed(after-all.1), infeed_config="\010\002\022\005feed0", sharding={{maximal device=0}, {maximal device=0}}
  get-tuple-element.5 = (f32[2]) get-tuple-element(infeed), index=0, sharding={{maximal device=0}}, backend_config="{\"isInplace\":true}"
  get-tuple-element.6 = f32[2] get-tuple-element(get-tuple-element.5), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  multiply = f32[2] multiply(get-tuple-element.6, get-tuple-element.6), sharding={maximal device=0}
  constant.7 = s32[] constant(2), sharding={maximal device=0}
  arg_tuple.0 = (s32[], f32[2], s32[]) parameter(0), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  get-tuple-element.4 = f32[2] get-tuple-element(arg_tuple.0), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  add.1 = f32[2] add(get-tuple-element.4, get-tuple-element.6), sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  add.2 = f32[2] add(add.1, multiply), sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  get-tuple-element.3 = s32[] get-tuple-element(arg_tuple.0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  ROOT tuple.1 = (s32[], f32[2], s32[]) tuple(get-tuple-element.3, add.2, constant.7), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"isInplace\":true}"
}

_pop_op_wide_const () -> f32[2] {
  constant.1 = f32[] constant(0)
  ROOT broadcast.2 = f32[2] broadcast(constant.1), dimensions={}
}

ENTRY entry () -> f32[2] {
  fusion = f32[2] fusion(), kind=kCustom, calls=_pop_op_wide_const, sharding={maximal device=0}, backend_config="{}"
  constant.6 = s32[] constant(2), sharding={maximal device=0}
  tuple.7 = (s32[], f32[2], s32[]) tuple(constant.6, fusion, constant.6), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"isInplace\":true}"
  call = (s32[], f32[2], s32[]) call(tuple.7), to_apply=loop_body, sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"repeatConfig\":{\"isRepeatLoop\":true,\"repeatCount\":\"2\"},\"isInplace\":true}"
  ROOT get-tuple-element.52 = f32[2] get-tuple-element(call), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* repeat = comp->GetInstructionWithName("call");
  auto* body = repeat->to_apply();

  EXPECT_EQ(body->instruction_count(), 12);
  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  // Expect three inter IPU copies to have been inserted.
  EXPECT_EQ(body->instruction_count(), 15);
  ASSERT_THAT(body->instructions(),
              ContainsNPoplarOps(PoplarOp::IpuInterCopy, 3));

  // Schedule and combine.
  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_inter_ipu_copies_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());
  // Two IPU copies have been merged.
  EXPECT_THAT(body->instructions(),
              ContainsNPoplarOps(PoplarOp::IpuInterCopy, 2));
  EXPECT_EQ(body->instruction_count(), 16);

  // Combined IpuInterCopy and it's GTEs will be inserted between add.1 and
  // add.2
  const HloSharding expected_sharding = HloSharding::AssignDevice(1);

  auto add_1 = FindInstruction(module, "add.1");
  EXPECT_THAT(add_1->users(), AllSharded(expected_sharding));

  auto add_2 = FindInstruction(module, "add.2");
  EXPECT_THAT(add_2->operands(), AllSharded(expected_sharding));
}

const char* gradient_accumulation_hlo = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %a0 = f16[4] all-reduce(arg0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga0 = f16[4] custom-call(norm0), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %arg1 = f16[4] parameter(1)
  %a1 = f16[4] all-reduce(arg1), to_apply=add
  %norm1 = f16[4] custom-call(a1), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga1 = f16[4] custom-call(norm1), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %arg2 = f16[4] parameter(2)
  %a2 = f16[4] all-reduce(arg2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga2 = f16[4] custom-call(norm2), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(ga0, ga1, ga2)
}
  )";
TEST_F(CombineInstructionsTest, TestLookAheadSchedulerGradientAccumulation) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(gradient_accumulation_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 11);
  ASSERT_THAT(seq, ContainsNPoplarOps(
                       PoplarOp::StatefulGradientAccumulateAndAllReduce, 1));
}

const char* gradient_accumulation_with_momentum_hlo = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %momentum = f16[] parameter(0)
  %grad0 = f16[4] parameter(1)
  %accum0 = f16[4] parameter(2)
  %grad1 = f16[4] parameter(3)
  %accum1 = f16[4] parameter(4)
  %grad2 = f16[4] parameter(5)
  %accum2 = f16[4] parameter(6)


  %a0 = f16[4] all-reduce(grad0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga0 = (f16[4], f16[4]) custom-call(accum0, norm0, momentum), custom_call_target="StatefulGradientAccumulateWithMomentum", backend_config="{\"num_mini_batches\":4}\n"
  %new_grad0 = f16[4] get-tuple-element(ga0), index=0
  %new_accum0 = f16[4] get-tuple-element(ga0), index=1

  %a1 = f16[4] all-reduce(grad1), to_apply=add
  %norm1 = f16[4] custom-call(a1), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga1 = (f16[4], f16[4]) custom-call(accum1, norm1, momentum), custom_call_target="StatefulGradientAccumulateWithMomentum", backend_config="{\"num_mini_batches\":4}\n"
  %new_grad1 = f16[4] get-tuple-element(ga1), index=0
  %new_accum1 = f16[4] get-tuple-element(ga1), index=1

  %a2 = f16[4] all-reduce(grad2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga2 = (f16[4], f16[4]) custom-call(accum2, norm2, momentum), custom_call_target="StatefulGradientAccumulateWithMomentum", backend_config="{\"num_mini_batches\":4}\n"
  %new_grad2 = f16[4] get-tuple-element(ga2), index=0
  %new_accum2 = f16[4] get-tuple-element(ga2), index=1

  ROOT %tuple = (f16[4], f16[4], f16[4], f16[4], f16[4], f16[4]) tuple(new_grad0, new_accum0, new_grad1, new_accum1, new_grad2, new_accum2)
}
  )";
TEST_F(CombineInstructionsTest,
       TestLookAheadSchedulerGradientAccumulationWithMomentum) {
  auto module_or_status = ParseAndReturnVerifiedModule(
      gradient_accumulation_with_momentum_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 23);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 23);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 17);

  auto scheduler = CreateHloScheduler(CreateClusteringMemoryScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024)));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are set.
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 9);

  auto s = module->schedule().sequence(entry);
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 24);
  ASSERT_THAT(
      seq,
      ContainsNPoplarOps(
          PoplarOp::StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm,
          1));
}

TEST_F(CombineInstructionsTest,
       TestLookAheadSchedulerGradientAccumulationDifferentMiniBatches) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %a0 = f16[4] all-reduce(arg0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga0 = f16[4] custom-call(norm0), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %arg1 = f16[4] parameter(1)
  %a1 = f16[4] all-reduce(arg1), to_apply=add
  %ga1 = f16[4] custom-call(a1), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":5}\n"
  %norm1 = f16[4] custom-call(ga1), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %arg2 = f16[4] parameter(2)
  %a2 = f16[4] all-reduce(arg2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga2 = f16[4] custom-call(norm2), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":6}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(ga0, norm1, ga2)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);
}

const char* reduce_scatter_hlo = R"(
HloModule top

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %r0 = f16[4] custom-call(arg0), custom_call_target="ReduceScatter", backend_config="{\"op\": \"COLLECTIVE_OP_ADD\", \"replica_group_size\": 2}\n"
  %arg1 = f16[4] parameter(1)
  %r1 = f16[4] custom-call(arg1), custom_call_target="ReduceScatter", backend_config="{\"op\": \"COLLECTIVE_OP_ADD\", \"replica_group_size\": 2}\n"
  %arg2 = f16[4] parameter(2)
  %r2 = f16[4] custom-call(arg2), custom_call_target="ReduceScatter", backend_config="{\"op\": \"COLLECTIVE_OP_ADD\", \"replica_group_size\": 4}\n"
  %arg3 = f16[4] parameter(3)
  %r3 = f16[4] custom-call(arg3), custom_call_target="ReduceScatter", backend_config="{\"op\": \"COLLECTIVE_OP_LOCAL\", \"replica_group_size\": 4}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(r0, r1, r2)
}
  )";
TEST_F(CombineInstructionsTest, TestCombineReduceScatter) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(reduce_scatter_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  ASSERT_THAT(entry->instructions(),
              ContainsNPoplarOps(PoplarOp::ReduceScatter, 4));

  // Schedule and combine.
  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_reduce_scatter_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  // There should be a three reduce scatters:
  // - One combined with with op=ADD replica_group_size=2
  // - One left alone with with op=ADD replica_group_size=4
  // - One left alone with with op=LOCAL replica_group_size=4
  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::ReduceScatter, 3));
}

const char* send_to_host_hlo = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[]) -> () {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2 = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg1\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %send2 = () custom-call(f32[] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg2\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %ret = () tuple()
}
  )";
TEST_F(CombineInstructionsTest, TestCombineSendToHost) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(send_to_host_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 8;

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_send_recv_cluster_size(
          max_send_recv_cluster_size));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::SendToHost, 1));

  auto send_inst = Cast<HloSendToHostInstruction>(
      *absl::c_find_if(seq, IsPoplarInstruction(SendToHost)));

  // Check that it was merged correctly.
  EXPECT_EQ(send_inst->operand_count(), 2);
  EXPECT_EQ(send_inst->RendezvousKeys().size(), 2);
  EXPECT_EQ(send_inst->RendezvousKeys()[0], send_inst->operand(0)->name());
  EXPECT_EQ(send_inst->RendezvousKeys()[1], send_inst->operand(1)->name());

  // Check that various other state was maintained. The metadata is maintained
  // on a best-effort basis, as it might not have been the same for all merged
  // instructions. But if they were, as in this test, it should be kept.
  EXPECT_TRUE(send_inst->custom_call_has_side_effect());
  EXPECT_EQ(send_inst->metadata().op_type(), "XlaHostCompute");
  EXPECT_EQ(send_inst->metadata().op_name(), "host_compute");
}

TEST_F(CombineInstructionsTest, TestSendToHostNotCombinedWhenBufferTooSmall) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(send_to_host_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 4;

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_send_recv_cluster_size(
          max_send_recv_cluster_size));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::SendToHost, 2));
}

const char* recv_from_host_hlo = R"(
HloModule top

ENTRY %top () -> (f32[], f32[]) {
  %recv1 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv1_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %recv2 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv2_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %tuple = (f32[], f32[]) tuple(%recv1, %recv2)
}
  )";
TEST_F(CombineInstructionsTest, TestCombineRecvFromHost) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(recv_from_host_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 8;

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_send_recv_cluster_size(
          max_send_recv_cluster_size));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::RecvFromHost, 1));

  auto recv_inst = Cast<HloRecvFromHostInstruction>(
      *absl::c_find_if(seq, IsPoplarInstruction(RecvFromHost)));

  // Check that it was merged correctly.
  EXPECT_EQ(recv_inst->operand_count(), 0);
  EXPECT_EQ(recv_inst->RendezvousKeys().size(), 2);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kGetTupleElement);

  const int64 ret0_index =
      Cast<HloGetTupleElementInstruction>(root->operand(0))->tuple_index();
  const int64 ret1_index =
      Cast<HloGetTupleElementInstruction>(root->operand(1))->tuple_index();

  EXPECT_EQ(recv_inst->RendezvousKeys()[ret0_index], "recv1_key");
  EXPECT_EQ(recv_inst->RendezvousKeys()[ret1_index], "recv2_key");
}

TEST_F(CombineInstructionsTest, TestCombineSendRecvFromHostInplace) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[2]) -> (f32[], f32[2]) {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2 = f32[2] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send1_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  %send2 = () custom-call(f32[2] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"send2_key\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  %recv1 = f32[] custom-call(f32[] %arg1), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv1_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  %recv2 = f32[2] custom-call(f32[2] %arg2), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv2_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}, sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[2]) tuple(%recv1, %recv2)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  CompilerAnnotations annotations(module);
  EXPECT_TRUE(InplaceFinder(annotations).Run(module).ValueOrDie());
  EXPECT_TRUE(HostComputeBarrierInserter().Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 12;

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_send_recv_cluster_size(
          max_send_recv_cluster_size));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  EXPECT_THAT(seq, ContainsNPoplarOps(PoplarOp::SendToHost, 1));
  EXPECT_THAT(seq, ContainsNPoplarOps(PoplarOp::RecvFromHost, 1));

  auto send_inst = Cast<HloSendToHostInstruction>(
      *absl::c_find_if(seq, IsPoplarInstruction(SendToHost)));

  auto recv_inst = Cast<HloRecvFromHostInstruction>(
      *absl::c_find_if(seq, IsPoplarInstruction(RecvFromHost)));

  auto* barrier_inst = module->entry_computation()->GetInstructionWithName(
      "host_compute.barrier");
  ASSERT_NE(barrier_inst, nullptr);

  // Check that they were merged.
  EXPECT_EQ(send_inst->operand_count(), 2);
  EXPECT_EQ(send_inst->RendezvousKeys().size(), 2);
  EXPECT_EQ(recv_inst->operand_count(), 2);
  EXPECT_EQ(recv_inst->RendezvousKeys().size(), 2);

  // Check that the shapes match up.
  EXPECT_EQ(recv_inst->operand(0)->shape(), recv_inst->shape().tuple_shapes(0));
  EXPECT_EQ(recv_inst->operand(1)->shape(), recv_inst->shape().tuple_shapes(1));

  // Check that in-placeness and sharding was kept.
  EXPECT_TRUE(IsLoweredInplace(recv_inst));
  EXPECT_TRUE(send_inst->has_sharding());
  EXPECT_TRUE(recv_inst->has_sharding());
  EXPECT_EQ(recv_inst->sharding_unique_device().value(), 1);
  EXPECT_EQ(send_inst->sharding_unique_device().value(), 1);

  // Check that control dependencies wrt. barrier are kept.
  EXPECT_EQ(barrier_inst->control_predecessors().size(), 1);
  EXPECT_EQ(barrier_inst->control_predecessors()[0], send_inst);
  EXPECT_EQ(barrier_inst->control_successors().size(), 1);
  EXPECT_EQ(barrier_inst->control_successors()[0], recv_inst);
}

const char* reduce_hlo = R"(
HloModule top

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0, p1)
}

multiply {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT multiply.1 = f32[] multiply(p0, p1)
}

cluster_1  {
  arg0 = f32[512,8] parameter(0)
  arg1 = f32[512,8] parameter(1)
  arg2 = f32[512,8] parameter(2)
  c0 = f32[] constant(0)
  c1 = f32[] constant(0)
  c2 = f32[] constant(1)
  r0 = f32[512] reduce(arg0, c0), dimensions={1}, to_apply=add
  r1 = f32[512] reduce(arg1, c1), dimensions={1}, to_apply=multiply
  r2 = f32[512] reduce(arg2, c2), dimensions={1}, to_apply=add
  ROOT %tuple = (f32[512], f32[512], f32[512]) tuple(r0, r1, r2)
}
  )";
TEST_F(CombineInstructionsTest, TestCombineReductionsIntoReduceMany) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(reduce_hlo, config_));

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  ASSERT_EQ(absl::c_count_if(entry->instructions(), IsReduceAddOrMultiply), 3);

  uint64 node_size = 4 * 512;  // float32 * 512.
  // Schedule and combine.
  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_reduce_many_buffer_size(2 * node_size));
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  auto s = module.get()->schedule().sequence(entry);
  auto seq = s.instructions();

  // Two reduces should be combined into a reduce many, while one is left
  // unmodified.
  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::ReduceMany, 1));
  ASSERT_EQ(absl::c_count_if(seq, IsReduceAddOrMultiply), 1);
}

const char* reduction_fusion_hlo = R"(
HloModule top

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0, p1)
}

multiply {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT multiply.1 = f32[] multiply(p0, p1)
}

_pop_op_reduction_fp16_input {
  p0 = f16[512,8] parameter(0)
  p1 = f32[] parameter(1)
  convert = f32[512,8] convert(p0)
  ROOT reduce = f32[512] reduce(convert, p1), dimensions={1}, to_apply=multiply
}

_pop_op_reduction_square_add {
  p0 = f16[512,8] parameter(0)
  p1 = f32[] parameter(1)
  multiply = f16[512,8] multiply(p0, p0)
  convert = f32[512,8] convert(multiply)
  ROOT reduce = f32[512] reduce(convert, p1), dimensions={1}, to_apply=add
}

cluster_1  {
  arg0 = f16[512,8] parameter(0)
  arg1 = f16[512,8] parameter(1)
  c0 = f32[] constant(0)
  c1 = f32[] constant(1)
  r0 = f32[512] fusion(arg0, c0), kind=kCustom, calls=_pop_op_reduction_square_add
  r1 = f32[512] fusion(arg1, c1), kind=kCustom, calls=_pop_op_reduction_fp16_input
  ROOT %tuple = (f32[512], f32[512]) tuple(r0, r1)
}
  )";
TEST_F(CombineInstructionsTest, TestCombineReduceFusionsIntoReduceMany) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(reduction_fusion_hlo, config_));

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  ASSERT_EQ(absl::c_count_if(entry->instructions(), IsReductionFusion), 2);

  uint64 node_size = 4 * 512;  // float32 * 512.
  // Schedule and combine.
  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_reduce_many_buffer_size(2 * node_size));
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  auto s = module.get()->schedule().sequence(entry);
  auto seq = s.instructions();

  // Both reduce fusions should be combined into a ReduceMany.
  ASSERT_THAT(seq, ContainsNPoplarOps(PoplarOp::ReduceMany, 1));
  ASSERT_EQ(absl::c_count_if(seq, IsReductionFusion), 0);
}

TEST_F(CombineInstructionsTest, TestOutputOfCombinedReducesIntoReduceMany) {
  std::string hlo_string = R"(
HloModule top

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0, p1)
}

multiply {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT multiply.1 = f32[] multiply(p0, p1)
}

_pop_op_reduction_fp16_input {
  p0 = f16[16,8] parameter(0)
  p1 = f32[] parameter(1)
  convert = f32[16,8] convert(p0)
  ROOT reduce = f32[16] reduce(convert, p1), dimensions={1}, to_apply=multiply
}

_pop_op_reduction_square_add {
  p0 = f16[16,8] parameter(0)
  p1 = f32[] parameter(1)
  multiply = f16[16,8] multiply(p0, p0)
  convert = f32[16,8] convert(multiply)
  ROOT reduce = f32[16] reduce(convert, p1), dimensions={1}, to_apply=add
}

ENTRY cluster_1  {
  arg0 = f16[] constant(2)
  arg1 = f32[] constant(2)
  b0 = f16[16,8] broadcast(arg0), dimensions={}
  b1 = f32[16,8] broadcast(arg1), dimensions={}
  c0 = f32[] constant(5)
  c1 = f32[] constant(3)
  c2 = f32[] constant(0)
  r0 = f32[16] fusion(b0, c0), kind=kCustom, calls=_pop_op_reduction_square_add
  r1 = f32[16] fusion(b0, c1), kind=kCustom, calls=_pop_op_reduction_fp16_input
  r2 = f32[16] reduce(b1, c2), dimensions={1}, to_apply=add
  ROOT %tuple = (f32[16], f32[16], f32[16]) tuple(r0, r1, r2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config_));
  using ::testing::Each;

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  uint64 node_size = 4 * 32;  // float32 * 32.
  // Schedule and combine.
  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_reduce_many_buffer_size(4 * node_size));
  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  // Prepare to execute graph.
  auto device = CreateIpuModel(1, 32);
  auto resources = GetMockResources(device, module.get());

  auto order = module->schedule().sequence(entry).instructions();
  EntryVisitor visitor(*resources, entry);
  TF_CHECK_OK(entry->AcceptOrdered(&visitor, order));

  poplar::program::Sequence main_program;
  main_program.add(resources->preamble_sequence);
  main_program.add(visitor.GetSequenceAndInitializeCounters());

  poplar::Engine engine(*resources->main_graph, main_program);

  // Connect i/o
  auto& io_map = resources->annotations.input_output_aliasing_map;
  auto& outputs = io_map.GetEntryOutputInfos();
  EXPECT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[0].Handles().size(), 1);
  EXPECT_EQ(outputs[1].Handles().size(), 1);
  EXPECT_EQ(outputs[2].Handles().size(), 1);

  std::array<float, 16> out0, out1, out2;
  engine.connectStream(outputs[0].Handles()[0], out0.data(),
                       out0.data() + out0.size());
  engine.connectStream(outputs[1].Handles()[0], out1.data(),
                       out1.data() + out1.size());
  engine.connectStream(outputs[2].Handles()[0], out2.data(),
                       out2.data() + out2.size());

  // Run the program.
  device.attach();
  engine.load(device);
  engine.run(0);
  device.detach();

  // Check outputs
  ASSERT_THAT(out0, Each(37));   // 5 + ((2^2) * 8)
  ASSERT_THAT(out1, Each(768));  // 3 * (2 ^ 8);
  ASSERT_THAT(out2, Each(16));   // 0 + (2 * 8)
}

TEST_F(CombineInstructionsTest, TestInplace) {
  auto module_or_status =
      ParseAndReturnVerifiedModule(gradient_accumulation_hlo, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  // Run the inplacer.
  InplaceFinder inplace_finder{annotations};
  EXPECT_TRUE(inplace_finder.Run(module).ValueOrDie());

  // Expect the gradient accumulations to be inplace.
  auto inplace_instructions = GetInplaceInstructions(module);
  ASSERT_THAT(
      inplace_instructions,
      ContainsNPoplarOps(PoplarOp::StatefulGradientAccumulateAndAllReduce, 3));

  // Make one of the gradient accumulations not inplace.
  auto root = entry->root_instruction();
  auto norm1 = root->mutable_operand(1);
  auto ga_and_ar = norm1->mutable_operand(0);
  MakeUsedNotInplace(ga_and_ar);

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 11);
  // Expect two gradient accumulation instructions.
  ASSERT_THAT(seq, ContainsNPoplarOps(
                       PoplarOp::StatefulGradientAccumulateAndAllReduce, 2));
}

TEST_F(CombineInstructionsTest, TestDataDependency) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(a2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 2);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 2);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 7);
  ASSERT_THAT(seq, ContainsNOps(HloOpcode::kAllReduce, 2));
}

TEST_F(CombineInstructionsTest, TestControlDependency) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add, control-predecessors={a1}
  %a3 = f16[4] all-reduce(a2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());
}

TEST_F(CombineInstructionsTest, TestControlDependency2) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add, control-predecessors={a1}
  %a3 = f16[4] all-reduce(arg2), to_apply=add, control-predecessors={a1}
  %arg3 = f16[4] parameter(3), control-predecessors={a1, a2, a3}
  %a4 = f16[4] all-reduce(arg3), to_apply=add, control-predecessors={a2}
  %a5 = f16[4] all-reduce(a2), to_apply=add, control-predecessors={a4}
  %a6 = f16[4] all-reduce(a1), to_apply=add, control-predecessors={arg3}
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a4, f16[4] %a5, f16[4] %a6)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 4);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 4);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 13);
  ASSERT_THAT(seq, ContainsNOps(HloOpcode::kAllReduce, 4));
}

TEST_F(CombineInstructionsTest, TestMultipleTypes) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f32[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f32[4] parameter(2)
  %arg3 = f16[4] parameter(3)
  %arg4 = f32[4] parameter(4)
  %arg5 = f16[4] parameter(5)


  %a1 = f32[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f32[4] all-reduce(arg2), to_apply=add
  %a4 = f16[4] all-reduce(arg3), to_apply=add
  %a5 = f32[4] all-reduce(arg4), to_apply=add
  %a6 = f16[4] all-reduce(arg5), to_apply=add
  ROOT %tuple = (f32[4], f16[4], f32[4], f16[4], f32[4], f16[4]) tuple(f32[4] %a1, f16[4] %a2, f32[4] %a3, f16[4] %a4, f32[4] %a5, f16[4] %a6)
}
  )";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config_);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  auto scheduler = CreateLookAheadScheduler(
      CompilerInformation().set_max_all_reduce_buffer_size(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  // 6 Arguments + 2 all reduces + 2*3 GTE (one for each element of the fused
  // all reduce) + 1 output tuple. = 15 instructions.
  ASSERT_EQ(seq.size(), 15);

  // All of the fp16 kernels should be fused in one and all of the floating
  // point 32 in another, so there should only be two.
  ASSERT_THAT(seq, ContainsNOps(HloOpcode::kAllReduce, 2));
}

const char* all_gather_hlo = R"hlo(
HloModule top

entry {
  arg0 = f32[10] parameter(0)
  arg1 = f32[10] parameter(1)
  arg2 = f32[10] parameter(2)
  r0 = f32[4,10] custom-call(arg0), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":0}"
  r1 = f32[4,10] custom-call(arg1), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":0}"
  r2 = f32[4,10] custom-call(arg2), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":0}"
  r3 = f32[2,10] custom-call(arg0), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":2}"
  r4 = f32[2,10] custom-call(arg1), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":2}"
  r5 = f32[2,10] custom-call(arg2), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":2}"
  r6 = f32[4,10] custom-call(arg0), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":4}"
  r7 = f32[4,10] custom-call(arg1), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":4}"
  r8 = f32[4,10] custom-call(arg2), custom_call_target="AllGather",
    backend_config="{\"replica_group_size\":4}"
  ROOT %tuple = (f32[4,10], f32[4,10], f32[4,10], f32[2,10], f32[2,10], f32[2,10], f32[4,10],
    f32[4,10], f32[4,10]) tuple(r0, r1, r2, r3, r4, r5, r6, r7, r8)
}
  )hlo";
class CombineInstructionsAllGatherTest : public CombineInstructionsTest {
  void SetUp() override {
    config_.set_replica_count(_replication_factor);

    TF_ASSERT_OK_AND_ASSIGN(
        _module, ParseAndReturnVerifiedModule(all_gather_hlo, config_));
    EXPECT_TRUE(CustomOpReplacer().Run(_module.get()).ValueOrDie());
  }

 protected:
  const int _replication_factor = 4;
  const int _num_output_tuple_elements = 9;
  const int _num_replicas = _replication_factor > 0 ? _replication_factor : 1;
  const int _output_size = _num_replicas * 4 * 10;

  std::unique_ptr<VerifiedHloModule> _module;

  void ScheduleInstructions() {
    uint64 node_size = 4 * 512;  // float32 * 512.
    // Scheduling is required to run the combiner.
    auto scheduler = CreateLookAheadScheduler(
        CompilerInformation().set_max_all_gather_buffer_size(
            2 * node_size));  // 2 nodes out of the 3 in the graph.
    EXPECT_TRUE(scheduler.Run(_module.get()).ValueOrDie());
  }
};

TEST_F(CombineInstructionsAllGatherTest, TestCombineAllGather) {
  auto* entry = _module->entry_computation();

  // Ensure we got 9 AllGather instructions.
  ASSERT_THAT(entry->instructions(),
              ContainsNPoplarOps(PoplarOp::AllGather, 9));

  ScheduleInstructions();
  EXPECT_TRUE(CombineInstructions().Run(_module.get()).ValueOrDie());

  // Ensure we have the nine GTE instructions inplace.
  auto inplace_instructions = GetInplaceInstructions(_module.get());
  EXPECT_EQ(inplace_instructions.size(), 9);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_LT(inplace_inst->tuple_index(), 3);
  }

  // Three groups of three gathers should be each combined into a single
  // PoplarAllGather.
  auto instructions = _module->schedule().sequence(entry).instructions();
  ASSERT_THAT(instructions, ContainsNPoplarOps(PoplarOp::AllGather, 3));
}

TEST_F(CombineInstructionsAllGatherTest, TestCombineAllGatherOutputsIdentical) {
  // Ensure this test runs on hardware.
  TF_ASSERT_OK_AND_ASSIGN(auto tf_ipu_count, GetMaxIpuCount());
  if (_replication_factor > tf_ipu_count) {
    GTEST_SKIP() << "Skipping test, replication factor " << _replication_factor
                 << ", max ipu: " << tf_ipu_count;
    return;
  }

  // Acquire our device.
  TF_ASSERT_OK_AND_ASSIGN(poplar::Device device,
                          CreateIpuDevice(_replication_factor));

  ScheduleInstructions();

  // Clone the scheduled original module before we combine the instructions so
  // we can compare the module with instructions combined and the module
  // without.
  auto original_module = _module->Clone();

  EXPECT_TRUE(CombineInstructions().Run(_module.get()).ValueOrDie());

  auto run_module = [&](HloModule* module) -> std::vector<std::vector<float>> {
    auto resources = GetMockResources(device, module, _replication_factor);
    auto engine = Compile(*resources, module).ConsumeValueOrDie();

    // Check that only one computations have been compiled – entry.
    CHECK_EQ(resources->tensor_maps.size(), 1);

    // Load program onto device.
    engine.load(device);

    std::vector<float> inputs = {0.f, 1.f, 2.f, 3.f, 4.f,
                                 5.f, 6.f, 7.f, 8.f, 9.f};
    engine.connectStream("0.0", inputs.data(), inputs.data() + inputs.size());
    engine.connectStream("1.0", inputs.data(), inputs.data() + inputs.size());
    engine.connectStream("2.0", inputs.data(), inputs.data() + inputs.size());

    // Get the values back.
    std::vector<std::vector<float>> outputs;
    for (int64 i = 0; i < _num_output_tuple_elements; ++i) {
      outputs.emplace_back(_output_size);
      std::stringstream label;
      label << "out_" << i << ".0";
      engine.connectStream(label.str(), outputs[i].data(),
                           outputs[i].data() + outputs[i].size());
    }

    // Run the program.
    engine.run(0);

    return outputs;
  };

  auto new_module_ptr = static_cast<HloModule*>(_module.get());
  auto outputs_new = run_module(new_module_ptr);
  auto outputs_original = run_module(original_module.get());

  ASSERT_EQ(outputs_original.size(), outputs_new.size());
  for (int i = 0; i < outputs_original.size(); i++) {
    for (int j = 0; j < outputs_original[i].size(); j++) {
      ASSERT_EQ(outputs_original[i][j], outputs_new[i][j]);
    }
  }
}

struct ShardingPropogationTest : CombineInstructionsTest,
                                 ::testing::WithParamInterface<HloTestCase> {
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        module_, ParseAndReturnVerifiedModule(GetParam().hlo, config_));

    CustomOpReplacer custom_op_replacer;
    custom_op_replacer.Run(module_.get());

    CompilerAnnotations annotations(module_.get());
    GradientAccumulationFuser fuser(annotations);
    fuser.Run(module_.get());

    information_.set_max_reduce_scatter_buffer_size(60 * 1024);
    information_.set_max_all_reduce_buffer_size(64 * 1024);
    information_.set_max_send_recv_cluster_size(8);
    information_.set_max_reduce_many_buffer_size(8 * 1024);
  }

  std::unique_ptr<VerifiedHloModule> module_;
  CompilerInformation information_;
};

TEST_P(ShardingPropogationTest, IsSharded) {
  // This test checks that the CombineInstructions pass adds sharding
  // information to the instructions it creates. We set a particular sharding on
  // every instruction and expect new instructions to have that same sharding.

  const auto expected_sharding = HloSharding::AssignDevice(0);

  auto* comp = module_->entry_computation();
  for (auto* inst : comp->instructions()) {
    inst->set_sharding(expected_sharding);
  }

  auto scheduler = CreateLookAheadScheduler(information_);
  TF_ASSERT_OK_AND_ASSIGN(bool scheduled, scheduler.Run(module_.get()));
  ASSERT_TRUE(scheduled);

  CombineInstructions combine_instructions;
  TF_ASSERT_OK_AND_ASSIGN(bool combined,
                          combine_instructions.Run(module_.get()));
  ASSERT_TRUE(combined);

  ASSERT_THAT(comp->instructions(), AllSharded(expected_sharding));
}

INSTANTIATE_TEST_SUITE_P(
    CombineInstructionsHLO, ShardingPropogationTest,
    ::testing::Values(
        MAKE_HLO_TEST_CASE(all_reduce_hlo),
        MAKE_HLO_TEST_CASE(gradient_accumulation_hlo),
        MAKE_HLO_TEST_CASE(gradient_accumulation_with_momentum_hlo),
        MAKE_HLO_TEST_CASE(reduce_scatter_hlo),
        MAKE_HLO_TEST_CASE(send_to_host_hlo),
        MAKE_HLO_TEST_CASE(recv_from_host_hlo), MAKE_HLO_TEST_CASE(reduce_hlo),
        MAKE_HLO_TEST_CASE(reduction_fusion_hlo),
        MAKE_HLO_TEST_CASE(all_gather_hlo)),
    HloTestCaseName);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
