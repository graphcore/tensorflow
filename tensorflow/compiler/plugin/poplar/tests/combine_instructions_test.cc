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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recv_from_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/send_to_host.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_test_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using CombineInstructionsTest = HloPoplarTestBase;

TEST_F(CombineInstructionsTest, TestSyncScheduler) {
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
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(IpuToMemorySchedulerAlgorithm(
          CreateSyncListMemoryScheduler(64 * 1024))));
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

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 1);
}

TEST_F(CombineInstructionsTest, TestLookAheadScheduler) {
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
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
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

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 1);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
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
  ASSERT_EQ(absl::c_count_if(body->instructions(),
                             IsPoplarInstruction(PoplarOp::IpuInterCopy)),
            3);

  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_inter_ipu_copies_buffer_size(
                  64 * 1024)))));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());
  // Two IPU copies have been merged.
  EXPECT_EQ(absl::c_count_if(body->instructions(),
                             IsPoplarInstruction(PoplarOp::IpuInterCopy)),
            2);
  EXPECT_EQ(body->instruction_count(), 16);
}

TEST_F(CombineInstructionsTest, TestLookAheadSchedulerGradientAccumulation) {
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
  %norm1 = f16[4] custom-call(a1), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga1 = f16[4] custom-call(norm1), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %arg2 = f16[4] parameter(2)
  %a2 = f16[4] all-reduce(arg2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga2 = f16[4] custom-call(norm2), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(ga0, ga1, ga2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
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

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
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
  ASSERT_EQ(absl::c_count_if(
                seq, IsPoplarInstruction(
                         PoplarOp::StatefulGradientAccumulateAndAllReduce)),
            1);
}

TEST_F(CombineInstructionsTest,
       TestLookAheadSchedulerGradientAccumulationWithMomentum) {
  std::string hlo_string = R"(
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
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

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are set.
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 9);

  auto s = module->schedule().sequence(entry);
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 24);
  ASSERT_EQ(
      absl::c_count_if(
          seq,
          IsPoplarInstruction(
              PoplarOp::
                  StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm)),
      1);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
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

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);
}

TEST_F(CombineInstructionsTest, TestCombineReduceScatter) {
  std::string hlo_string = R"(
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  ASSERT_EQ(absl::c_count_if(entry->instructions(),
                             IsPoplarInstruction(PoplarOp::ReduceScatter)),
            4);

  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_reduce_scatter_buffer_size(
                  64 * 1024)))));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  // There should be a three reduce scatters:
  // - One combined with with op=ADD replica_group_size=2
  // - One left alone with with op=ADD replica_group_size=4
  // - One left alone with with op=LOCAL replica_group_size=4
  ASSERT_EQ(absl::c_count_if(seq, IsPoplarInstruction(PoplarOp::ReduceScatter)),
            3);
}

TEST_F(CombineInstructionsTest, TestCombineSendToHost) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[]) -> () {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2 = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg1\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %send2 = () custom-call(f32[] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg2\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %ret = () tuple()
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 8;

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_send_recv_cluster_size(
                  max_send_recv_cluster_size)))));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  EXPECT_EQ(absl::c_count_if(seq, IsPoplarInstruction(SendToHost)), 1);

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
  std::string hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[]) -> () {
  %arg1 = f32[] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send1 = () custom-call(f32[] %arg1), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg1\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %arg2 = f32[] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %send2 = () custom-call(f32[] %arg2), custom_call_target="SendToHost", backend_config="{\"rendezvous_key\":\"arg2\"}", custom_call_has_side_effect=true, metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %tuple = () tuple()
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 4;

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_send_recv_cluster_size(
                  max_send_recv_cluster_size)))));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  EXPECT_EQ(absl::c_count_if(seq, IsPoplarInstruction(SendToHost)), 2);
}

TEST_F(CombineInstructionsTest, TestCombineRecvFromHost) {
  std::string hlo_string = R"(
HloModule top

ENTRY %top () -> (f32[], f32[]) {
  %recv1 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv1_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  %recv2 = f32[] custom-call(), custom_call_target="RecvFromHost", backend_config="{\"rendezvous_key\":\"recv2_key\"}", metadata={op_type="XlaHostCompute" op_name="host_compute"}
  ROOT %tuple = (f32[], f32[]) tuple(%recv1, %recv2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 8;

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_send_recv_cluster_size(
                  max_send_recv_cluster_size)))));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  EXPECT_EQ(absl::c_count_if(seq, IsPoplarInstruction(RecvFromHost)), 1);

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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());
  EXPECT_TRUE(InplaceFinder().Run(module).ValueOrDie());
  EXPECT_TRUE(HostComputeBarrierInserter().Run(module).ValueOrDie());

  const int64 max_send_recv_cluster_size = 12;

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_send_recv_cluster_size(
                  max_send_recv_cluster_size)))));
  ASSERT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  ASSERT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  EXPECT_EQ(absl::c_count_if(seq, IsPoplarInstruction(SendToHost)), 1);
  EXPECT_EQ(absl::c_count_if(seq, IsPoplarInstruction(RecvFromHost)), 1);

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

TEST_F(CombineInstructionsTest, TestCombineReductionsIntoReduceMany) {
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  ASSERT_EQ(absl::c_count_if(entry->instructions(), IsReduceAddOrMultiply), 3);

  uint64 node_size = 4 * 512;  // float32 * 512.
  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_reduce_many_buffer_size(
                  2 * node_size)))));  // 2 nodes out of the 3 in the graph.

  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  auto s = module.get()->schedule().sequence(entry);
  auto seq = s.instructions();

  // Two reduces should be combined into a reduce many, while one is left
  // unmodified.
  ASSERT_EQ(absl::c_count_if(seq, IsPoplarInstruction(PoplarOp::ReduceMany)),
            1);
  ASSERT_EQ(absl::c_count_if(seq, IsReduceAddOrMultiply), 1);
}

TEST_F(CombineInstructionsTest, TestCombineReduceFusionsIntoReduceMany) {
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  ASSERT_EQ(absl::c_count_if(entry->instructions(), IsReductionFusion), 2);

  uint64 node_size = 4 * 512;  // float32 * 512.
  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_reduce_many_buffer_size(
                  2 * node_size)))));  // Sufficient for both reduces.

  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  auto s = module.get()->schedule().sequence(entry);
  auto seq = s.instructions();

  // Both reduce fusions should be combined into a ReduceMany.
  ASSERT_EQ(absl::c_count_if(seq, IsPoplarInstruction(PoplarOp::ReduceMany)),
            1);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CompilerAnnotations annotations(module.get());
  auto* entry = module.get()->entry_computation();

  uint64 node_size = 4 * 32;  // float32 * 32.
  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_reduce_many_buffer_size(
                  4 * node_size)))));  // Sufficient for all reduces.

  EXPECT_TRUE(scheduler.Run(module.get()).ValueOrDie());
  EXPECT_TRUE(CombineInstructions().Run(module.get()).ValueOrDie());

  // Prepare to execute graph.
  TF_ASSERT_OK_AND_ASSIGN(auto device, CreateIpuModel(1, 32));
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
  for (auto& n : out0) {
    ASSERT_EQ(n, 37);  // 5 + ((2^2) * 8)
  }
  for (auto& n : out1) {
    ASSERT_EQ(n, 768);  // 3 * (2 ^ 8)
  }
  for (auto& n : out2) {
    ASSERT_EQ(n, 16);  // 0 + (2 * 8)
  }
}

TEST_F(CombineInstructionsTest, TestInplace) {
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
  %norm1 = f16[4] custom-call(a1), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga1 = f16[4] custom-call(norm1), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %arg2 = f16[4] parameter(2)
  %a2 = f16[4] all-reduce(arg2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="ReplicationNormalise", backend_config="{}\n"
  %ga2 = f16[4] custom-call(norm2), custom_call_target="StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(ga0, ga1, ga2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
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
  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module).ValueOrDie());

  // Expect the gradient accumulations to be inplace.
  auto inplace_instructions = GetInplaceInstructions(module);
  ASSERT_EQ(
      absl::c_count_if(inplace_instructions,
                       IsPoplarInstruction(
                           PoplarOp::StatefulGradientAccumulateAndAllReduce)),
      3);

  // Make one of the gradient accumulations not inplace.
  auto root = entry->root_instruction();
  auto norm1 = root->mutable_operand(1);
  auto ga_and_ar = norm1->mutable_operand(0);
  MakeUsedNotInplace(ga_and_ar);

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 11);
  // Expect two gradient accumulation instructions.
  ASSERT_EQ(absl::c_count_if(
                seq, IsPoplarInstruction(
                         PoplarOp::StatefulGradientAccumulateAndAllReduce)),
            2);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
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

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 2);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
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

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 4);
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

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      ComputationSchedulerToModuleScheduler(
          IpuToMemorySchedulerAlgorithm(CreateClusteringMemoryScheduler(
              CompilerInformation().set_max_all_reduce_buffer_size(64 *
                                                                   1024)))));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();

  // 6 Arguments + 2 all reduces + 2*3 GTE (one for each element of the fused
  // all reduce) + 1 output tuple. = 15 instructions.
  ASSERT_EQ(seq.size(), 15);

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };

  // All of the fp16 kernels should be fused in one and all of the floating
  // point 32 in another, so there should only be two.
  ASSERT_EQ(absl::c_count_if(seq, pred), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
