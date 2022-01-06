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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/plugin/poplar/driver/passes/all_to_all_finder.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloReplaceEmbeddingAllRed = HloTestBase;

TEST_F(HloReplaceEmbeddingAllRed, F32AllReduce) {
  std::string hlo = R"(
HloModule top

%to_apply_func (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0), backend_config="{}"
  %y = f32[] parameter(1), backend_config="{}"
  ROOT %add = f32[] add(f32[] %x, f32[] %y), backend_config="{\"isInplace\":true}"
}

ENTRY c1 {
  %updates = f32[1000, 200] parameter(0)
  %indices = s32[1000] parameter(1)
  %embeddings = f32[100000, 200] parameter(2)
  %zero_tensor = f32[100000, 200] parameter(3)
  %scale = f32[] parameter(4)
  %learning_rate = f32[] parameter(5)
  %broadcast = f32[100000, 200]{1,0} broadcast(f32[] %learning_rate), dimensions={}, backend_config="{}"


  %multi_update_add = f32[100000,200]{1,0} custom-call(f32[100000, 200]{1,0} %zero_tensor,
       s32[1000]{0} %indices, f32[1000, 200]{1,0} %updates,
       f32[] %scale), custom_call_target="MultiUpdateAdd", sharding={maximal device=1}, backend_config="{\"isInplace\":true,\"hashOfCustomAttributes\":\"-7046029254386353152\"}"

  %all-reduce = f32[100000,200]{1,0} all-reduce(f32[100000,200]{1,0} %multi_update_add),to_apply=%to_apply_func, replica_groups={}, sharding={maximal device=1}, backend_config="{}"

  %normalize = f32[100000,200]{1,0} custom-call(f32[100000,200]{1,0} %all-reduce), custom_call_target="ReplicationNormalise", sharding={maximal device=1}, backend_config="{\"isInplace\":true}"

  %multiply = f32[100000,200]{1,0} multiply(f32[100000,200]{1,0} %broadcast, f32[100000,200]{1,0} %normalize), sharding={maximal device=1}, backend_config="{}"
  ROOT %subtract = f32[100000,200]{1,0} subtract(f32[100000,200]{1,0} %embeddings, f32[100000,200]{1,0} %multiply), sharding={maximal device=1},backend_config="{\"isInplace\":true}"
}

)";
  /*
  Expected output.

  ENTRY %c1 (updates: f32[1000,200], indices: s32[1000], embeddings:
  f32[100000,200], zero_tensor: f32[100000,200], scale: f32[], learning_rate:
  f32[]) -> f32[100000,200] { %zero_tensor = f32[100000,200]{1,0} parameter(3)
    %scale = f32[] parameter(4)
    %embeddings = f32[100000,200]{1,0} parameter(2)
    %indices = s32[1000]{0} parameter(1)
    %custom-call.2 = s32[4,1000]{1,0} custom-call(s32[1000]{0} %indices),
  custom_call_target="AllGather", backend_config="{}" %reshape = s32[4000]{0}
  reshape(s32[4,1000]{1,0} %custom-call.2) %updates = f32[1000,200]{1,0}
  parameter(0) %custom-call.3 = f32[4,1000,200]{2,1,0}
  custom-call(f32[1000,200]{1,0} %updates), custom_call_target="AllGather",
  backend_config="{}" %reshape.1 = f32[4000,200]{1,0}
  reshape(f32[4,1000,200]{2,1,0} %custom-call.3) %custom-call.4 =
  f32[4000,200]{1,0} custom-call(f32[4000,200]{1,0} %reshape.1),
  custom_call_target="ReplicationNormalise", backend_config="{}" %learning_rate
  = f32[] parameter(5) %broadcast.1 = f32[4000,200]{1,0} (broadcast(f32[]
  %learning_rate), dimensions={} %multiply.1 = f32[4000,200]{1,0}
  multiply(f32[4000,200]{1,0} %custom-call.4, f32[4000,200]{1,0} %broadcast.1)
    %constant = f32[] constant(-1)
    ROOT %custom-call.5 = f32[100000,200]{1,0} custom-call(f32[100000,200]{1,0}
  %embeddings, s32[4000]{0} %reshape, f32[4000,200]{1,0} %multiply.1, f32[]
  %constant), custom_call_target="MultiUpdateAdd",
  update_dim=1, sharding={maximal device=1},
  backend_config="{\"hashOfCustomAttributes\":\"-7046029254386353152\"}"
  }
  */

  auto module = ParseAndReturnVerifiedModule(hlo);

  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);

  HloComputation* computation = hlo_module->entry_computation();

  const HloInstruction* updates = computation->parameter_instruction(0);
  const HloInstruction* indices = computation->parameter_instruction(1);
  const HloInstruction* embeddings = computation->parameter_instruction(2);
  const HloInstruction* learning_rate = computation->parameter_instruction(5);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(hlo_module).ValueOrDie());

  AllToAllFinder matcher(annotations, 4);
  EXPECT_TRUE(matcher.Run(hlo_module).ValueOrDie());

  // We expect the root to be now a multi update add.
  HloInstruction* inst = computation->root_instruction();

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst));

  const HloInstruction* expected_embeddings =
      inst->operand(0);  // Expected to be embeddings.
  const HloInstruction* expected_indices =
      inst->operand(1);  // Expected to be gathered indices.
  const HloInstruction* expected_updates =
      inst->operand(2);  // Expected to be gathered updates.
  const HloInstruction* expected_scale =
      inst->operand(3);  // Expected to be the scale.

  // Check the embedding.
  EXPECT_EQ(expected_embeddings, embeddings);

  // Check the indices.
  EXPECT_EQ(expected_indices->shape().dimensions_size(), 1);
  EXPECT_EQ(expected_indices->shape().dimensions()[0],
            4000);  // 4000 = 4 replicas * 1000

  const HloReshapeInstruction* reshape_ind =
      dynamic_cast<const HloReshapeInstruction*>(expected_indices);

  EXPECT_NE(reshape_ind, nullptr);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(reshape_ind->operand(0)));
  EXPECT_EQ(reshape_ind->operand(0)->operand(0), indices);

  // Check the scale.
  const HloConstantInstruction* scale_as_const =
      dynamic_cast<const HloConstantInstruction*>(expected_scale);
  EXPECT_NE(scale_as_const, nullptr);

  EXPECT_TRUE(scale_as_const->literal().IsAllFloat(-1.0f));

  // Check update.
  EXPECT_EQ(expected_updates->opcode(), HloOpcode::kMultiply);

  const HloBroadcastInstruction* expected_broadcast =
      dynamic_cast<const HloBroadcastInstruction*>(
          expected_updates->operand(1));
  EXPECT_NE(expected_broadcast, nullptr);
  EXPECT_EQ(learning_rate, expected_broadcast->operand(0));

  const HloInstruction* expected_replication_normalize =
      expected_updates->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReplicationNormalise)(
      expected_replication_normalize));

  const HloReshapeInstruction* reshape_up =
      dynamic_cast<const HloReshapeInstruction*>(
          expected_replication_normalize->operand(0));

  EXPECT_NE(reshape_up, nullptr);

  const HloInstruction* expected_update_gather = reshape_up->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(expected_update_gather));
  EXPECT_EQ(expected_update_gather->operand(0), updates);
}

TEST_F(HloReplaceEmbeddingAllRed, F16AllReduce) {
  std::string hlo = R"(
HloModule top

%to_apply_func (x: f16[], y: f16[]) -> f16[] {
  %x = f16[] parameter(0), backend_config="{}"
  %y = f16[] parameter(1), backend_config="{}"
  ROOT %add = f16[] add(f16[] %x, f16[] %y), backend_config="{\"isInplace\":true}"
}

ENTRY c1 {
  %updates = f16[1000, 200] parameter(0)
  %indices = s32[1000] parameter(1)
  %embeddings = f16[100000, 200] parameter(2)
  %zero_tensor = f16[100000, 200] parameter(3)
  %scale = f16[] parameter(4)
  %learning_rate = f16[] parameter(5)
  %broadcast = f16[100000, 200]{1,0} broadcast(f16[] %learning_rate), dimensions={}, backend_config="{}"


  %multi_update_add = f16[100000,200]{1,0} custom-call(f16[100000, 200]{1,0} %zero_tensor,
       s32[1000]{0} %indices, f16[1000, 200]{1,0} %updates,
       f16[] %scale), custom_call_target="MultiUpdateAdd", sharding={maximal device=1}, backend_config="{\"isInplace\":true,\"hashOfCustomAttributes\":\"-7046029254386353152\"}"

  %all-reduce = f16[100000,200]{1,0} all-reduce(f16[100000,200]{1,0} %multi_update_add),to_apply=%to_apply_func, replica_groups={}, sharding={maximal device=1}, backend_config="{}"

  %normalize = f16[100000,200]{1,0} custom-call(f16[100000,200]{1,0} %all-reduce), custom_call_target="ReplicationNormalise", sharding={maximal device=1}, backend_config="{\"isInplace\":true}"

  %multiply = f16[100000,200]{1,0} multiply(f16[100000,200]{1,0} %broadcast, f16[100000,200]{1,0} %normalize), sharding={maximal device=1}, backend_config="{}"
  ROOT %subtract = f16[100000,200]{1,0} subtract(f16[100000,200]{1,0} %embeddings, f16[100000,200]{1,0} %multiply), sharding={maximal device=1},backend_config="{\"isInplace\":true}"
}

)";
  /*
  Expected output.

  ENTRY %c1 (updates: f16[1000,200], indices: s32[1000], embeddings:
  f16[100000,200], zero_tensor: f16[100000,200], scale: f16[], learning_rate:
  f16[]) -> f16[100000,200] { %zero_tensor = f16[100000,200]{1,0} parameter(3)
    %scale = f16[] parameter(4)
    %embeddings = f16[100000,200]{1,0} parameter(2)
    %indices = s32[1000]{0} parameter(1)
    %custom-call.2 = s32[4,1000]{1,0} custom-call(s32[1000]{0} %indices),
  custom_call_target="AllGather", backend_config="{}" %reshape = s32[4000]{0}
  reshape(s32[4,1000]{1,0} %custom-call.2) %updates = f16[1000,200]{1,0}
  parameter(0) %custom-call.3 = f16[4,1000,200]{2,1,0}
  custom-call(f16[1000,200]{1,0} %updates), custom_call_target="AllGather",
  backend_config="{}" %reshape.1 = f16[4000,200]{1,0}
  reshape(f16[4,1000,200]{2,1,0} %custom-call.3) %custom-call.4 =
  f16[4000,200]{1,0} custom-call(f16[4000,200]{1,0} %reshape.1),
  custom_call_target="ReplicationNormalise", backend_config="{}" %learning_rate
  = f16[] parameter(5) %broadcast.1 = f16[4000,200]{1,0} broadcast(f16[]
  %learning_rate), dimensions={} %multiply.1 = f16[4000,200]{1,0}
  multiply(f16[4000,200]{1,0} %custom-call.4, f16[4000,200]{1,0} %broadcast.1)
    %constant = f16[] constant(-1)
    ROOT %custom-call.5 = f16[100000,200]{1,0} custom-call(f16[100000,200]{1,0}
  %embeddings, s32[4000]{0} %reshape, f16[4000,200]{1,0} %multiply.1, f16[]
  %constant), custom_call_target="MultiUpdateAdd",
  update_dim=1, sharding={maximal device=1},
  backend_config="{\"hashOfCustomAttributes\":\"-7046029254386353152\"}"
  }
  */

  auto module = ParseAndReturnVerifiedModule(hlo);

  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);

  HloComputation* computation = hlo_module->entry_computation();

  const HloInstruction* updates = computation->parameter_instruction(0);
  const HloInstruction* indices = computation->parameter_instruction(1);
  const HloInstruction* embeddings = computation->parameter_instruction(2);
  const HloInstruction* learning_rate = computation->parameter_instruction(5);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(hlo_module).ValueOrDie());

  AllToAllFinder matcher(annotations, 4);
  EXPECT_TRUE(matcher.Run(hlo_module).ValueOrDie());

  // We expect the root to be now a multi update add.
  HloInstruction* inst = computation->root_instruction();

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst));

  const HloInstruction* expected_embeddings =
      inst->operand(0);  // Expected to be embeddings.
  const HloInstruction* expected_indices =
      inst->operand(1);  // Expected to be gathered indices.
  const HloInstruction* expected_updates =
      inst->operand(2);  // Expected to be gathered updates.
  const HloInstruction* expected_scale =
      inst->operand(3);  // Expected to be the scale.

  // Check the embedding.
  EXPECT_EQ(expected_embeddings, embeddings);

  // Check the indices.
  EXPECT_EQ(expected_indices->shape().dimensions_size(), 1);
  EXPECT_EQ(expected_indices->shape().dimensions()[0],
            4000);  // 4000 = 4 replicas * 1000

  const HloReshapeInstruction* reshape_ind =
      dynamic_cast<const HloReshapeInstruction*>(expected_indices);

  EXPECT_NE(reshape_ind, nullptr);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(reshape_ind->operand(0)));
  EXPECT_EQ(reshape_ind->operand(0)->operand(0), indices);

  // Check the scale.
  const HloConstantInstruction* scale_as_const =
      dynamic_cast<const HloConstantInstruction*>(expected_scale);
  EXPECT_NE(scale_as_const, nullptr);

  EXPECT_TRUE(scale_as_const->literal().IsAllFloat(-1.0f));

  // Check update.
  EXPECT_EQ(expected_updates->opcode(), HloOpcode::kMultiply);

  const HloBroadcastInstruction* expected_broadcast =
      dynamic_cast<const HloBroadcastInstruction*>(
          expected_updates->operand(1));
  EXPECT_NE(expected_broadcast, nullptr);
  EXPECT_EQ(learning_rate, expected_broadcast->operand(0));

  const HloInstruction* expected_replication_normalize =
      expected_updates->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReplicationNormalise)(
      expected_replication_normalize));

  const HloReshapeInstruction* reshape_up =
      dynamic_cast<const HloReshapeInstruction*>(
          expected_replication_normalize->operand(0));

  EXPECT_NE(reshape_up, nullptr);

  const HloInstruction* expected_update_gather = reshape_up->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(expected_update_gather));
  EXPECT_EQ(expected_update_gather->operand(0), updates);
}

TEST_F(HloReplaceEmbeddingAllRed, NoRunUnsupportedType) {
  std::string hlo = R"(
HloModule top

%to_apply_func (x: s32[], y: s32[]) -> s32[] {
  %x = s32[] parameter(0), backend_config="{}"
  %y = s32[] parameter(1), backend_config="{}"
  ROOT %add = s32[] add(s32[] %x, s32[] %y), backend_config="{\"isInplace\":true}"
}

ENTRY c1 {
  %updates = s32[1000, 200] parameter(0)
  %indices = s32[1000] parameter(1)
  %embeddings = s32[100000, 200] parameter(2)
  %zero_tensor = s32[100000, 200] parameter(3)
  %scale = s32[] parameter(4)
  %learning_rate = s32[] parameter(5)
  %broadcast = s32[100000, 200]{1,0} broadcast(s32[] %learning_rate), dimensions={}, backend_config="{}"


  %multi_update_add = s32[100000,200]{1,0} custom-call(s32[100000, 200]{1,0} %zero_tensor,
       s32[1000]{0} %indices, s32[1000, 200]{1,0} %updates,
       s32[] %scale), custom_call_target="MultiUpdateAdd", sharding={maximal device=1}, backend_config="{\"isInplace\":true,\"hashOfCustomAttributes\":\"-7046029254386353152\"}"

  %all-reduce = s32[100000,200]{1,0} all-reduce(s32[100000,200]{1,0} %multi_update_add),to_apply=%to_apply_func, replica_groups={}, sharding={maximal device=1}, backend_config="{}"

  %normalize = s32[100000,200]{1,0} custom-call(s32[100000,200]{1,0} %all-reduce), custom_call_target="ReplicationNormalise", sharding={maximal device=1}, backend_config="{\"isInplace\":true}"

  %multiply = s32[100000,200]{1,0} multiply(s32[100000,200]{1,0} %broadcast, s32[100000,200]{1,0} %normalize), sharding={maximal device=1}, backend_config="{}"
  ROOT %subtract = s32[100000,200]{1,0} subtract(s32[100000,200]{1,0} %embeddings, s32[100000,200]{1,0} %multiply), sharding={maximal device=1},backend_config="{\"isInplace\":true}"
}

)";

  auto module = ParseAndReturnVerifiedModule(hlo);

  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);

  HloComputation* computation = hlo_module->entry_computation();

  const HloInstruction* updates = computation->parameter_instruction(0);
  const HloInstruction* indices = computation->parameter_instruction(1);
  const HloInstruction* embeddings = computation->parameter_instruction(2);
  const HloInstruction* learning_rate = computation->parameter_instruction(5);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(hlo_module).ValueOrDie());

  AllToAllFinder matcher(annotations, 4);
  EXPECT_FALSE(matcher.Run(hlo_module).ValueOrDie());
}

TEST_F(HloReplaceEmbeddingAllRed, NoRunTooCostly) {
  std::string hlo = R"(
HloModule top

%to_apply_func (x: f16[], y: f16[]) -> f16[] {
  %x = f16[] parameter(0), backend_config="{}"
  %y = f16[] parameter(1), backend_config="{}"
  ROOT %add = f16[] add(f16[] %x, f16[] %y), backend_config="{\"isInplace\":true}"
}

ENTRY c1 {
  %updates = f16[10000, 200] parameter(0)
  %indices = s32[10000] parameter(1)
  %embeddings = f16[10000, 200] parameter(2)
  %zero_tensor = f16[10000, 200] parameter(3)
  %scale = f16[] parameter(4)
  %learning_rate = f16[] parameter(5)
  %broadcast = f16[10000, 200]{1,0} broadcast(f16[] %learning_rate), dimensions={}, backend_config="{}"

  %multi_update_add = f16[10000,200]{1,0} custom-call(f16[10000, 200]{1,0} %zero_tensor,
       s32[10000]{0} %indices, f16[10000, 200]{1,0} %updates,
       f16[] %scale), custom_call_target="MultiUpdateAdd", sharding={maximal device=1}, backend_config="{\"isInplace\":true,\"hashOfCustomAttributes\":\"-7046029254386353152\"}"

  %all-reduce = f16[10000,200]{1,0} all-reduce(f16[10000,200]{1,0} %multi_update_add),to_apply=%to_apply_func, replica_groups={}, sharding={maximal device=1}, backend_config="{}"

  %normalize = f16[10000,200]{1,0} custom-call(f16[10000,200]{1,0} %all-reduce), custom_call_target="ReplicationNormalise", sharding={maximal device=1}, backend_config="{\"isInplace\":true}"

  %multiply = f16[10000,200]{1,0} multiply(f16[10000,200]{1,0} %broadcast, f16[10000,200]{1,0} %normalize), sharding={maximal device=1}, backend_config="{}"
  ROOT %subtract = f16[10000,200]{1,0} subtract(f16[10000,200]{1,0} %embeddings, f16[10000,200]{1,0} %multiply), sharding={maximal device=1},backend_config="{\"isInplace\":true}"
}

)";
  auto module = ParseAndReturnVerifiedModule(hlo);

  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);

  HloComputation* computation = hlo_module->entry_computation();

  const HloInstruction* updates = computation->parameter_instruction(0);
  const HloInstruction* indices = computation->parameter_instruction(1);
  const HloInstruction* embeddings = computation->parameter_instruction(2);
  const HloInstruction* learning_rate = computation->parameter_instruction(5);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(hlo_module).ValueOrDie());

  AllToAllFinder matcher(annotations, 4);
  EXPECT_FALSE(matcher.Run(hlo_module).ValueOrDie());
}

TEST_F(HloReplaceEmbeddingAllRed, SerialiseDueToReplicationFactor) {
  std::string hlo = R"(
HloModule top

%to_apply_func (x: f16[], y: f16[]) -> f16[] {
  %x = f16[] parameter(0), backend_config="{}"
  %y = f16[] parameter(1), backend_config="{}"
  ROOT %add = f16[] add(f16[] %x, f16[] %y), backend_config="{\"isInplace\":true}"
}

ENTRY c1 {
  %updates = f16[1000, 200] parameter(0)
  %indices = s32[1000] parameter(1)
  %embeddings = f16[100000, 200] parameter(2)
  %zero_tensor = f16[100000, 200] parameter(3)
  %scale = f16[] parameter(4)
  %learning_rate = f16[] parameter(5)
  %broadcast = f16[100000, 200]{1,0} broadcast(f16[] %learning_rate), dimensions={}, backend_config="{}"


  %multi_update_add = f16[100000,200]{1,0} custom-call(f16[100000, 200]{1,0} %zero_tensor,
       s32[1000]{0} %indices, f16[1000, 200]{1,0} %updates,
       f16[] %scale), custom_call_target="MultiUpdateAdd", sharding={maximal device=1}, backend_config="{\"isInplace\":true,\"hashOfCustomAttributes\":\"-7046029254386353152\"}"

  %all-reduce = f16[100000,200]{1,0} all-reduce(f16[100000,200]{1,0} %multi_update_add),to_apply=%to_apply_func, replica_groups={}, sharding={maximal device=1}, backend_config="{}"

  %normalize = f16[100000,200]{1,0} custom-call(f16[100000,200]{1,0} %all-reduce), custom_call_target="ReplicationNormalise", sharding={maximal device=1}, backend_config="{\"isInplace\":true}"

  %multiply = f16[100000,200]{1,0} multiply(f16[100000,200]{1,0} %broadcast, f16[100000,200]{1,0} %normalize), sharding={maximal device=1}, backend_config="{}"
  ROOT %subtract = f16[100000,200]{1,0} subtract(f16[100000,200]{1,0} %embeddings, f16[100000,200]{1,0} %multiply), sharding={maximal device=1},backend_config="{\"isInplace\":true}"
}

)";
  /*
  Expected output.

  ENTRY %c1 (updates: f16[1000,200], indices: s32[1000], embeddings:
  f16[100000,200], zero_tensor: f16[100000,200], scale: f16[], learning_rate:
  f16[]) -> f16[100000,200] { %zero_tensor = f16[100000,200]{1,0} parameter(3)
    %scale = f16[] parameter(4)
    %embeddings = f16[100000,200]{1,0} parameter(2)
    %indices = s32[1000]{0} parameter(1)
    %custom-call.2 = s32[8,1000]{1,0} custom-call(s32[1000]{0} %indices),
  custom_call_target="AllGather", backend_config="{}" %reshape = s32[8000]{0}
  reshape(s32[8,1000]{1,0} %custom-call.2) %slice = s32[4000]{0}
  slice(s32[8000]{0} %reshape), slice={[0:4000]} %updates = f16[1000,200]{1,0}
  parameter(0) %custom-call.3 = f16[8,1000,200]{2,1,0}
  custom-call(f16[1000,200]{1,0} %updates), custom_call_target="AllGather",
  backend_config="{}" %reshape.1 = f16[8000,200]{1,0}
  reshape(f16[8,1000,200]{2,1,0} %custom-call.3) %custom-call.4 =
  f16[8000,200]{1,0} custom-call(f16[8000,200]{1,0} %reshape.1),
  custom_call_target="ReplicationNormalise", backend_config="{}" %learning_rate
  = f16[] parameter(5) %broadcast.1 = f16[8000,200]{1,0} broadcast(f16[]
  %learning_rate), dimensions={} %multiply.1 = f16[8000,200]{1,0}
  multiply(f16[8000,200]{1,0} %custom-call.4, f16[8000,200]{1,0} %broadcast.1)
    %slice.1 = f16[4000,200]{1,0} slice(f16[8000,200]{1,0} %multiply.1),
  slice={[0:4000], [0:200]} %constant = f16[] constant(-1) %custom-call.5 =
  f16[100000,200]{1,0} custom-call(f16[100000,200]{1,0} %embeddings,
  s32[4000]{0} %slice, f16[4000,200]{1,0} %slice.1, f16[] %constant),
  custom_call_target="MultiUpdateAdd", update_dim=1,
  backend_config="{\"hashOfCustomAttributes\":\"-7046029254386353152\"}"
    %slice.2 = s32[4000]{0} slice(s32[8000]{0} %reshape), slice={[4000:8000]}
    %slice.3 = f16[4000,200]{1,0} slice(f16[8000,200]{1,0} %multiply.1),
  slice={[4000:8000], [0:200]} ROOT %custom-call.6 = f16[100000,200]{1,0}
  custom-call(f16[100000,200]{1,0} %custom-call.5, s32[4000]{0} %slice.2,
  f16[4000,200]{1,0} %slice.3, f16[] %constant),
  custom_call_target="MultiUpdateAdd", update_dim=1,
  sharding={maximal device=1},
  backend_config="{\"hashOfCustomAttributes\":\"-7046029254386353152\"}"
  }
  */
  auto module = ParseAndReturnVerifiedModule(hlo);

  EXPECT_TRUE(module.ok());
  auto* hlo_module = module.ValueOrDie().get();

  CompilerAnnotations annotations(hlo_module);

  HloComputation* computation = hlo_module->entry_computation();

  const HloInstruction* updates = computation->parameter_instruction(0);
  const HloInstruction* indices = computation->parameter_instruction(1);
  const HloInstruction* embeddings = computation->parameter_instruction(2);
  const HloInstruction* learning_rate = computation->parameter_instruction(5);

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(hlo_module).ValueOrDie());

  AllToAllFinder matcher(annotations, 8);
  EXPECT_TRUE(matcher.Run(hlo_module).ValueOrDie());
  // We expect the root to be now a multi update add.
  HloInstruction* inst = computation->root_instruction();

  // We are expecting two multiupdate adds.
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst));

  const HloInstruction* expected_2nd_multi_update_add =
      inst->operand(0);  // Expected to be embeddings.
  const HloInstruction* expected_indices_slice_1 =
      inst->operand(1);  // Expected to be gathered indices.
  const HloInstruction* expected_updates_slice_1 =
      inst->operand(2);  // Expected to be gathered updates.
  const HloInstruction* expected_scale_1 =
      inst->operand(3);  // Expected to be the scale.

  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(
      expected_2nd_multi_update_add));

  const HloInstruction* expected_embeddings =
      expected_2nd_multi_update_add->operand(0);  // Expected to be embeddings.
  const HloInstruction* expected_indices_slice_2 =
      expected_2nd_multi_update_add->operand(
          1);  // Expected to be gathered indices.
  const HloInstruction* expected_updates_slice_2 =
      expected_2nd_multi_update_add->operand(
          2);  // Expected to be gathered updates.
  const HloInstruction* expected_scale =
      expected_2nd_multi_update_add->operand(3);  // Expected to be the scale.

  // Check the embedding.
  EXPECT_EQ(expected_embeddings, embeddings);

  EXPECT_EQ(expected_scale_1, expected_scale);

  // Check the indices.
  EXPECT_EQ(expected_indices_slice_1->shape().dimensions_size(), 1);
  EXPECT_EQ(expected_indices_slice_1->shape().dimensions()[0],
            4000);  // 4000 = (8 replicas * 1000) / 2
  EXPECT_EQ(expected_indices_slice_2->shape().dimensions_size(), 1);
  EXPECT_EQ(expected_indices_slice_2->shape().dimensions()[0],
            4000);  // 4000 = (8 replicas * 1000) / 2

  // Check that it is a slice.
  const HloSliceInstruction* expected_slice_ind_1 =
      dynamic_cast<const HloSliceInstruction*>(expected_indices_slice_1);
  const HloSliceInstruction* expected_slice_ind_2 =
      dynamic_cast<const HloSliceInstruction*>(expected_indices_slice_2);

  EXPECT_NE(expected_slice_ind_1, nullptr);
  EXPECT_NE(expected_slice_ind_2, nullptr);
  EXPECT_NE(expected_slice_ind_1, expected_slice_ind_2);
  EXPECT_EQ(expected_slice_ind_1->operand(0), expected_slice_ind_2->operand(0));

  const HloReshapeInstruction* reshape_ind =
      dynamic_cast<const HloReshapeInstruction*>(
          expected_slice_ind_1->operand(0));

  EXPECT_NE(reshape_ind, nullptr);
  EXPECT_TRUE(
      IsPoplarInstruction(PoplarOp::AllGather)(reshape_ind->operand(0)));
  EXPECT_EQ(reshape_ind->operand(0)->operand(0), indices);

  // Check the scale.
  const HloConstantInstruction* scale_as_const =
      dynamic_cast<const HloConstantInstruction*>(expected_scale);
  EXPECT_NE(scale_as_const, nullptr);

  EXPECT_TRUE(scale_as_const->literal().IsAllFloat(-1.0f));

  // Check update.
  EXPECT_EQ(expected_updates_slice_1->shape().dimensions_size(), 2);
  EXPECT_EQ(expected_updates_slice_1->shape().dimensions()[0],
            4000);  // 4000 = (8 replicas * 1000) / 2
  EXPECT_EQ(expected_updates_slice_1->shape().dimensions()[1], 200);

  EXPECT_EQ(expected_updates_slice_2->shape().dimensions_size(), 2);
  EXPECT_EQ(expected_updates_slice_2->shape().dimensions()[0],
            4000);  // 4000 = (8 replicas * 1000) / 2
  EXPECT_EQ(expected_updates_slice_2->shape().dimensions()[1], 200);

  // Check that it is a slice.
  const HloSliceInstruction* expected_slice_up_1 =
      dynamic_cast<const HloSliceInstruction*>(expected_updates_slice_1);
  const HloSliceInstruction* expected_slice_up_2 =
      dynamic_cast<const HloSliceInstruction*>(expected_updates_slice_2);

  EXPECT_NE(expected_slice_up_1, nullptr);
  EXPECT_NE(expected_slice_up_2, nullptr);
  EXPECT_NE(expected_slice_up_1, expected_slice_up_2);
  EXPECT_EQ(expected_slice_up_1->operand(0), expected_slice_up_2->operand(0));

  const HloInstruction* expected_multiply = expected_slice_up_1->operand(0);
  EXPECT_EQ(expected_multiply->opcode(), HloOpcode::kMultiply);

  const HloBroadcastInstruction* expected_broadcast =
      dynamic_cast<const HloBroadcastInstruction*>(
          expected_multiply->operand(1));
  EXPECT_NE(expected_broadcast, nullptr);
  EXPECT_EQ(learning_rate, expected_broadcast->operand(0));

  const HloInstruction* expected_replication_normalize =
      expected_multiply->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::ReplicationNormalise)(
      expected_replication_normalize));

  const HloReshapeInstruction* reshape_up =
      dynamic_cast<const HloReshapeInstruction*>(
          expected_replication_normalize->operand(0));

  EXPECT_NE(reshape_up, nullptr);

  const HloInstruction* expected_update_gather = reshape_up->operand(0);
  EXPECT_TRUE(IsPoplarInstruction(PoplarOp::AllGather)(expected_update_gather));
  EXPECT_EQ(expected_update_gather->operand(0), updates);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
