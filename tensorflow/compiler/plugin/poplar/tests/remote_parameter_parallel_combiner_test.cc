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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_parameter_parallel_combiner.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remote_parameter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using RemoteParameterParallelCombinerTest = HloTestBase;

const auto is_load = IsPoplarInstruction(RemoteParameterLoad);
const auto is_store = IsPoplarInstruction(RemoteParameterStore);

TEST_F(RemoteParameterParallelCombinerTest, TestCombineTwoLoads) {
  const auto hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[2]) -> (f32[], f32[2]) {
  %arg1 = f32[] parameter(0)
  %arg2 = f32[2] parameter(1)
  %load1 = f32[] custom-call(f32[] %arg1), custom_call_target="RemoteParameterLoad", sharding={maximal device=0}
  %load2 = f32[2] custom-call(f32[2] %arg2), custom_call_target="RemoteParameterLoad", sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[2]) tuple(%load1, %load2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  TensorAllocationMap allocation_map;
  ASSERT_TRUE(RemoteParameterParallelCombiner(allocation_map)
                  .RunOnComputation(module->entry_computation())
                  .ValueOrDie());

  auto seq = module->entry_computation()->MakeInstructionPostOrder();

  EXPECT_EQ(absl::c_count_if(seq, is_load), 1);

  auto load_inst = Cast<HloRemoteParameterLoad>(*absl::c_find_if(seq, is_load));

  // Check that they were merged.
  EXPECT_EQ(load_inst->operand_count(), 2);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kGetTupleElement);

  // Check that the inputs and outputs are wired correctly, irrespective of the
  // order in which they were combind.
  const int64 gte0_index =
      Cast<HloGetTupleElementInstruction>(root->operand(0))->tuple_index();
  const int64 gte1_index =
      Cast<HloGetTupleElementInstruction>(root->operand(1))->tuple_index();

  const auto* param0 =
      Cast<HloParameterInstruction>(load_inst->operand(gte0_index));
  const auto* param1 =
      Cast<HloParameterInstruction>(load_inst->operand(gte1_index));

  EXPECT_EQ(param0->parameter_number(), 0);
  EXPECT_EQ(param1->parameter_number(), 1);

  // Check the sharding.
  EXPECT_TRUE(load_inst->sharding().IsTuple());
  const auto shardings = load_inst->sharding().tuple_elements();
  EXPECT_EQ(shardings.size(), 2);
  EXPECT_EQ(shardings.at(gte0_index).UniqueDevice().value(), 0);
  EXPECT_EQ(shardings.at(gte1_index).UniqueDevice().value(), 1);
}

TEST_F(RemoteParameterParallelCombinerTest, TestControlPredecessor) {
  const auto hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[2]) -> (f32[], f32[2]) {
  %arg1 = f32[] parameter(0)
  %arg2 = f32[2] parameter(1)
  %load1 = f32[] custom-call(f32[] %arg1), custom_call_target="RemoteParameterLoad", sharding={maximal device=0}
  %load2 = f32[2] custom-call(f32[2] %arg2), custom_call_target="RemoteParameterLoad", sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[2]) tuple(%load1, %load2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  // Add control dependency.
  auto* load1 =
      module->entry_computation()->parameter_instruction(0)->users().at(0);
  auto* load2 =
      module->entry_computation()->parameter_instruction(1)->users().at(0);
  TF_CHECK_OK(load1->AddControlDependencyTo(load2));

  // Expect that we do nothing.
  TensorAllocationMap allocation_map;
  ASSERT_FALSE(RemoteParameterParallelCombiner(allocation_map)
                   .RunOnComputation(module->entry_computation())
                   .ValueOrDie());
}

TEST_F(RemoteParameterParallelCombinerTest, TestCombineTwoStores) {
  const auto hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[], arg2: f32[]) -> (f32[], f32[]) {
  %arg1 = f32[] parameter(0)
  %arg2 = f32[] parameter(1)
  %c1 = f32[] constant(1)
  %c2 = f32[] constant(2)
  %store1 = f32[] custom-call(f32[] %arg1, f32[] %c1), custom_call_target="RemoteParameterStore", sharding={maximal device=0}
  %store2 = f32[] custom-call(f32[] %arg2, f32[] %c2), custom_call_target="RemoteParameterStore", sharding={maximal device=1}
  ROOT %tuple = (f32[], f32[]) tuple(%store1, %store2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  TensorAllocationMap allocation_map;
  ASSERT_TRUE(RemoteParameterParallelCombiner(allocation_map)
                  .RunOnComputation(module->entry_computation())
                  .ValueOrDie());

  auto seq = module->entry_computation()->MakeInstructionPostOrder();

  EXPECT_EQ(absl::c_count_if(seq, is_store), 1);

  auto store_inst =
      Cast<HloRemoteParameterStore>(*absl::c_find_if(seq, is_store));

  // Check that they were merged.
  EXPECT_EQ(store_inst->operand_count(), 4);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kGetTupleElement);

  // Check that the inputs and outputs are wired correctly, irrespective of the
  // order in which they were combind.
  const int64 gte0_index =
      Cast<HloGetTupleElementInstruction>(root->operand(0))->tuple_index();
  const int64 gte1_index =
      Cast<HloGetTupleElementInstruction>(root->operand(1))->tuple_index();

  const auto* param0 =
      Cast<HloParameterInstruction>(store_inst->operand(gte0_index));
  const auto* param1 =
      Cast<HloParameterInstruction>(store_inst->operand(gte1_index));

  EXPECT_EQ(param0->parameter_number(), 0);
  EXPECT_EQ(param1->parameter_number(), 1);

  const auto* const0 =
      Cast<HloConstantInstruction>(store_inst->operand(2 + gte0_index));
  const auto* const1 =
      Cast<HloConstantInstruction>(store_inst->operand(2 + gte1_index));

  EXPECT_EQ(const0->literal().data<float>()[0], 1.0f);
  EXPECT_EQ(const1->literal().data<float>()[0], 2.0f);

  // Check the sharding.
  EXPECT_TRUE(store_inst->sharding().IsTuple());
  const auto shardings = store_inst->sharding().tuple_elements();
  EXPECT_EQ(shardings.size(), 2);
  EXPECT_EQ(shardings.at(gte0_index).UniqueDevice().value(), 0);
  EXPECT_EQ(shardings.at(gte1_index).UniqueDevice().value(), 1);
}

TEST_F(RemoteParameterParallelCombinerTest, TestSchedulingConstraints) {
  const auto hlo_string = R"(
HloModule top

ENTRY %top (arg1: f32[2], arg2: f32[2], arg3: f32[], arg4: f32[]) -> (f32[2], f32[2], f32[], f32[]) {
  %arg1 = f32[2] parameter(0)
  %arg2 = f32[2] parameter(1)
  %arg3 = f32[] parameter(2)
  %arg4 = f32[] parameter(3)

  %const1 = f32[2] constant(1)
  %const2 = f32[] constant(1)

  %load1 = f32[2] custom-call(f32[2] %arg1), custom_call_target="RemoteParameterLoad", sharding={maximal device=0}
  %load2 = f32[2] custom-call(f32[2] %arg2), custom_call_target="RemoteParameterLoad", sharding={maximal device=1}
  %load3 = f32[] custom-call(f32[] %arg3), custom_call_target="RemoteParameterLoad", sharding={maximal device=0}
  %load4 = f32[] custom-call(f32[] %arg4), custom_call_target="RemoteParameterLoad", sharding={maximal device=1}

  %add1 = f32[2] add(load1, const1), sharding={maximal device=0}
  %add2 = f32[2] add(load2, const1), sharding={maximal device=1}
  %add3 = f32[] add(load3, const2), sharding={maximal device=0}
  %add4 = f32[] add(load4, const2), sharding={maximal device=1}

  %store1 = f32[2] custom-call(f32[2] %arg1, %add1), custom_call_target="RemoteParameterStore", sharding={maximal device=0}
  %store2 = f32[2] custom-call(f32[2] %arg2, %add1), custom_call_target="RemoteParameterStore", sharding={maximal device=1}
  %store3 = f32[] custom-call(f32[] %arg3, %add1), custom_call_target="RemoteParameterStore", sharding={maximal device=0}
  %store4 = f32[] custom-call(f32[] %arg4, %add1), custom_call_target="RemoteParameterStore", sharding={maximal device=1}

  ROOT %tuple = (f32[2], f32[2], f32[], f32[]) tuple(%store1, %store2, %store3, %store4)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());

  TensorAllocationMap allocation_map;
  ASSERT_TRUE(RemoteParameterParallelCombiner(allocation_map)
                  .RunOnComputation(module->entry_computation())
                  .ValueOrDie());

  const auto* store1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_EQ(
      Cast<HloParameterInstruction>(store1->operand(0))->parameter_number(), 0);
  EXPECT_EQ(
      Cast<HloParameterInstruction>(store1->operand(1))->parameter_number(), 1);

  // Check that store1 is scheduled before load2.
  EXPECT_EQ(store1->control_successors().size(), 1);

  auto const* load2 =
      Cast<HloRemoteParameterLoad>(store1->control_successors()[0]);
  EXPECT_EQ(
      Cast<HloParameterInstruction>(load2->operand(0))->parameter_number(), 2);
  EXPECT_EQ(
      Cast<HloParameterInstruction>(load2->operand(1))->parameter_number(), 3);
}

TEST_F(RemoteParameterParallelCombinerTest, TestPipeline) {
  const auto hlo_string = R"(
HloModule top

stage_0_fwd {
  in0 = f32[1,4,4,2] parameter(0)
  in1 = f32[1,4,4,1] parameter(1)
  in2 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in0, in1, in2)
}

stage_1_fwd {
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,1] parameter(1)
  in3 = f32[] parameter(2)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[]) tuple(in1, in2, in3)
}

stage_1_bwd {
  in1_grad = f32[1,4,4,2] parameter(0)
  in2_grad = f32[1,4,4,1] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in1_grad, in2_grad)
}

stage_0_bwd {
  in0_grad = f32[1,4,4,2] parameter(0)
  in1_grad = f32[1,4,4,2] parameter(1)
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2]) tuple(in0_grad, in1_grad)
}

resource_update {
  arg0 = f32[1,4,4,2] parameter(0)
  arg1 = f32[1,4,4,2] parameter(1)
  arg2 = f32[1,4,4,1] parameter(2)
  arg3 = f32[] parameter(3)
  bcast1 = f32[1,4,4,1] broadcast(arg3), dimensions={}
  bcast2 = f32[1,4,4,2] broadcast(arg3), dimensions={}
  bcast3 = f32[1,4,2,4] broadcast(arg3), dimensions={}

  load0 = f32[1,4,4,2] custom-call(f32[1,4,4,2] %arg0), custom_call_target="RemoteParameterLoad"
  load1 = f32[1,4,4,2] custom-call(f32[1,4,4,2] %arg1), custom_call_target="RemoteParameterLoad"
  load2 = f32[1,4,4,1] custom-call(f32[1,4,4,1] %arg2), custom_call_target="RemoteParameterLoad"

  load0_t = f32[1,4,2,4] transpose(load0), dimensions={0, 1, 3, 2}

  new_arg0 = f32[1,4,2,4] add(load0_t, bcast3)
  new_arg1 = f32[1,4,4,2] add(load1, bcast2)
  new_arg2 = f32[1,4,4,1] add(load2, bcast1)

  new_arg0_t = f32[1,4,4,2] transpose(new_arg0), dimensions={0, 1, 3, 2}

  store0 = f32[1,4,4,2] custom-call(f32[1,4,4,2] %arg0, new_arg0_t), custom_call_target="RemoteParameterStore"
  store1 = f32[1,4,4,2] custom-call(f32[1,4,4,2] %arg1, new_arg1), custom_call_target="RemoteParameterStore"
  store2 = f32[1,4,4,1] custom-call(f32[1,4,4,1] %arg2, new_arg2), custom_call_target="RemoteParameterStore"

  ROOT t = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,1]) tuple(store0, store1, store2)
}

pipeline {
  after-all = token[] after-all()
  infeed = (f32[1,4,4,1], token[]) infeed(after-all), infeed_config="140121807314576"
  in0 = f32[1,4,4,1] get-tuple-element(infeed), index=0
  in1 = f32[1,4,4,2] parameter(0)
  in2 = f32[1,4,4,3] parameter(1)
  in3 = f32[1,4,4,4] parameter(2)
  in4 = f32[] parameter(3)
  stage_0 = (f32[1,4,4,2], f32[1,4,4,1], f32[]) call(in1, in0, in4), to_apply=stage_0_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_0 = f32[1,4,4,2] get-tuple-element(stage_0), index=0
  stage_0_1 = f32[1,4,4,1] get-tuple-element(stage_0), index=1
  stage_0_2 = f32[] get-tuple-element(stage_0), index=2
  stage_1 = (f32[1,4,4,2], f32[1,4,4,1], f32[]) call(stage_0_0, stage_0_1, stage_0_2), to_apply=stage_1_fwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStage\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=1}
  stage_1_1 = f32[1,4,4,2] get-tuple-element(stage_1), index=0
  stage_1_2 = f32[1,4,4,1] get-tuple-element(stage_1), index=1
  stage_1_3 = f32[] get-tuple-element(stage_1), index=2
  stage_1_bwd = (f32[1,4,4,2], f32[1,4,4,1]) call(stage_1_1, stage_1_2), to_apply=stage_1_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}", sharding={maximal device=0}
  stage_1_bwd_1 = f32[1,4,4,2] get-tuple-element(stage_1_bwd), index=0
  stage_1_bwd_2 = f32[1,4,4,1] get-tuple-element(stage_1_bwd), index=1
  stage_0_bwd = (f32[1,4,4,2], f32[1,4,4,2]) call(stage_1_bwd_1, stage_0_0), to_apply=stage_0_bwd, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"0\"}}}", sharding={maximal device=0}
  stage_0_bwd_0 = f32[1,4,4,2] get-tuple-element(stage_0_bwd), index=0
  call_ru = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,1]) call(stage_0_bwd_0, stage_1_bwd_1, stage_1_bwd_2, stage_1_3), to_apply=resource_update, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\"}}"
  gte0 = f32[1,4,4,2] get-tuple-element(call_ru), index=0
  gte1 = f32[1,4,4,2] get-tuple-element(call_ru), index=1
  gte2 = f32[1,4,4,1] get-tuple-element(call_ru), index=2
  ROOT tuple = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,1]) tuple(gte0, gte1, gte2, in4)
}

ENTRY e {
  e.in0 = f32[1,4,4,2] parameter(0), parameter_replication={false}
  e.in1 = f32[1,4,4,3] parameter(1), parameter_replication={false}
  e.in2 = f32[1,4,4,4] parameter(2), parameter_replication={false}
  e.in3 = f32[] parameter(3), parameter_replication={false}
  ROOT e.call = (f32[1,4,4,2], f32[1,4,4,2], f32[1,4,4,1], f32[]) call(e.in0, e.in1, e.in2, e.in3), to_apply=pipeline, backend_config="{\"callConfig\":{\"type\":\"Pipeline\"}}"
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingPass().Run(module));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  CompilerAnnotations annotations(module);

  EXPECT_TRUE(InplaceFinder().Run(module).ValueOrDie());
  EXPECT_TRUE(AllocationFinder(annotations).Run(module).ValueOrDie());
  EXPECT_TRUE(ForwardAllocation(annotations).Run(module).ValueOrDie());

  RemoteParameterParallelCombiner combiner(annotations.tensor_allocation_map);
  ASSERT_TRUE(combiner.Run(module).ValueOrDie());

  const auto pipeline_comp = FindComputation(module, "pipeline");
  TF_ASSERT_OK_AND_ASSIGN(auto stages, GetPipelineStages(pipeline_comp));

  ASSERT_TRUE(stages.resource_update);
  const auto* resource_update_comp = (*stages.resource_update)->to_apply();
  const auto seq = resource_update_comp->MakeInstructionPostOrder();

  EXPECT_EQ(absl::c_count_if(seq, is_load), 2);
  EXPECT_EQ(absl::c_count_if(seq, is_store), 2);

  bool found_combined_load = false;
  bool found_combined_store = false;

  const auto single_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});
  const auto combined_shape =
      ShapeUtil::MakeTupleShape({single_shape, single_shape});

  // Check that we merged the largest ones.
  for (const auto* inst : seq) {
    if (is_load(inst) && inst->operand_count() == 2) {
      found_combined_load = true;
      EXPECT_EQ(inst->shape(), combined_shape);

      // Check that the allocation map was updated.
      const auto& target0 =
          annotations.tensor_allocation_map.at(TensorLocation(inst, 0));
      EXPECT_EQ(target0.tgt->name(), "new_arg0");
      EXPECT_EQ(target0.backward_path.size(), 2);
      EXPECT_EQ(target0.backward_path[0]->opcode(),
                HloOpcode::kGetTupleElement);
      EXPECT_EQ(target0.backward_path[1]->opcode(), HloOpcode::kTranspose);

      const auto& target1 =
          annotations.tensor_allocation_map.at(TensorLocation(inst, 1));
      EXPECT_EQ(target1.tgt->name(), "new_arg1");
      EXPECT_EQ(target1.backward_path.size(), 1);
      EXPECT_EQ(target1.backward_path.front()->opcode(),
                HloOpcode::kGetTupleElement);
    } else if (is_store(inst) && inst->operand_count() == 4) {
      found_combined_store = true;
      EXPECT_EQ(inst->shape(), combined_shape);
    }
  }

  EXPECT_TRUE(found_combined_load);
  EXPECT_TRUE(found_combined_store);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
