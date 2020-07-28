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
#include "tensorflow/compiler/plugin/poplar/driver/passes/embeddings_gradient_optimizer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

struct EmbeddingsGradientOptimizerTestSpec {
  int embedding_rows;
  int embedding_size;
  int batch_size;
  int num_batches;

  bool changed;
};

class EmbeddingsGradientOptimizerTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<EmbeddingsGradientOptimizerTestSpec, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    EmbeddingsGradientOptimizerTestCases, EmbeddingsGradientOptimizerTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<EmbeddingsGradientOptimizerTestSpec>{
            {100, 10, 2, 10, true}, {10, 1000, 2, 10, false}}),
        ::testing::Bool()));

std::string GetTemplateHloString() {
  return R"(
    HloModule main

    %WeightUpdate (arg0.31: f32[$R,$E], arg1.32: f32[$R,$E], arg2.33: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = f32[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = f32[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %add = f32[$R,$E] add(f32[$R,$E] %arg0, f32[$R,$E] %arg2), metadata={op_type="ResourceApplyMomentum" op_name="Add"}, backend_config="{}"
    ROOT %mul = f32[$R,$E] subtract(f32[$R,$E] %arg1, f32[$R,$E] %add), metadata={op_type="ResourceApplyMomentum" op_name="Mul"}, backend_config="{}"
    }

    %RepeatLoopBody (arg0: f32[$R,$E], arg1: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0)
    %arg1 = f32[$R,$E] parameter(1)
    %const.scale = f32[] constant(0.1)
    %const.indice = s32[] constant(1)
    %const.update = f32[] constant(1)

    %broadcast.indices = s32[$BS,1] broadcast(s32[] %const.indice), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Indices"}, backend_config="{}"
    %broadcast.update = f32[$BS,$E] broadcast(f32[] %const.update), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Update"}, backend_config="{}"

    %custom-call.1 = f32[$R,$E] custom-call(f32[$R,$E] %arg0), custom_call_target="GradientAccumulatorCreate", custom_call_has_side_effect=true, metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate"}, backend_config="{}"
    %custom-call.2 = f32[$R,$E] custom-call(f32[$R,$E] %arg1, s32[$BS,1] %broadcast.indices, f32[$BS,$E] %broadcast.update, f32[] %const.scale), custom_call_target="MultiUpdateAdd", metadata={op_type="IpuMultiUpdateAdd" op_name="gradients/embedding_lookup_grad/IpuMultiUpdateAdd"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1,\"serialization_factor\":1}"
    %custom-call.3 = f32[$R,$E] custom-call(f32[$R,$E] %custom-call.1, f32[$R,$E] %custom-call.2), custom_call_target="GradientAccumulatorAdd", metadata={op_type="GradientAccumulatorAdd" op_name="GradientAccumulatorAdd"}, backend_config="{}"
    %custom-call.4 = f32[$R,$E] custom-call(f32[$R,$E] %custom-call.3), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink"}, backend_config="{\"num_mini_batches\":$BN}"
    ROOT %call.1 = f32[$R,$E] call(f32[$R,$E] %arg0, f32[$R,$E] %arg1, f32[$R,$E] %custom-call.4), to_apply=%WeightUpdate, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"numBatchesToAccumulate\":\"$BN\",\"offloadVariables\":true}}}"
    }

    ENTRY %main (arg0: f32[$R,$E], arg1: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = f32[$R,$E] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/b"}, backend_config="{}"
    ROOT %call.2 = f32[$R,$E] call(f32[$R,$E] %arg1, f32[$R,$E] %arg0), to_apply=%RepeatLoopBody, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"40\"}}}"
    }
    )";
}

std::string GetPipelineTemplateHloString() {
  return R"(
    HloModule main

    %WeightUpdate (arg0: f32[$R,$E], arg1: f32[$R,$E], arg2: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = f32[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = f32[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %add = f32[$R,$E] add(f32[$R,$E] %arg0, f32[$R,$E] %arg2), metadata={op_type="ResourceApplyMomentum" op_name="Add"}, backend_config="{}"
    ROOT %mul = f32[$R,$E] subtract(f32[$R,$E] %arg1, f32[$R,$E] %add), metadata={op_type="ResourceApplyMomentum" op_name="Mul"}, backend_config="{}"
    }

    %PipelineStage (arg0: f32[$R,$E], arg1: f32[$R,$E], arg2: f32[$R,$E]) -> (f32[$R,$E]) {
    %arg0 = f32[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = f32[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = f32[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"

    %const.scale = f32[] constant(0.1)
    %const.indice = s32[] constant(1)
    %const.update = f32[] constant(1)

    %broadcast.indices = s32[$BS,1] broadcast(s32[] %const.indice), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Indices"}, backend_config="{}"
    %broadcast.update = f32[$BS,$E] broadcast(f32[] %const.update), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Update"}, backend_config="{}"

    %add = f32[$R,$E] add(f32[$R,$E] %arg0, f32[$R,$E] %arg1)
    %custom-call.1 = f32[$R,$E] custom-call(f32[$R,$E] %add, s32[$BS,1] %broadcast.indices, f32[$BS,$E] %broadcast.update, f32[] %const.scale), custom_call_target="MultiUpdateAdd", metadata={op_type="IpuMultiUpdateAdd" op_name="gradients/embedding_lookup_grad/IpuMultiUpdateAdd"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1,\"serialization_factor\":1}"
    %custom-call.2 = f32[$R,$E] custom-call(f32[$R,$E] %arg2, f32[$R,$E] %custom-call.1), custom_call_target="GradientAccumulatorAdd", metadata={op_type="GradientAccumulatorAdd" op_name="GradientAccumulatorAdd"}, backend_config="{}"

    ROOT %tuple = (f32[$R,$E]) tuple(f32[$R,$E] %custom-call.2)
    }

    %Pipeline (arg0: f32[$R,$E], arg1: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0)
    %arg1 = f32[$R,$E] parameter(1)

    %custom-call.1 = f32[$R,$E] custom-call(f32[$R,$E] %arg0), custom_call_target="GradientAccumulatorCreate", custom_call_has_side_effect=true, metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate"}, backend_config="{}"
    %call.1 = (f32[$R,$E]) call(f32[$R,$E] %arg0, f32[$R,$E] %arg1, f32[$R,$E] %custom-call.1), to_apply=%PipelineStage, frontend_attributes={CALL_CONFIG_TYPE=PipelineStageBackward}, metadata={op_type="PipelineStageBackward" op_name="gradients/pipeline_stage_1/PipelineStage_grad/PipelineStageBackward"}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
    %get-tuple-element.1 = f32[$R,$E] get-tuple-element((f32[$R,$E]) %call.1), index=0
    %custom-call.2 = f32[$R,$E] custom-call(f32[$R,$E] %get-tuple-element.1), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink"}, backend_config="{\"num_mini_batches\":$BN}"
    ROOT %call.2 = f32[$R,$E] call(f32[$R,$E] %arg0, f32[$R,$E] %arg1, f32[$R,$E] %custom-call.2), to_apply=%WeightUpdate, frontend_attributes={CALL_CONFIG_TYPE=ResourceUpdate}, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"numBatchesToAccumulate\":\"$BN\",\"offloadVariables\":true}}}"
    }

    ENTRY %main (arg0: f32[$R,$E], arg1: f32[$R,$E]) -> f32[$R,$E] {
    %arg0 = f32[$R,$E] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = f32[$R,$E] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/b"}, backend_config="{}"
    ROOT %call.1 = f32[$R,$E] call(f32[$R,$E] %arg0, f32[$R,$E] %arg1), to_apply=%Pipeline, frontend_attributes={CALL_CONFIG_TYPE=Pipeline}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"pipelineDepth\":\"$BN\",\"repeatCount\":\"2\",\"schedule\":\"Interleaved\"}}}"
    }
    )";
}

std::string ReplaceTemplateValues(
    std::string hlo_string, const EmbeddingsGradientOptimizerTestSpec& spec) {
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$R", std::to_string(spec.embedding_rows), true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$E", std::to_string(spec.embedding_size), true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$BS", std::to_string(spec.batch_size), true);
  hlo_string = tensorflow::str_util::StringReplace(
      hlo_string, "$BN", std::to_string(spec.num_batches), true);
  return hlo_string;
}

std::string GetHloString(const EmbeddingsGradientOptimizerTestSpec& spec) {
  return ReplaceTemplateValues(GetTemplateHloString(), spec);
}

std::string GetPipelineHloString(
    const EmbeddingsGradientOptimizerTestSpec& spec) {
  return ReplaceTemplateValues(GetPipelineTemplateHloString(), spec);
}

TEST_P(EmbeddingsGradientOptimizerTest, DoTest) {
  auto param = GetParam();
  auto spec = std::get<0>(param);
  auto pipeline = std::get<1>(param);

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto hlo_string = pipeline ? GetPipelineHloString(spec) : GetHloString(spec);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  auto module0 = module.get();
  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module0));
  EXPECT_TRUE(custom_op_replaced);

  EmbeddingsGradientOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  EXPECT_EQ(changed, spec.changed);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
