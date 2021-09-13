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
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
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
  int64 batch_serialization_iterations;

  PrimitiveType data_type;
  PrimitiveType gradient_accumulator_type;
};

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingsGradientOptimizerTestSpec& spec) {
  return os << "{ embedding rows: " << spec.embedding_rows
            << ", size:" << spec.embedding_size
            << ", batch size: " << spec.batch_size
            << ", num batches: " << spec.num_batches
            << ", changed: " << spec.changed
            << ", batch_serialization_iterations: "
            << spec.batch_serialization_iterations;
}

class EmbeddingsGradientOptimizerTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<EmbeddingsGradientOptimizerTestSpec, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    EmbeddingsGradientOptimizerTestCases, EmbeddingsGradientOptimizerTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<EmbeddingsGradientOptimizerTestSpec>{
            {100, 10, 2, 10, true, 1, F32, F32},
            {100, 10, 2, 10, true, 1, F16, F32},
            {10, 1000, 2, 10, false, 1, F32, F32}}),
        ::testing::Bool()));

std::string GetTemplateHloString() {
  return R"(
    HloModule main

    %WeightUpdate {
    %arg0 = $DT[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = $DT[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = $AT[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %arg2_convert = $DT[$R,$E] convert(%arg2)
    %add = $DT[$R,$E] add($DT[$R,$E] %arg0, $DT[$R,$E] %arg2_convert), metadata={op_type="ResourceApplyMomentum" op_name="Add"}, backend_config="{}"
    ROOT %mul = $DT[$R,$E] subtract($DT[$R,$E] %arg1, $DT[$R,$E] %add), metadata={op_type="ResourceApplyMomentum" op_name="Mul"}, backend_config="{}"
    %counter_0 = s32[] constant($BN)
    gac = () custom-call(s32[] %counter_0), custom_call_target="GradientAccumulationCount"
    }

    %RepeatLoopBody {
    %arg0 = $DT[$R,$E] parameter(0)
    %arg1 = $DT[$R,$E] parameter(1)
    %const.scale = $DT[] constant(0.1)
    %const.indice = s32[] constant(1)
    %const.update = $DT[] constant(1)

    %broadcast.indices = s32[$BS,1] broadcast(s32[] %const.indice), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Indices"}, backend_config="{}"
    %broadcast.update = $DT[$BS,$E] broadcast($DT[] %const.update), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Update"}, backend_config="{}"

    %custom-call.1 = $AT[$R,$E] custom-call($DT[$R,$E] %arg0), custom_call_target="GradientAccumulatorCreate", custom_call_has_side_effect=true, metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate"}, backend_config="{}"
    %custom-call.2 = $DT[$R,$E] custom-call($DT[$R,$E] %arg1, s32[$BS,1] %broadcast.indices, $DT[$BS,$E] %broadcast.update, $DT[] %const.scale), custom_call_target="MultiUpdateAdd", metadata={op_type="IpuMultiUpdateAdd" op_name="gradients/embedding_lookup_grad/IpuMultiUpdateAdd"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1,\"serialization_factor\":1}"
    %custom-call.3 = $AT[$R,$E] custom-call($AT[$R,$E] %custom-call.1, $DT[$R,$E] %custom-call.2), custom_call_target="GradientAccumulatorAdd", metadata={op_type="GradientAccumulatorAdd" op_name="GradientAccumulatorAdd"}, backend_config="{}"
    %custom-call.4 = $AT[$R,$E] custom-call($AT[$R,$E] %custom-call.3), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink"}, backend_config="{\"num_mini_batches\":$BN}"
    ROOT %call.1 = $DT[$R,$E] call($DT[$R,$E] %arg0, $DT[$R,$E] %arg1, $AT[$R,$E] %custom-call.4), to_apply=%WeightUpdate, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
    }

    ENTRY %main {
    %arg0 = $DT[$R,$E] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = $DT[$R,$E] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/b"}, backend_config="{}"
    ROOT %call.2 = $DT[$R,$E] call($DT[$R,$E] %arg1, $DT[$R,$E] %arg0), to_apply=%RepeatLoopBody, backend_config="{\"callConfig\":{\"type\":\"RepeatLoop\",\"repeatConfig\":{\"repeatCount\":\"40\"}}}"
    }
    )";
}

std::string GetPipelineTemplateHloString() {
  return R"(
    HloModule main

    %WeightUpdate {
    %arg0 = $DT[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = $DT[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = $AT[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %arg3 = $DT[$E] parameter(3), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %arg2_convert = $DT[$R,$E] convert(%arg2)
    %add = $DT[$R,$E] add($DT[$R,$E] %arg0, $DT[$R,$E] %arg2_convert), metadata={op_type="ResourceApplyMomentum" op_name="Add"}, backend_config="{}"
    ROOT %mul = $DT[$R,$E] subtract($DT[$R,$E] %arg1, $DT[$R,$E] %add), metadata={op_type="ResourceApplyMomentum" op_name="Mul"}, backend_config="{}"
    %counter_0 = s32[] constant($BN)
    gac = () custom-call(s32[] %counter_0), custom_call_target="GradientAccumulationCount"
    }

    %PipelineStage {
    %arg0 = $DT[$R,$E] parameter(0), metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = $DT[$R,$E] parameter(1), metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = $AT[$R,$E] parameter(2), metadata={op_name="XLA_Args/c"}, backend_config="{}"
    %arg3 = $DT[$E] parameter(3), metadata={op_name="XLA_Args/c"}, backend_config="{}"

    %const.scale = $DT[] constant(0.1)
    %const.indice = s32[] constant(1)
    %const.update = $DT[] constant(1)

    %broadcast.indices = s32[$BS,1] broadcast(s32[] %const.indice), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Indices"}, backend_config="{}"
    %broadcast.update = $DT[$BS,$E] broadcast($DT[] %const.update), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="Update"}, backend_config="{}"

    %add = $DT[$R,$E] add($DT[$R,$E] %arg0, $DT[$R,$E] %arg1)
    %custom-call.1 = $DT[$R,$E] custom-call($DT[$R,$E] %add, s32[$BS,1] %broadcast.indices, $DT[$BS,$E] %broadcast.update, $DT[] %const.scale), custom_call_target="MultiUpdateAdd", metadata={op_type="IpuMultiUpdateAdd" op_name="gradients/embedding_lookup_grad/IpuMultiUpdateAdd"}, backend_config="{\"index_vector_dim\":1,\"update_dim\":1,\"serialization_factor\":1}"
    %custom-call.2 = $AT[$R,$E] custom-call($AT[$R,$E] %arg2, $DT[$R,$E] %custom-call.1), custom_call_target="GradientAccumulatorAdd", metadata={op_type="GradientAccumulatorAdd" op_name="GradientAccumulatorAdd"}, backend_config="{}"

    ROOT %tuple = ($AT[$R,$E], $DT[$E]) tuple(%custom-call.2, arg3)
    }

    %Pipeline {
    %arg0 = $DT[$R,$E] parameter(0)
    %arg1 = $DT[$R,$E] parameter(1)
    %arg2 = $DT[$E] parameter(2)

    %custom-call.1 = $AT[$R,$E] custom-call($DT[$R,$E] %arg0), custom_call_target="GradientAccumulatorCreate", custom_call_has_side_effect=true, metadata={op_type="GradientAccumulatorCreate" op_name="GradientAccumulatorCreate"}, backend_config="{}"
    %call.1 = ($AT[$R,$E], $DT[$E]) call($DT[$R,$E] %arg0, $DT[$R,$E] %arg1, $AT[$R,$E] %custom-call.1, arg2), to_apply=%PipelineStage, frontend_attributes={CALL_CONFIG_TYPE="PipelineStageBackward"}, metadata={op_type="PipelineStageBackward" op_name="gradients/pipeline_stage_1/PipelineStage_grad/PipelineStageBackward"}, backend_config="{\"callConfig\":{\"type\":\"PipelineStageBackward\",\"pipelineStageConfig\":{\"stageId\":\"1\"}}}"
    %get-tuple-element.1 = $AT[$R,$E] get-tuple-element(%call.1), index=0
    %get-tuple-element.2 = $DT[$E] get-tuple-element(%call.1), index=1
    %get-tuple-element.dummy.3 = $DT[$E] get-tuple-element(%call.1), index=1
    %get-tuple-element.dummy.4 = $DT[$E] get-tuple-element(%call.1), index=1
    %get-tuple-element.dummy.5 = $DT[$E] get-tuple-element(%call.1), index=1
    %get-tuple-element.dummy.6 = $DT[$E] get-tuple-element(%call.1), index=1
    %get-tuple-element.dummy.7 = $DT[$E] get-tuple-element(%call.1), index=1
    %custom-call.2 = $AT[$R,$E] custom-call($AT[$R,$E] %get-tuple-element.1), custom_call_target="GradientAccumulatorSink", metadata={op_type="GradientAccumulatorSink" op_name="GradientAccumulatorSink"}, backend_config="{\"num_mini_batches\":$BN}"
    ROOT %call.2 = $DT[$R,$E] call(%arg0, %arg1, %custom-call.2, %get-tuple-element.2), to_apply=%WeightUpdate, frontend_attributes={CALL_CONFIG_TYPE="ResourceUpdate"}, metadata={op_type="ResourceUpdate" op_name="ResourceUpdate"}, backend_config="{\"callConfig\":{\"type\":\"ResourceUpdate\",\"resourceUpdateConfig\":{\"offloadVariables\":\"THREESTATE_ON\"}}}"
    }

    ENTRY %main {
    %arg0 = $DT[$R,$E] parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args/a"}, backend_config="{}"
    %arg1 = $DT[$R,$E] parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args/b"}, backend_config="{}"
    %arg2 = $DT[$E]    parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args/b"}, backend_config="{}"
    ROOT %call.1 = $DT[$R,$E] call(%arg0, %arg1, %arg2), to_apply=%Pipeline, frontend_attributes={CALL_CONFIG_TYPE="Pipeline"}, metadata={op_type="Pipeline" op_name="pipeline/Pipeline"}, backend_config="{\"callConfig\":{\"type\":\"Pipeline\",\"pipelineConfig\":{\"repeatCount\":\"2\",\"batchSerializationIterations\":\"$BI\",\"schedule\":\"Interleaved\"}}}"
    }
    )";
}

std::string ReplaceTemplateValues(
    std::string hlo_string, const EmbeddingsGradientOptimizerTestSpec& spec) {
  return absl::StrReplaceAll(
      hlo_string,
      {
          {"$R", std::to_string(spec.embedding_rows)},
          {"$E", std::to_string(spec.embedding_size)},
          {"$BS", std::to_string(spec.batch_size)},
          {"$BN", std::to_string(spec.num_batches)},
          {"$BI", std::to_string(spec.batch_serialization_iterations)},
          {"$DT", primitive_util::LowercasePrimitiveTypeName(spec.data_type)},
          {"$AT", primitive_util::LowercasePrimitiveTypeName(
                      spec.gradient_accumulator_type)},
      });
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
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false)
          .Run(module.get())
          .status());
  EXPECT_EQ(changed, spec.changed);
}

TEST_F(EmbeddingsGradientOptimizerTest, BatchSerializationIterations) {
  EmbeddingsGradientOptimizerTestSpec spec{100, 10, 2, 10, false, 10, F32, F32};
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           GetPipelineHloString(spec), config));

  auto module0 = module.get();
  TF_ASSERT_OK_AND_ASSIGN(bool custom_op_replaced,
                          CustomOpReplacer().Run(module0));
  EXPECT_TRUE(custom_op_replaced);

  EmbeddingsGradientOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, optimizer.Run(module.get()));
  TF_ASSERT_OK(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false)
          .Run(module.get())
          .status());
  EXPECT_EQ(changed, spec.changed);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
