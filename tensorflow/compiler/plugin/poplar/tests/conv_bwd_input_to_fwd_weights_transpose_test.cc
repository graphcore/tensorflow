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

#include "tensorflow/compiler/plugin/poplar/driver/passes/conv_bwd_input_to_fwd_weights_transpose.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/module_flatten.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include "google/protobuf/util/message_differencer.h"

namespace xla {
namespace poplarplugin {
namespace {

int64_t GetNumConvolutionWithReverse(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(), [](const HloInstruction* inst) {
    return IsPopOpsConvolutionWithReverse(inst);
  });
}

std::vector<HloInstruction*> GetConvolutionWithReverse(
    const HloComputation* comp) {
  std::vector<HloInstruction*> insts;
  absl::c_copy_if(comp->instructions(), std::back_inserter(insts),
                  IsPopOpsConvolutionWithReverse);
  return insts;
}

int64_t GetNumWeightsTransposeChansFlipXY(const HloComputation* comp) {
  return absl::c_count_if(comp->instructions(), [](const HloInstruction* inst) {
    return IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY)(inst);
  });
}

std::vector<HloInstruction*> GetInstsWeightsTransposeChansFlipXY(
    const HloComputation* comp) {
  std::vector<HloInstruction*> insts_wt;
  absl::c_copy_if(
      comp->instructions(), std::back_inserter(insts_wt),
      [](HloInstruction* inst) {
        return IsPoplarInstruction(PoplarOp::WeightsTransposeChansFlipXY)(inst);
      });
  return insts_wt;
}

std::string GetHloString(const std::string& dims) {
  const std::string hlo = R"(
 HloModule top

 _pop_op_relu (arg_0.2: f32[1,4,4,2]) -> f32[1,4,4,2] {
   arg_0.2 = f32[1,4,4,2] parameter(0)
   constant.20.clone.1 = f32[] constant(0)
   broadcast.21.clone.1 = f32[1,4,4,2] broadcast(f32[] constant.20.clone.1), dimensions={}
   ROOT maximum.22.clone = f32[1,4,4,2] maximum(f32[1,4,4,2] arg_0.2, f32[1,4,4,2] broadcast.21.clone.1)
 }

 _pop_op_conv_scaled_inplace (arg_0.6: f32[1,1,2,2], arg_1.4: f32[1,4,4,2], arg_2.1: f32[1,4,4,2]) -> f32[1,1,2,2] {
   arg_0.6 = f32[1,1,2,2] parameter(0)
   arg_1.4 = f32[1,4,4,2] parameter(1)
   arg_2.1 = f32[1,4,4,2] parameter(2)
   convolution.63.clone = f32[1,1,2,2] convolution(f32[1,4,4,2] arg_1.4, f32[1,4,4,2] arg_2.1), window={size=4x4}, dim_labels=f01b_i01o->01bf
   constant.55.clone.1 = f32[] constant(0.1)
   broadcast.67.clone.1 = f32[1,1,2,2] broadcast(f32[] constant.55.clone.1), dimensions={}
   multiply.68.clone = f32[1,1,2,2] multiply(f32[1,1,2,2] convolution.63.clone, f32[1,1,2,2] broadcast.67.clone.1)
   ROOT subtract.69.clone = f32[1,1,2,2] subtract(f32[1,1,2,2] arg_0.6, f32[1,1,2,2] multiply.68.clone)
 }

 ENTRY top (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[2], arg3.4: f32[1,1,2,2], arg4.5: f32[2], arg5.6: f32[2], arg6.7: f32[1,1,2,2]) -> (f32[1,1,2,2], f32[1,4,4,2]) {
   arg0.1 = f32[1,4,4,2] parameter(0), metadata={op_name="XLA_Args"}
   arg6.7 = f32[1,1,2,2] parameter(6), metadata={op_name="XLA_Args"}
   convolution.11 = f32[1,4,4,2] convolution(f32[1,4,4,2] arg0.1, f32[1,1,2,2] arg6.7), window={size=1x1}, dim_labels=$DIMS
   arg5.6 = f32[2] parameter(5), control-predecessors={convolution.11}, metadata={op_name="XLA_Args"}
   arg4.5 = f32[2] parameter(4), control-predecessors={convolution.11}, metadata={op_name="XLA_Args"}
   batch-norm-training.13 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(f32[1,4,4,2] convolution.11, f32[2] arg5.6, f32[2] arg4.5), epsilon=0.001, feature_index=3
   get-tuple-element.14 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[2], f32[2]) batch-norm-training.13), index=0
   fusion.2 = f32[1,4,4,2] fusion(f32[1,4,4,2] get-tuple-element.14), kind=kCustom, calls=_pop_op_relu, backend_config="{}"
   arg3.4 = f32[1,1,2,2] parameter(3), metadata={op_name="XLA_Args"}
   convolution.23 = f32[1,4,4,2] convolution(f32[1,4,4,2] fusion.2, f32[1,1,2,2] arg3.4), window={size=1x1}, dim_labels=$DIMS
   arg2.3 = f32[2] parameter(2), control-predecessors={convolution.23}, metadata={op_name="XLA_Args"}
   arg1.2 = f32[2] parameter(1), control-predecessors={convolution.23}, metadata={op_name="XLA_Args"}
   conv-with-reverse = f32[1,4,4,2] custom-call(f32[1,4,4,2] convolution.23, f32[1,1,2,2] arg3.4), custom_call_target="ConvWithReverse", batch_group_count=1, feature_group_count=1, dim_labels=b01f_01oi->b01f, window={size=1x1}, backend_config="{}"
   fusion.6 = f32[1,1,2,2] fusion(f32[1,1,2,2] arg3.4, f32[1,4,4,2] fusion.2, f32[1,4,4,2] convolution.23), kind=kCustom, calls=_pop_op_conv_scaled_inplace, backend_config="{\"fusionConfig\":{\"window\":{\"dimensions\":[{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"},{\"size\":\"4\",\"stride\":\"1\",\"windowDilation\":\"1\",\"baseDilation\":\"1\"}]},\"dimensionNumbers\":{\"kernelOutputFeatureDimension\":\"3\",\"kernelSpatialDimensions\":[\"1\",\"2\"],\"inputBatchDimension\":\"3\",\"outputBatchDimension\":\"2\",\"outputFeatureDimension\":\"3\",\"inputSpatialDimensions\":[\"1\",\"2\"],\"outputSpatialDimensions\":[\"0\",\"1\"]},\"featureGroupCount\":\"1\",\"batchGroupCount\":\"1\"}}"
   ROOT tuple.107 = (f32[1,1,2,2], f32[1,4,4,2]) tuple(f32[1,1,2,2] fusion.6, f32[1,4,4,2] convolution.11), metadata={op_name="XLA_Retvals"}
 }
  )";
  std::string hlo_string =
      tensorflow::str_util::StringReplace(hlo, "$DIMS", dims, true);
  return hlo_string;
}

struct ConvBwdInputToFwdWeightsTransposeTestSpec {
  std::string dims;
  bool matched;
};

class ConvBwdInputToFwdWeightsTransposeTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          ConvBwdInputToFwdWeightsTransposeTestSpec> {};

INSTANTIATE_TEST_SUITE_P(
    ConvBwdInputToFwdWeightsTransposeTestCases,
    ConvBwdInputToFwdWeightsTransposeTest,
    ::testing::ValuesIn(std::vector<ConvBwdInputToFwdWeightsTransposeTestSpec>{
        {"b01f_01io->b01f", true},
        {"b01f_01oi->b01f", false},
    }));

std::ostream& operator<<(
    std::ostream& os, const ConvBwdInputToFwdWeightsTransposeTestSpec& spec) {
  return os << "{ dim_labels: " << spec.dims << ", matched: " << spec.matched
            << " }";
}

// Check that convolution with reverse was replaced by
// weights transpose chans flipxy and convolution instructions.
TEST_P(ConvBwdInputToFwdWeightsTransposeTest, DoTest) {
  auto param = GetParam();
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module0 = ParseAndReturnVerifiedModule(GetHloString(param.dims), config)
                     .ValueOrDie();
  auto* module = module0.get();

  EXPECT_TRUE(CustomOpReplacer().Run(module).ValueOrDie());

  EXPECT_EQ(GetNumConvolutionWithReverse(module->entry_computation()), 1);
  EXPECT_EQ(GetNumWeightsTransposeChansFlipXY(module->entry_computation()), 0);
  auto bwd_conv = GetConvolutionWithReverse(module->entry_computation());
  EXPECT_EQ(bwd_conv.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto bwd_dims, GetConvolutionDims(bwd_conv[0]));

  CompilerAnnotations annotations(module);

  EXPECT_TRUE(ModuleFlatten(annotations).Run(module).ValueOrDie());
  EXPECT_TRUE(ConvolutionClassifier(annotations).Run(module).ValueOrDie());
  bool matched = ConvBwdInputToFwdWeightsTranspose().Run(module).ValueOrDie();
  EXPECT_EQ(matched, param.matched);
  if (!matched) {
    EXPECT_EQ(GetNumConvolutionWithReverse(module->entry_computation()), 1);
    return;
  }
  EXPECT_EQ(GetNumConvolutionWithReverse(module->entry_computation()), 0);
  EXPECT_EQ(GetNumWeightsTransposeChansFlipXY(module->entry_computation()), 1);

  std::vector<HloInstruction*> insts_wt =
      GetInstsWeightsTransposeChansFlipXY(module->entry_computation());
  EXPECT_EQ(insts_wt.size(), 1);

  HloInstruction* inst_wt = insts_wt.at(0);
  EXPECT_EQ(inst_wt->users().size(), 1);
  HloInstruction* fwd_conv = inst_wt->users()[0];
  EXPECT_THAT(fwd_conv->operand(0)->control_successors(),
              ::testing::Contains(inst_wt));
  TF_ASSERT_OK_AND_ASSIGN(auto fwd_dims, GetConvolutionDims(fwd_conv));
  EXPECT_TRUE(
      ForwardBackwardConvolutionDimensionNumbersMatch(fwd_dims, bwd_dims));

  const std::vector<HloInstruction*>& users_of_wt = inst_wt->users();
  EXPECT_EQ(users_of_wt.size(), 1);
  EXPECT_EQ(users_of_wt.at(0)->opcode(), HloOpcode::kConvolution);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
