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

#include "tensorflow/compiler/plugin/poplar/driver/passes/matmul_combiner.h"

#include <functional>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matmul_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("matmul_matmul_shared_lhs"),
    PatternMetaTarget(0),
    PatternInputs({2, 3, 4}),
    PatternOutputs({0, 1}),
    Pattern({
      {HloOpcode::kDot, NodeOperands({2, 3})},
      {HloOpcode::kDot, NodeOperands({2, 4})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),
  HloMatcherPattern(
    PatternType("matmul_matmul_shared_rhs"),
    PatternMetaTarget(0),
    PatternInputs({2, 3, 4}),
    PatternOutputs({0, 1}),
    Pattern({
      {HloOpcode::kDot, NodeOperands({2, 3})},
      {HloOpcode::kDot, NodeOperands({4, 3})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })
  )
};
// clang-format on

// Reshape to [Batch, Contracting, N]
HloInstruction* PrepareRHS(HloInstruction* rhs,
                           const DotDimensionNumbers& dot_dims,
                           HloComputation* computation) {
  const auto shape = rhs->shape();
  std::vector<int64> permutations;
  Shape shuffled_shape, packed_shape;
  std::tie(packed_shape, shuffled_shape, permutations) =
      RightMatMulPrepare(shape, dot_dims);

  HloInstruction* shuffled = computation->AddInstruction(
      HloInstruction::CreateTranspose(shuffled_shape, rhs, permutations));
  rhs->SetupDerivedInstruction(shuffled);

  HloInstruction* reshaped = computation->AddInstruction(
      HloInstruction::CreateReshape(packed_shape, shuffled));
  rhs->SetupDerivedInstruction(reshaped);
  return reshaped;
}

// Reshape to [Batch, M, Contracting]
HloInstruction* PrepareLHS(HloInstruction* lhs,
                           const DotDimensionNumbers& dot_dims,
                           HloComputation* computation) {
  const auto shape = lhs->shape();
  std::vector<int64> permutations;
  Shape shuffled_shape, packed_shape;
  std::tie(packed_shape, shuffled_shape, permutations) =
      LeftMatMulPrepare(shape, dot_dims);

  HloInstruction* shuffled = computation->AddInstruction(
      HloInstruction::CreateTranspose(shuffled_shape, lhs, permutations));
  lhs->SetupDerivedInstruction(shuffled);

  HloInstruction* reshaped = computation->AddInstruction(
      HloInstruction::CreateReshape(packed_shape, shuffled));
  lhs->SetupDerivedInstruction(reshaped);
  return reshaped;
}

}  // namespace

MatmulCombiner::MatmulCombiner(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

bool MatmulCombiner::HandleMatch(HloMatcherMatched& match,
                                 const absl::optional<int64> sharding_device) {
  auto pattern = patterns_[match.pattern_idx];
  HloComputation* computation = match.computation;

  HloDotInstruction* matmul1 =
      Cast<HloDotInstruction>(match.instruction_mapping.at(0));
  HloDotInstruction* matmul2 =
      Cast<HloDotInstruction>(match.instruction_mapping.at(1));

  const DotDimensionNumbers dot_dims1 = matmul1->dot_dimension_numbers();
  const DotDimensionNumbers dot_dims2 = matmul2->dot_dimension_numbers();

  if (dot_dims1.SerializeAsString() != dot_dims2.SerializeAsString()) {
    VLOG(1) << "Can't fuse two matmuls with different dot_dimension_numbers";
    return false;
  }
  if (matmul1->precision_config().SerializeAsString() !=
      matmul2->precision_config().SerializeAsString()) {
    VLOG(1) << "Can't fuse two matmuls with different precision_config";
    return false;
  }

  if (matmul1->frontend_attributes().SerializeAsString() !=
      matmul2->frontend_attributes().SerializeAsString()) {
    VLOG(1) << "Can't fuse two matmuls with different frontend_attributes";
    return false;
  }

  if (matmul1->operand(0) == matmul2->operand(0)) {
    // Shared lhs
    HloInstruction* lhs = match.instruction_mapping.at(2);
    HloInstruction* rhs1 = match.instruction_mapping.at(3);
    HloInstruction* rhs2 = match.instruction_mapping.at(4);

    // Reshape to [Batch, Contracting, N]
    rhs1 = PrepareRHS(rhs1, dot_dims1, computation);
    rhs2 = PrepareRHS(rhs2, dot_dims2, computation);
    // Reshape to [Batch, M, Contracting]
    lhs = PrepareLHS(lhs, dot_dims1, computation);

    const int64 output_width1 = rhs1->shape().dimensions(2);
    const int64 output_width2 = rhs2->shape().dimensions(2);
    const int64 output_width = output_width1 + output_width2;
    const int64 output_height = lhs->shape().dimensions(1);
    const int64 output_batch = lhs->shape().dimensions(0);

    std::vector<HloInstruction*> rhs = {rhs1, rhs2};
    HloInstruction* new_rhs =
        computation->AddInstruction(HloInstruction::CreateConcatenate(
            GetConcatenatedShape(rhs, 2), rhs, 2));
    rhs1->SetupDerivedInstruction(new_rhs);

    Shape matmul_shape =
        ShapeUtil::MakeShape(matmul1->shape().element_type(),
                             {output_batch, output_height, output_width});
    DotDimensionNumbers new_dot_dim;
    new_dot_dim.add_lhs_contracting_dimensions(2);
    new_dot_dim.add_rhs_contracting_dimensions(1);
    new_dot_dim.add_lhs_batch_dimensions(0);
    new_dot_dim.add_rhs_batch_dimensions(0);

    HloInstruction* new_matmul = computation->AddInstruction(
        HloInstruction::CreateDot(matmul_shape, lhs, new_rhs, new_dot_dim,
                                  matmul1->precision_config()));
    matmul1->SetupDerivedInstruction(new_matmul);

    HloInstruction* output1 =
        computation->AddInstruction(HloInstruction::CreateSlice(
            matmul1->shape(), new_matmul, {0, 0, 0},
            {output_batch, output_height, output_width1}, {1, 1, 1}));
    computation->ReplaceInstruction(matmul1, output1);

    HloInstruction* output2 =
        computation->AddInstruction(HloInstruction::CreateSlice(
            matmul2->shape(), new_matmul, {0, 0, output_width1},
            new_matmul->shape().dimensions(), {1, 1, 1}));
    computation->ReplaceInstruction(matmul2, output2);
  } else {
    // Shared rhs
    HloInstruction* lhs1 = match.instruction_mapping.at(2);
    HloInstruction* rhs = match.instruction_mapping.at(3);
    HloInstruction* lhs2 = match.instruction_mapping.at(4);

    // Reshape to [Batch, Contracting, N]
    rhs = PrepareRHS(rhs, dot_dims1, computation);
    // Reshape to [Batch, M, Contracting]
    lhs1 = PrepareLHS(lhs1, dot_dims1, computation);
    lhs2 = PrepareLHS(lhs2, dot_dims2, computation);

    const int64 output_width = rhs->shape().dimensions(2);
    const int64 output_height1 = lhs1->shape().dimensions(1);
    const int64 output_height2 = lhs2->shape().dimensions(1);
    const int64 output_height = output_height1 + output_height2;
    const int64 output_batch = rhs->shape().dimensions(0);

    std::vector<HloInstruction*> lhs = {lhs1, lhs2};
    HloInstruction* new_lhs =
        computation->AddInstruction(HloInstruction::CreateConcatenate(
            GetConcatenatedShape(lhs, 1), lhs, 1));
    lhs1->SetupDerivedInstruction(new_lhs);

    Shape matmul_shape =
        ShapeUtil::MakeShape(matmul1->shape().element_type(),
                             {output_batch, output_height, output_width});
    DotDimensionNumbers new_dot_dim;
    new_dot_dim.add_lhs_contracting_dimensions(2);
    new_dot_dim.add_rhs_contracting_dimensions(1);
    new_dot_dim.add_lhs_batch_dimensions(0);
    new_dot_dim.add_rhs_batch_dimensions(0);

    HloInstruction* new_matmul = computation->AddInstruction(
        HloInstruction::CreateDot(matmul_shape, new_lhs, rhs, new_dot_dim,
                                  matmul1->precision_config()));
    matmul1->SetupDerivedInstruction(new_matmul);

    HloInstruction* output1 =
        computation->AddInstruction(HloInstruction::CreateSlice(
            matmul1->shape(), new_matmul, {0, 0, 0},
            {output_batch, output_height1, output_width}, {1, 1, 1}));
    computation->ReplaceInstruction(matmul1, output1);

    HloInstruction* output2 =
        computation->AddInstruction(HloInstruction::CreateSlice(
            matmul2->shape(), new_matmul, {0, output_height1, 0},
            new_matmul->shape().dimensions(), {1, 1, 1}));
    computation->ReplaceInstruction(matmul2, output2);
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
