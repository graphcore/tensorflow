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
#include <poplar/DebugContext.hpp>
#include <popops/Gather.hpp>
#include <popops/Sort.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla {
namespace m = match;

namespace poplarplugin {
namespace {
bool IsSimpleComparison(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());

  if (!Match(root, m::Compare(m::Parameter(0), m::Parameter(1)))) {
    return false;
  }

  switch (root->comparison_direction()) {
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
    case ComparisonDirection::kLe:
    case ComparisonDirection::kLt:
      return true;
    default:
      return false;
  }
}

bool ReverseSortOutput(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  switch (root->comparison_direction()) {
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
      return true;
    default:
      return false;
  }
}

class SortOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "SortOp");
    const HloSortInstruction* sort = Cast<HloSortInstruction>(inst);

    if (!IsSimpleComparison(inst)) {
      return xla::Unimplemented(
          "Current Sort implementation only supports GT/LT/GE/LE comparisons");
    }

    if (sort->is_stable()) {
      LOG(INFO) << "Detected a stable sort instruction " << inst->ToString()
                << ", however stable sort is not supported and an unstable "
                   "sort is performed instead.";
    }

    poplar::program::Sequence prog({}, debug_info);
    // Get the inplace input/outputs.
    TF_ASSIGN_OR_RETURN(
        TensorVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, prog, debug_info));
    if (sort->operand_count() == 1) {
      CHECK_EQ(inputs.size(), 1);
      CHECK_EQ(inputs[0].size(), 1);
      poplar::Tensor to_sort = inputs[0][0];

      popops::sortInPlace(graph, to_sort, sort->dimensions(0), prog,
                          {debug_info});

      if (ReverseSortOutput(inst)) {
        TF_ASSIGN_OR_RETURN(to_sort,
                            ReverseTensor(to_sort, sort->dimensions()));
      }

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, to_sort));
    } else if (sort->operand_count() == 2) {
      CHECK_EQ(inputs.size(), 2);
      CHECK_EQ(inputs[0].size(), 1);
      CHECK_EQ(inputs[1].size(), 1);
      poplar::Tensor key = inputs[0][0];
      poplar::Tensor value = inputs[1][0];

      popops::sortKeyValueInPlace(graph, key, value, sort->dimensions(0), prog,
                                  {debug_info});

      if (ReverseSortOutput(inst)) {
        TF_ASSIGN_OR_RETURN(key, ReverseTensor(key, sort->dimensions()));
        TF_ASSIGN_OR_RETURN(value, ReverseTensor(value, sort->dimensions()));
      }

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, key));

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, value));
    } else {
      return xla::Unimplemented(
          "Current Sort implementation only supports up to 2 operands, where "
          "as "
          "%s has %d",
          sort->name().c_str(), sort->operand_count());
    }

    return prog;
  }
};
REGISTER_HLO_OP(kSort, SortOp);
}  // anonymous namespace
}  // namespace poplarplugin
}  // namespace xla
