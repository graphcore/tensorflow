/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <sstream>
#include <stack>
#include <vector>

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

// Clone select's predicate as a module to evaluate.
StatusOr<std::unique_ptr<HloModule>> CloneSelectPredicate(
    const HloInstruction* select) {
  std::unique_ptr<HloModule> module =
      absl::make_unique<HloModule>("mask", HloModuleConfig());
  HloComputation::Builder builder(select->name());
  // Create dummy root instruction, because we must have root to create empty
  // computation. nullptr passed as root_instruction triggers an assert inside
  // HloComputation::Build.
  HloInstruction* dummy_root =
      builder.AddInstruction(HloInstruction::CreateIota(select->shape(), 0));
  HloComputation* comp = module->AddEntryComputation(builder.Build());

  // Outlined fusion don't have any parameters except the one that goes to
  // select. This means we can extract comparison predicate for evaluation.
  // Clone computation subtree starting from select's first operand.
  HloCloneContext context(module.get());
  TF_ASSIGN_OR_RETURN(
      auto eval_comp_root,
      CloneComputationSubtree(select->operand(0), comp, "", &context));
  // Change root instruction to the compare predicate clone.
  comp->set_root_instruction(context.GetInstruction(select->operand(0)),
                             /*accept_different_shape=*/true);
  // Remove dummy root instruction to keep hlo clean.
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(dummy_root));
  return module;
}

StatusOr<Literal> EvaluateSelectPredicate(const HloInstruction* select) {
  TF_ASSIGN_OR_RETURN(auto module, CloneSelectPredicate(select));
  HloEvaluator eval;
  TF_ASSIGN_OR_RETURN(Literal predicate, eval.Evaluate(*module, {}));
  return predicate;
}

class MaskOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "TensorMaskOp");
    DriverProgramSequence seq(graph, debug_info);

    TF_ASSIGN_OR_RETURN(
        DriverTensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, {debug_info}));

    DriverTensor out;
    bool is_inplace = AreInplaceOutputTensorsWritable(tensor_map, res, inst);

    VLOG(2) << "Creating mask " << inst->name() << ", inplace: " << is_inplace;
    if (is_inplace) {
      out = input;
    } else {
      out = ExtendedTensor(
          poputil::duplicate(graph, input, seq, {debug_info, "clone"}), graph);
    }
    const HloInstruction* select = inst->fused_expression_root();
    CHECK_EQ(select->opcode(), HloOpcode::kSelect);

    TF_ASSIGN_OR_RETURN(Literal mask_pred, EvaluateSelectPredicate(select));

    const Shape& shape = mask_pred.shape();
    int64_t rank = shape.rank();
    const int64_t cols = shape.dimensions(rank - 1);

    const HloInstruction* mask_const;
    bool inverse;
    if (IsConstantBroadcast(select->operand(1))) {
      inverse = true;
      mask_const = select->operand(1)->operand(0);
    } else {
      CHECK(IsConstantBroadcast(select->operand(2)));
      inverse = false;
      mask_const = select->operand(2)->operand(0);
    }
    const Literal& literal = mask_const->literal();
    TF_ASSIGN_OR_RETURN(poplar::Type poplar_type,
                        PoplarDataType(literal.shape()));
    TF_ASSIGN_OR_RETURN(
        DriverTensor mask,
        CreateConstantTensor(graph, literal, literal.shape(), poplar_type,
                             {debug_info, "mask"}));
    poputil::broadcastToMatch(mask, input.shape());

    bool copy_started = false;
    int64_t copy_started_at = 0;

    ShapeUtil::ForEachIndex(shape, [&](absl::Span<const int64_t> index) {
      bool pred = mask_pred.Get<bool>(index);
      if (inverse) {
        pred = !pred;
      }

      int64_t col = index.back();

      // Creates poplar copy with the following indices:
      // Copy(begin={i, j, ..., first}, end={i + 1, j + 1, ..., last })
      // Copy either from input or mask to the output tensor.
      auto emit_copy = [&](bool from_input, std::size_t first,
                           std::size_t last) {
        std::vector<std::size_t> begin(index.size());
        std::vector<std::size_t> end(index.size());
        for (int64_t i = 0; i < rank - 1; ++i) {
          begin[i] = index[i];
          end[i] = index[i] + 1;
        }
        begin[rank - 1] = first;
        end[rank - 1] = last;
        VLOG(3) << "Emitting copy from " << (from_input ? "input" : "mask")
                << " {" << absl::StrJoin(begin, ",") << "} {"
                << absl::StrJoin(end, ",") << "}";
        seq.add(DriverProgramCopy(
            from_input ? input.slice(begin, end) : mask.slice(begin, end),
            out.slice(begin, end), /*optimiseMemory=*/false,
            {debug_info, "Mask"}));
        VLOG(3) << "Copy created";
      };

      // Inplace version - emit only masking copies, leaving unmasked data
      // intact.
      if (is_inplace) {
        auto emit_copy_from_mask = [&](std::size_t j) {
          if (copy_started) {
            emit_copy(false, copy_started_at, j);
            copy_started = false;
          }
        };

        // Predicate evaluated to true, no masking.
        if (pred) {
          emit_copy_from_mask(col);
        } else if (!copy_started) {
          copy_started = true;
          copy_started_at = col;
        }
        // Last item on the row - emitting the copy.
        if (col + 1 >= cols) {
          emit_copy_from_mask(cols);
        }
      } else {
        // Non inplace version - emit copy each time predicate value changes,
        // store it in copy_started and on the last dimension boundaries.
        if (col == 0) {
          copy_started_at = 0;
          copy_started = pred;
        } else if (pred != copy_started) {
          emit_copy(copy_started, copy_started_at, col);
          copy_started = pred;
          copy_started_at = col;
        }
        // End of the strided copy on the last dimension boundary.
        if (col + 1 == cols) {
          emit_copy(copy_started, copy_started_at, cols);
        }
      }
      return true;
    });

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};

REGISTER_POPLAR_OP(Mask, MaskOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
