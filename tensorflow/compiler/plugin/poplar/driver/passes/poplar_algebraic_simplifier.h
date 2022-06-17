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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_ALGEBRAIC_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_ALGEBRAIC_SIMPLIFIER_H_

#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/poplar_algebraic_simplifier/poplar_algebraic_simplifier_dot.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

// A pass which performs algebraic simplifications.
class PoplarAlgebraicSimplifier : public HloModulePass {
 public:
  explicit PoplarAlgebraicSimplifier(
      poplarplugin::IpuOptions_IpuAlgebraicSimplifierConfig config = {});
  ~PoplarAlgebraicSimplifier() override = default;
  absl::string_view name() const override {
    return "poplar-algebraic-simplifier";
  }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Create constant from literal with tiles and element size updated in the
  // constant's layout.
  std::unique_ptr<HloInstruction> CreateConstantWithLayoutUpdated(
      Literal literal) {
    auto constant = HloInstruction::CreateConstant(std::move(literal));
    UpdateLayout(constant->mutable_shape());
    return constant;
  }

 private:
  const poplarplugin::IpuOptions_IpuAlgebraicSimplifierConfig config_;
};

// AlgebraicSimplifierVisitor traverses the HLO computation and reduces certain
// algebraic expressions to simplified forms. Note: This only supports
// simplifications that simply look at the operands of an instruction. For the
// more general case a worklist based approach would be needed.
class AlgebraicSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AlgebraicSimplifierVisitor(
      PoplarAlgebraicSimplifier* simplifier,
      poplarplugin::IpuOptions_IpuAlgebraicSimplifierConfig config)
      : simplifier_(simplifier), config_(std::move(config)) {}

  friend StatusOr<HloInstruction*>
  xla::poplarplugin::algebraic_simplifier::dot::OptimizeDotOfConcatHelper(
      AlgebraicSimplifierVisitor* visitor, const HloInstruction& dot,
      HloInstruction* lhs, int64_t lhs_contracting_dim, HloInstruction* rhs,
      int64_t rhs_contracting_dim, bool swapped);

  friend StatusOr<HloInstruction*>
  xla::poplarplugin::algebraic_simplifier::dot::OptimizeDotOfGather(
      AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

  friend StatusOr<HloInstruction*> xla::poplarplugin::algebraic_simplifier::
      dot::OptimizeDotOfReorderContractingDims(
          AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

  friend StatusOr<HloInstruction*>
  xla::poplarplugin::algebraic_simplifier::dot::OptimizeDotStrengthReduction(
      AlgebraicSimplifierVisitor* visitor, HloInstruction* dot);

  Status HandleAdd(HloInstruction* add) override;

  Status HandleAllReduce(HloInstruction* all_reduce) override;

  Status HandleAnd(HloInstruction* logical_and) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBitcastConvert(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleCompare(HloInstruction* compare) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConvert(HloInstruction* convert) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleIota(HloInstruction* instruction) override;

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleDivide(HloInstruction* divide) override;

  Status HandleDot(HloInstruction* dot) override;

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMaximum(HloInstruction* maximum) override;

  Status HandleMinimum(HloInstruction* minimum) override;

  Status HandleClamp(HloInstruction* clamp) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandleNegate(HloInstruction* negate) override;

  Status HandleNot(HloInstruction* logical_not) override;

  Status HandleOr(HloInstruction* logical_or) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power) override;

  Status HandleRemainder(HloInstruction* remainder) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleScatter(HloInstruction* scatter) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMap(HloInstruction* map) override;

  Status HandleCustomCall(HloInstruction* custom_call) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation, PoplarAlgebraicSimplifier* simplifier);

 private:
  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64_t> dims,
                            PrimitiveType type);

  // Converts to primitive type if the input hlo is not that type, otherwise
  // returns the original hlo.
  HloInstruction* AsType(HloInstruction* hlo, const PrimitiveType element_type);

  static HloInstruction* PreserveFrontendAttributesIfNeeded(
      HloInstruction* new_inst, const HloInstruction* old_inst);

  static StatusOr<HloInstruction*> PreserveFrontendAttributesIfNeeded(
      StatusOr<HloInstruction*> new_inst, const HloInstruction* old_inst);

  // Removes degenerate dimension from dot.
  StatusOr<bool> RemoveDegenerateDimensionFromDot(HloInstruction* dot);

  // Replace old instruction with new instruction if old and new instructions
  // have the same shape. Updates uses and root instruction. Returns whether a
  // replacement was made.
  bool ReplaceInstructionIfSameShape(HloInstruction* old_instruction,
                                     HloInstruction* new_instruction);

  // A Broadcast that feeds an element-wise operation with a unique non-scalar
  // operand can sink to after the operation.
  StatusOr<bool> TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
      HloInstruction* broadcast);

  // Dot product helper functions.
  StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      const HloInstruction& dot, HloInstruction* lhs,
      int64_t lhs_contracting_dim, HloInstruction* rhs,
      int64_t rhs_contracting_dim, bool swapped);
  StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
      HloInstruction* dot);

  HloComputation* GetOrCreateScalarAddComputation(PrimitiveType type);

  // Tries to fold a kPad in the input or filter into the convolution
  // instruction's window.
  StatusOr<bool> FoldConvFilterPad(HloInstruction* convolution);

  // Tries to simplify a slice where the result of the slice is a scalar.
  StatusOr<bool> TrySimplifyScalarSlice(HloInstruction* slice);

  // Tries to convert slice(reshape(X)) into reshape(slice(X))
  StatusOr<bool> TryToReorderSliceAndReshape(HloInstruction* slice);

  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // Tries to replace concatenate of same slices to broadcast.
  bool TrySimplifyConcatenateOfSameSlices(HloInstruction* concatenate);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // Whether algebraic simplification has occurred.
  bool changed_ = false;

  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;

  PoplarAlgebraicSimplifier* simplifier_ = nullptr;

  const poplarplugin::IpuOptions_IpuAlgebraicSimplifierConfig config_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPLAR_ALGEBRAIC_SIMPLIFIER_H_
