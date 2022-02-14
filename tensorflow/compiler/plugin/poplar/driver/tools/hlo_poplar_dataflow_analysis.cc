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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace poplarplugin {

using ComputationInputBufferSets = std::vector<InstructionPoplarBufferSet*>;

class DataflowAnalysisBufferVisitor : public DfsHloVisitorWithDefault {
 public:
  DataflowAnalysisBufferVisitor(
      HloPoplarDataflowAnalysis* analysis,
      const CompilerAnnotations* annotations,
      ComputationInputBufferSets input_buffer_sets = {},
      bool is_entry_computation = false)
      : analysis_(analysis),
        annotations_(annotations),
        input_buffer_sets_(input_buffer_sets),
        is_entry_computation_(is_entry_computation) {}

  Status DefaultAction(HloInstruction* inst) override {
    return UnimplementedStrCat("Trying to obtain buffer information for ",
                               inst->ToString(), " which is not implemented.");
  }

  Status Postprocess(HloInstruction* inst) override {
    const auto& instruction_set = analysis_->GetInstructionBufferSet(inst);

    // Check each shape is defined at a leaf.
    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
      if (instruction_set.GetOutputBufferSet(indexed_shape.index).size() == 0) {
        return InternalErrorStrCat("Cannot find buffer outputs for ",
                                   inst->ToString(), " output index ",
                                   indexed_shape.index.ToString());
      }
    }

    return Status::OK();
  }

  Status HandleParameter(HloInstruction* inst) override {
    // Create a buffer for each shape.

    const auto shapes = ShapeUtil::GetLeafShapes(inst->shape());
    // If shapes is empty then won't hit for loop below so special case
    // this and add an empty buffer.
    if (shapes.size() == 0) {
      analysis_->SetInstructionBufferSet(
          inst, InstructionPoplarBufferSet(inst->shape()));
    }
    for (auto& indexed_shape : shapes) {
      HloPoplarBufferSet buffer_set;
      if (input_buffer_sets_.size()) {
        // A buffer set was passed in at the callsite.
        CHECK_LT(inst->parameter_number(), input_buffer_sets_.size());
        buffer_set =
            input_buffer_sets_[inst->parameter_number()]->GetOutputBufferSet(
                indexed_shape.index);
      } else {
        // A buffer was not passed in at the callsite - create one.

        // Check whether this buffer is a remote buffer.
        const bool is_remote_buffer_parameter =
            is_entry_computation_ && annotations_
                ? ContainsKey(annotations_->remote_parameter_infos,
                              RemoteParameterInfo(inst->parameter_number()))
                : false;

        const BufferLocality locality = is_remote_buffer_parameter
                                            ? BufferLocality::kRemoteMemory
                                            : BufferLocality::kDeviceMemory;

        HloPoplarBuffer* buffer =
            analysis_->NewHloPoplarBuffer(inst, indexed_shape.index, locality);
        buffer_set = HloPoplarBufferSet({buffer});
      }

      analysis_->SetInstructionBufferSetOutput(inst, indexed_shape.index,
                                               buffer_set);
    }
    return Status::OK();
  }

  Status HandleGetTupleElement(HloInstruction* inst) override {
    const int64 tuple_index = inst->tuple_index();
    HloInstruction* input = inst->mutable_operand(0);
    // Forward the buffers from the right shape tree index.
    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
      ShapeIndex input_index = indexed_shape.index;
      input_index.push_front(tuple_index);

      const HloPoplarPosition input_position{input, input_index};
      const HloPoplarPosition output_position{inst, indexed_shape.index};

      HloPoplarBufferSet buffer_set = analysis_->GetBufferSet(input_position);
      buffer_set.AddNewBufferUse(BufferUseKind::USE_ALIAS_READ_ONLY);

      analysis_->SetInstructionBufferSetOutput(inst, indexed_shape.index,
                                               buffer_set);

      VLOG(3) << "Forwarding buffer set " << buffer_set << " from "
              << input_position << " to " << output_position;
    }
    return Status::OK();
  }

  Status HandleMap(HloInstruction* inst) override {
    // Maps are inplace on all the inputs, and the output buffers are derrived
    // from that.
    return HandleInplaceVisitor(inst, inst->to_apply());
  }

  Status HandleFusion(HloInstruction* inst) override {
    if (!IsPopOpsFusion(inst)) {
      // A non poplibs fusion is inplace on all operands and the output buffers
      // are derrived from the evaluated fusion computation.
      return HandleInplaceVisitor(inst, inst->fused_instructions_computation());
    }
    // Get the input to output aliasing description.
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        inst->backend_config<PoplarBackendConfig>());
    const auto& fusion_config = backend_config.fusion_config();
    const auto& inplace_descriptions_proto =
        fusion_config.inplace_descriptions();

    InstructionPoplarBufferSet instruction_set(inst->shape());
    // Forward the buffer sets accordingly.
    for (const auto& inplace_description_proto : inplace_descriptions_proto) {
      const auto inplace_description =
          HloPoplarUseDescription::FromProto(inplace_description_proto);
      const HloInstruction* operand =
          inst->operand(inplace_description.operand_number());
      const auto& operand_buffer_set =
          analysis_->GetBufferSet(operand, inplace_description.operand_index());
      const BufferUseKind kind = inplace_description.kind();

      VLOG(3) << "Forwarding buffer set " << operand_buffer_set << " from "
              << operand->ToString() << " (" << BufferUseKind_Name(kind)
              << ") to " << inst->ToString();

      instruction_set.SetOutputToBufferSetUnion(
          inplace_description.output_index(), operand_buffer_set, kind);
    }

    // Generate buffers for all the outputs which don't have aliasing.
    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
      if (instruction_set.GetOutputBufferSet(indexed_shape.index).size() == 0) {
        HloPoplarBuffer* buffer = analysis_->NewHloPoplarBuffer(
            inst, indexed_shape.index,
            /*locality=*/BufferLocality::kDeviceMemory);
        instruction_set.SetOutputBufferSet(indexed_shape.index,
                                           HloPoplarBufferSet({buffer}));
      }
    }

    analysis_->SetInstructionBufferSet(inst, instruction_set);
    return Status::OK();
  }

  Status HandleConditional(HloInstruction* inst) override {
    // Visit each branch.
    for (HloComputation* branch_comp : inst->called_computations()) {
      DataflowAnalysisBufferVisitor visitor(analysis_, annotations_);
      TF_RETURN_IF_ERROR(branch_comp->Accept(&visitor));
    }

    return HandleNotInplace(inst);
  }

  Status HandleCustomCall(HloInstruction* inst) override {
    if (!IsPoplibsHloCustomOp(inst)) {
      if (HloPoplarInstructionFactory::IsCreatable(
              Cast<HloCustomCallInstruction>(inst))) {
        return FailedPrecondition("Run custom op replacer before pass");
      }
      // Can't tell what a generic custom call is up (I think) to so assume
      // not in place
      return HandleNotInplace(inst);
    }

    InstructionPoplarBufferSet instruction_set(inst->shape());
    auto poplar_inst = Cast<HloPoplarInstruction>(inst);
    // First populate the uses.
    for (const auto& use_description : poplar_inst->GetUseDescriptions()) {
      const HloInstruction* operand =
          inst->operand(use_description.operand_number());
      const auto& operand_buffer_set =
          analysis_->GetBufferSet(operand, use_description.operand_index());
      const BufferUseKind kind = use_description.kind();

      VLOG(3) << "Forwarding buffer set " << operand_buffer_set << " from "
              << operand->ToString() << " (" << BufferUseKind_Name(kind)
              << ") to " << inst->ToString();

      instruction_set.SetOutputToBufferSetUnion(use_description.output_index(),
                                                operand_buffer_set, kind);
    }

    // Then create the output buffers.
    for (const auto& buffer_description :
         poplar_inst->GetBufferDescriptions()) {
      HloPoplarBuffer* buffer =
          analysis_->NewHloPoplarBuffer(inst, buffer_description.output_index(),
                                        buffer_description.locality());

      instruction_set.SetOutputToBufferSetUnion(
          buffer_description.output_index(), HloPoplarBufferSet({buffer}),
          BufferUseKind::USE_NO_ALIAS);
    }
    analysis_->SetInstructionBufferSet(inst, instruction_set);

    return Status::OK();
  }

  Status HandleCall(HloInstruction* inst) override {
    HloComputation* comp = inst->to_apply();
    if (IsRepeatLoop(inst) || IsPipelineOp(inst)) {
      // These custom loop like constructions have an implicit copy from the
      // root instruction of the body computation to the inputs - this means
      // that the instruction outputs is just its inputs.

      // Get the input buffer sets.
      ComputationInputBufferSets input_buffer_sets;
      for (HloInstruction* operand : inst->operands()) {
        input_buffer_sets.push_back(
            &analysis_->GetInstructionBufferSet(operand));
      }

      // Call the loop with inplace inputs.
      DataflowAnalysisBufferVisitor visitor(analysis_, annotations_,
                                            input_buffer_sets);
      TF_RETURN_IF_ERROR(comp->Accept(&visitor));

      if (inst->operand_count() == 1 &&
          inst->shape() == inst->operand(0)->shape()) {
        const HloInstruction* operand = inst->operand(0);
        const auto& instruction_set =
            analysis_->GetInstructionBufferSet(operand);
        VLOG(3) << "Forwarding instruction buffer set " << instruction_set
                << " from " << operand->ToString() << " to "
                << inst->ToString();
        analysis_->SetInstructionBufferSet(inst, instruction_set);
      } else {
        return HandleInplaceForwardAllBuffers(
            inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_WRITE);
      }
      return Status::OK();

    } else if (IsResourceUpdate(inst) || IsAnyPipelineStageOp(inst)) {
      return HandleInplaceVisitor(inst, comp);
    } else {
      int64 num_inplace_operands = 0;
      int64 num_inplace_outputs = 0;
      if (IsFunction(inst)) {
        // Functions are inplace on remote buffer inputs.
        // Assume that the first "num_modified_remote_buffers" inputs are remote
        // buffers which are modified and they are also the first
        // "num_modified_remote_buffers" outputs.
        // Assume that the next "num_unmodified_remote_buffers" inputs are
        // remote buffers which are only loaded.
        const int64 num_modified_remote_buffers =
            GetFunctionNumberModifiedRemoteBufferInputs(inst);
        const int64 num_unmodified_remote_buffers =
            GetFunctionNumberUnmodifiedRemoteBufferInputs(inst);
        num_inplace_operands =
            num_modified_remote_buffers + num_unmodified_remote_buffers;
        num_inplace_outputs = num_modified_remote_buffers;
        VLOG(3) << "Function with " << num_modified_remote_buffers
                << " modified remote buffers, " << num_unmodified_remote_buffers
                << " unmodified remote buffers and " << num_inplace_operands
                << " inplace operands.";
      }

      CHECK_GE(num_inplace_outputs, 0);
      if (num_inplace_outputs) {
        CHECK(inst->shape().IsTuple());
      }

      // Create the computation parameter sets given the inplace operands - pass
      // the buffers through for inplace operands and create new buffers for
      // non-inplace operands.
      std::vector<InstructionPoplarBufferSet> parameter_sets;
      for (int64 i = 0; i != inst->operand_count(); ++i) {
        const HloInstruction* operand = inst->operand(i);
        HloInstruction* parameter = comp->parameter_instruction(i);
        if (i < num_inplace_operands) {
          parameter_sets.push_back(analysis_->GetInstructionBufferSet(operand));
        } else {
          InstructionPoplarBufferSet operand_set(parameter->shape());
          for (auto& indexed_shape :
               ShapeUtil::GetLeafShapes(parameter->shape())) {
            HloPoplarBuffer* buffer = analysis_->NewHloPoplarBuffer(
                parameter, indexed_shape.index,
                /*locality=*/BufferLocality::kDeviceMemory);
            operand_set.SetOutputBufferSet(indexed_shape.index,
                                           HloPoplarBufferSet({buffer}));
          }
          parameter_sets.push_back(operand_set);
        }
      }

      ComputationInputBufferSets input_buffer_sets(inst->operand_count());
      for (int64 i = 0; i != inst->operand_count(); ++i) {
        input_buffer_sets[i] = &parameter_sets[i];
      }

      DataflowAnalysisBufferVisitor visitor(analysis_, annotations_,
                                            input_buffer_sets);

      // Run the visitor on the mapped computation.
      TF_RETURN_IF_ERROR(comp->Accept(&visitor));

      // Create the computation outputs given the inplace outputs - pass
      // the buffers through for inplace outputs and create new buffers for
      // non-inplace outputs.
      for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
        const ShapeIndex& index = indexed_shape.index;
        HloPoplarBufferSet buffer_set;
        if (index[0] < num_inplace_outputs) {
          buffer_set = analysis_->GetBufferSet(comp->root_instruction(), index);
        } else {
          HloPoplarBuffer* buffer = analysis_->NewHloPoplarBuffer(
              inst, index,
              /*locality=*/BufferLocality::kDeviceMemory);
          buffer_set = HloPoplarBufferSet({buffer});
        }
        analysis_->SetInstructionBufferSetOutput(inst, index, buffer_set);
      }
      return Status::OK();
    }

    return FailedPrecondition("Unhandled kCall case.");
  }

  Status HandleWhile(HloInstruction* inst) override {
    CHECK_EQ(inst->operand_count(), 1);
    HloInstruction* operand = inst->mutable_operand(0);
    auto& instruction_set = analysis_->GetInstructionBufferSet(operand);

    // For a while loop:
    // * The condition computation is not inplace on any inputs
    // * The body computation is inplace on the inputs
    // * There is an implicit copy from the root instruction of the body
    // computation to the inputs.
    {
      HloComputation* condition_comp = inst->while_condition();
      DataflowAnalysisBufferVisitor visitor(analysis_, annotations_);
      TF_RETURN_IF_ERROR(condition_comp->Accept(&visitor));
    }
    {
      HloComputation* body_comp = inst->while_body();
      ComputationInputBufferSets input_buffer_sets = {&instruction_set};
      DataflowAnalysisBufferVisitor visitor(analysis_, annotations_,
                                            input_buffer_sets);
      TF_RETURN_IF_ERROR(body_comp->Accept(&visitor));
    }

    VLOG(3) << "Forwarding instruction buffer set " << instruction_set
            << " from " << operand->ToString() << " to " << inst->ToString();
    analysis_->SetInstructionBufferSet(inst, instruction_set);
    return Status::OK();
  }

  Status HandlePad(HloInstruction* inst) override {
    return HandleForwardsUnionOfAllOperands(
        inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_ONLY);
  }

  Status HandleConcatenate(HloInstruction* inst) override {
    return HandleForwardsUnionOfAllOperands(
        inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_ONLY);
  }

  Status HandleAllReduce(HloInstruction* inst) override {
    return HandleInplaceForwardAllBuffers(
        inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_WRITE);
  }

  Status HandleSort(HloInstruction* inst) override {
    return HandleInplaceForwardAllBuffers(
        inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_WRITE);
  }

  Status HandleTuple(HloInstruction* inst) override {
    return HandleInplaceForwardAllBuffers(
        inst, /*kind=*/BufferUseKind::USE_ALIAS_READ_ONLY);
  }

#define HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(Name)                             \
  Status Name(HloInstruction* inst) override {                                \
    return HandleSimpleInplace(inst,                                          \
                               /*kind=*/BufferUseKind::USE_ALIAS_READ_WRITE); \
  }

  // Unary Elementwise Ops - inplace on 0th operand.
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleAbs);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleCeil);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleCbrt);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleClz);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleCos);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleExp);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleExpm1);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleFloor);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleLog1p);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleLog);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleLogistic);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleNegate);

  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleNot);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandlePopulationCount);

  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleReal);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleRound);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleRsqrt);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleSign);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleSin);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleSqrt);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleTanh);
  // Binary Elementwise ops - inplace on 0th operand.
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleAdd);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleAtan2);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleComplex);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleDivide);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleMaximum);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleMinimum);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleMultiply);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandlePower);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleRemainder);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleSubtract);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleShiftLeft);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleShiftRightArithmetic);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleShiftRightLogical);
  // These ops are implemented as inplace ops on operand 0 as well.
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleDynamicUpdateSlice);
  HANDLE_AS_SIMPLE_INPLACE_READ_WRITE(HandleScatter);
#undef HANDLE_AS_SIMPLE_INPLACE_READ_WRITE

#define HANDLE_LOGICAL_BINARY_ELEMENTWISE(Name)  \
  Status Name(HloInstruction* inst) override {   \
    return HandleLogicalBinaryElementwise(inst); \
  }

  HANDLE_LOGICAL_BINARY_ELEMENTWISE(HandleAnd);
  HANDLE_LOGICAL_BINARY_ELEMENTWISE(HandleOr);
  HANDLE_LOGICAL_BINARY_ELEMENTWISE(HandleXor);

#undef HANDLE_LOGICAL_BINARY_ELEMENTWISE

#define HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(Name)                             \
  Status Name(HloInstruction* inst) override {                               \
    return HandleSimpleInplace(inst,                                         \
                               /*kind=*/BufferUseKind::USE_ALIAS_READ_ONLY); \
  }
  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleAddDependency);
  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleBitcastConvert);
  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleBroadcast);

  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleReshape);
  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleReverse);

  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleSlice);
  HANDLE_AS_SIMPLE_INPLACE_READ_ONLY(HandleTranspose);

#undef HANDLE_AS_SIMPLE_INPLACE_READ_ONLY

#define HANDLE_AS_NOT_INPLACE(Name) \
  Status Name(HloInstruction* inst) override { return HandleNotInplace(inst); }

  HANDLE_AS_NOT_INPLACE(HandleAfterAll);
  HANDLE_AS_NOT_INPLACE(HandleAllToAll);
  HANDLE_AS_NOT_INPLACE(HandleBatchNormGrad);
  HANDLE_AS_NOT_INPLACE(HandleBatchNormInference);
  HANDLE_AS_NOT_INPLACE(HandleBatchNormTraining);
  HANDLE_AS_NOT_INPLACE(HandleCholesky);
  HANDLE_AS_NOT_INPLACE(HandleCompare);
  HANDLE_AS_NOT_INPLACE(HandleConstant);
  HANDLE_AS_NOT_INPLACE(HandleConvert);
  HANDLE_AS_NOT_INPLACE(HandleConvolution);
  HANDLE_AS_NOT_INPLACE(HandleDomain);
  HANDLE_AS_NOT_INPLACE(HandleDot);
  HANDLE_AS_NOT_INPLACE(HandleDynamicSlice);
  HANDLE_AS_NOT_INPLACE(HandleGather);
  HANDLE_AS_NOT_INPLACE(HandleInfeed);
  HANDLE_AS_NOT_INPLACE(HandleIota);
  HANDLE_AS_NOT_INPLACE(HandleIsFinite);

  HANDLE_AS_NOT_INPLACE(HandleOutfeed);
  HANDLE_AS_NOT_INPLACE(HandleReduce);

  HANDLE_AS_NOT_INPLACE(HandleReducePrecision);
  HANDLE_AS_NOT_INPLACE(HandleReduceWindow);
  HANDLE_AS_NOT_INPLACE(HandleRng);
  HANDLE_AS_NOT_INPLACE(HandleReplicaId);
  HANDLE_AS_NOT_INPLACE(HandleSelectAndScatter);
  HANDLE_AS_NOT_INPLACE(HandleTriangularSolve);
  HANDLE_AS_NOT_INPLACE(HandleTupleSelect);
  // TODO(T20398): Clamp and Select could be inplace on operand index 1.
  HANDLE_AS_NOT_INPLACE(HandleClamp);
  HANDLE_AS_NOT_INPLACE(HandleSelect);

#undef HANDLE_AS_NOT_INPLACE

 private:
  // Handles instructions which are inplace on all the inputs, call the
  // "comp" computation and the output is the buffer outputs from the root
  // instruction of that computation.
  Status HandleInplaceVisitor(HloInstruction* inst, HloComputation* comp) {
    ComputationInputBufferSets map_input_buffer_sets(inst->operand_count());
    for (int64 i = 0; i != inst->operand_count(); ++i) {
      map_input_buffer_sets[i] =
          &analysis_->GetInstructionBufferSet(inst->operand(i));
    }

    DataflowAnalysisBufferVisitor visitor(analysis_, annotations_,
                                          map_input_buffer_sets);
    HloInstruction* root = comp->root_instruction();

    // Run the visitor on the mapped computation.
    TF_RETURN_IF_ERROR(comp->Accept(&visitor));

    // Get the output buffers.
    auto& root_set = analysis_->GetInstructionBufferSet(root);

    VLOG(3) << "Forwarding instruction buffer set " << root_set << " from "
            << root->ToString() << " to " << inst->ToString();
    analysis_->SetInstructionBufferSet(inst, root_set);
    return Status::OK();
  }

  // Inplace read-only which forward all the buffer sets from operands to an
  // array shaped output.
  Status HandleForwardsUnionOfAllOperands(HloInstruction* inst,
                                          BufferUseKind use_kind) {
    CHECK_GT(inst->operand_count(), 0);
    CHECK(inst->shape().IsArray());

    std::vector<const HloPoplarBufferSet*> buffer_sets;
    for (const HloInstruction* operand : inst->operands()) {
      CHECK(operand->shape().IsArray());
      buffer_sets.push_back(&analysis_->GetBufferSet(operand, {}));
    }
    HloPoplarBufferSet buffer_set;
    buffer_set.AssignUnionOf(buffer_sets, use_kind);

    VLOG(3) << "Setting a union of buffer sets " << buffer_set << " to "
            << inst->ToString();

    analysis_->SetInstructionBufferSetOutput(inst, ShapeIndex{}, buffer_set);
    return Status::OK();
  }

  // Inplace instructions where for each operand at index i, all the
  // buffers for operand i are forward to the corresponding output index with
  // prefix "i".
  Status HandleInplaceForwardAllBuffers(HloInstruction* inst,
                                        BufferUseKind kind) {
    const Shape& shape = inst->shape();
    bool is_tuple = shape.IsTuple();
    if (is_tuple && ShapeUtil::IsEmptyTuple(shape)) {
      analysis_->SetInstructionBufferSet(inst,
                                         InstructionPoplarBufferSet(shape));
      return Status::OK();
    }
    for (int64 i = 0; i != inst->operand_count(); ++i) {
      HloInstruction* operand = inst->mutable_operand(i);
      for (auto& indexed_shape : ShapeUtil::GetLeafShapes(operand->shape())) {
        const HloPoplarPosition input_position{operand, indexed_shape.index};

        ShapeIndex output_index = indexed_shape.index;
        if (is_tuple) {
          // If it's tuple, prepend index to output index.
          CHECK_EQ(ShapeUtil::TupleElementCount(shape), inst->operand_count());
          output_index.push_front(i);
        } else {
          // If it's not a tuple, use empty index and allow only one operand.
          CHECK_EQ(inst->operand_count(), 1);
        }
        const HloPoplarPosition output_position{inst, output_index};

        HloPoplarBufferSet buffer_set = analysis_->GetBufferSet(input_position);
        buffer_set.AddNewBufferUse(kind);

        VLOG(3) << "Forwarding buffer set " << buffer_set << " from "
                << input_position << " to " << output_position;

        analysis_->SetInstructionBufferSetOutput(inst, output_index,
                                                 buffer_set);
      }
    }
    return Status::OK();
  }

  // Simply defines instructions with non-tuple shapes as inplace on the 0th
  // operand.
  Status HandleSimpleInplace(HloInstruction* inst, BufferUseKind kind) {
    CHECK_GT(inst->operand_count(), 0);
    HloInstruction* operand_0 = inst->mutable_operand(0);

    auto is_tuple_shape = [](const HloInstruction* a) {
      return a->shape().IsTuple();
    };

    if (is_tuple_shape(inst) ||
        absl::c_any_of(inst->operands(), is_tuple_shape)) {
      return InternalErrorStrCat("Cannot handle tuple shaped instruction ",
                                 inst->ToString(),
                                 " as a simple inplace instruction");
    }

    // Simply forward the buffer set.
    HloPoplarBufferSet buffer_set = analysis_->GetBufferSet(operand_0);
    buffer_set.AddNewBufferUse(kind);  // either set, or update

    analysis_->SetInstructionBufferSetOutput(inst, ShapeIndex{}, buffer_set);

    VLOG(3) << "Forwarding buffer set " << buffer_set << " from "
            << operand_0->ToString() << " to " << inst->ToString();
    return Status::OK();
  }

  // Logical binary elementwise ops are inplace if their input/output type is
  // the same.
  Status HandleLogicalBinaryElementwise(HloInstruction* inst) {
    if (inst->shape().element_type() ==
        inst->operand(0)->shape().element_type()) {
      return HandleSimpleInplace(inst,
                                 /*kind=*/BufferUseKind::USE_ALIAS_READ_WRITE);
    } else {
      return HandleNotInplace(inst);
    }
  }

  // A not inplace instruction defines all the output buffers on device.
  Status HandleNotInplace(HloInstruction* inst) {
    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
      HloPoplarBuffer* buffer = analysis_->NewHloPoplarBuffer(
          inst, indexed_shape.index,
          /*locality=*/BufferLocality::kDeviceMemory);
      analysis_->SetInstructionBufferSetOutput(inst, indexed_shape.index,
                                               HloPoplarBufferSet({buffer}));
    }
    return Status::OK();
  }

  Status HandleCopy(HloInstruction* inst) override {
    TF_ASSIGN_OR_RETURN(auto clone_method_tree, GetCopyCloneMethod(inst));
    auto clone_method_it = clone_method_tree.leaf_begin();
    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
      auto clone_method = (*clone_method_it++).second;
      if (clone_method != CloneMethod_Bypass) {
        HloPoplarBuffer* buffer = analysis_->NewHloPoplarBuffer(
            inst, indexed_shape.index,
            /*locality=*/BufferLocality::kDeviceMemory);
        analysis_->SetInstructionBufferSetOutput(inst, indexed_shape.index,
                                                 HloPoplarBufferSet({buffer}));
      } else {
        const HloPoplarPosition input_position{inst->mutable_operand(0),
                                               indexed_shape.index};
        const HloPoplarPosition output_position{inst, indexed_shape.index};

        HloPoplarBufferSet buffer_set = analysis_->GetBufferSet(input_position);

        VLOG(3) << "Forwarding buffer set " << buffer_set << " from "
                << input_position << " to " << output_position;

        analysis_->SetInstructionBufferSetOutput(inst, indexed_shape.index,
                                                 buffer_set);
      }
    }
    return Status::OK();
  }

  HloPoplarDataflowAnalysis* analysis_;
  const CompilerAnnotations* annotations_;
  const bool is_entry_computation_;
  const ComputationInputBufferSets input_buffer_sets_;
};

HloPoplarDataflowAnalysis::HloPoplarDataflowAnalysis(const HloModule* module)
    : module_(module) {}

HloPoplarBuffer* HloPoplarDataflowAnalysis::NewHloPoplarBuffer(
    HloInstruction* instruction, const ShapeIndex& index,
    BufferLocality locality) {
  const HloPoplarBuffer::Id buffer_id = next_buffer_id_++;
  auto emplaced = buffers_.emplace(
      std::piecewise_construct, std::forward_as_tuple(buffer_id),
      std::forward_as_tuple(buffer_id, HloPoplarPosition{instruction, index},
                            locality));
  CHECK(emplaced.second);

  VLOG(3) << "New HloPoplarBuffer = " << emplaced.first->second.ToString();

  return &emplaced.first->second;
}

bool HloPoplarDataflowAnalysis::BufferIsDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const HloPoplarBufferSet& buffer_set = GetBufferSet(instruction, index);
  if (buffer_set.size() != 1) {
    return false;
  }
  return buffer_set.GetUniqueBuffer().instruction() == instruction;
}

const HloPoplarBuffer& HloPoplarDataflowAnalysis::GetBufferDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  CHECK(BufferIsDefinedAt(instruction, index)) << instruction->ToString();
  return GetUniqueBufferAt(instruction, index);
}

HloPoplarBuffer& HloPoplarDataflowAnalysis::GetBufferDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  CHECK(BufferIsDefinedAt(instruction, index));
  return GetUniqueBufferAt(instruction, index);
}

std::string HloPoplarDataflowAnalysis::ToString() const {
  std::string out =
      absl::StrCat("HloPoplarDataflowAnalysis, module ", module_->name(), "\n");
  absl::StrAppend(&out, "  Instruction buffer sets:\n");
  for (const HloComputation* computation :
       module_->MakeComputationPostOrder()) {
    if (!visited_computations_.contains(computation)) {
      continue;
    }
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      absl::StrAppend(&out, "Instruction: \n  ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
        for (auto& indexed_shape :
             ShapeUtil::GetLeafShapes(instruction->shape())) {
          absl::StrAppend(&out, "      tuple index ",
                          indexed_shape.index.ToString(), ":\n");
          const HloPoplarBufferSet& buffer_set =
              GetBufferSet(instruction, indexed_shape.index);
          for (const HloPoplarBuffer* buffer : buffer_set.buffers()) {
            absl::StrAppend(&out, "        ", buffer->ToString(),
                            BufferIsDefinedAt(instruction, indexed_shape.index)
                                ? " (def)"
                                : "",
                            "\n");
          }
        }
      } else {
        const HloPoplarBufferSet& buffer_set =
            GetBufferSet(instruction, /*index=*/{});
        for (const HloPoplarBuffer* buffer : buffer_set.buffers()) {
          absl::StrAppend(&out, "      ", buffer->ToString(),
                          BufferIsDefinedAt(instruction) ? " (def)" : "", "\n");
        }
      }
    }
  }
  absl::StrAppend(&out, "\nHloPoplarBuffers:\n");
  for (auto pair : buffers_) {
    absl::StrAppend(&out, pair.second.ToString(), "\n");
  }
  return out;
}

const HloPoplarBuffer& HloPoplarDataflowAnalysis::GetBuffer(
    HloPoplarBuffer::Id buffer_id) const {
  CHECK(ContainsKey(buffers_, buffer_id));
  return buffers_.at(buffer_id);
}

HloPoplarBuffer& HloPoplarDataflowAnalysis::GetBuffer(
    HloPoplarBuffer::Id buffer_id) {
  CHECK(ContainsKey(buffers_, buffer_id));
  return buffers_.at(buffer_id);
}

const HloPoplarBufferSet& HloPoplarDataflowAnalysis::GetBufferSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionBufferSet(instruction).GetOutputBufferSet(index);
}

HloPoplarBufferSet& HloPoplarDataflowAnalysis::GetBufferSet(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return GetInstructionBufferSet(instruction).GetMutableOutputBufferSet(index);
}

const HloPoplarBufferSet& HloPoplarDataflowAnalysis::GetBufferSet(
    const HloPoplarPosition& position) const {
  return GetBufferSet(position.instruction, position.index);
}

HloPoplarBufferSet& HloPoplarDataflowAnalysis::GetBufferSet(
    const HloPoplarPosition& position) {
  return GetBufferSet(position.instruction, position.index);
}

void HloPoplarDataflowAnalysis::SetInstructionBufferSetOutput(
    const HloInstruction* instruction, const ShapeIndex& index,
    const HloPoplarBufferSet& buffer_set) {
  auto itr = buffer_sets_.find(instruction);
  if (itr == buffer_sets_.end()) {
    itr = buffer_sets_
              .emplace(instruction,
                       InstructionPoplarBufferSet(instruction->shape()))
              .first;
    visited_computations_.insert(instruction->parent());
  }
  itr->second.SetOutputBufferSet(index, buffer_set);
}

void HloPoplarDataflowAnalysis::SetInstructionBufferSet(
    const HloInstruction* instruction,
    const InstructionPoplarBufferSet& instruction_buffer_set) {
  CHECK(!ContainsKey(buffer_sets_, instruction));
  visited_computations_.insert(instruction->parent());
  buffer_sets_.emplace(instruction, instruction_buffer_set);
}

const InstructionPoplarBufferSet&
HloPoplarDataflowAnalysis::GetInstructionBufferSet(
    const HloInstruction* instruction) const {
  CHECK(ContainsKey(buffer_sets_, instruction));
  return buffer_sets_.at(instruction);
}

InstructionPoplarBufferSet& HloPoplarDataflowAnalysis::GetInstructionBufferSet(
    const HloInstruction* instruction) {
  CHECK(ContainsKey(buffer_sets_, instruction));
  return buffer_sets_.at(instruction);
}

Status HloPoplarDataflowAnalysis::InitializeAndPropagate(
    const CompilerAnnotations* annotations, const HloComputation* entry) {
  DataflowAnalysisBufferVisitor visitor(this, annotations,
                                        /*input_buffer_sets=*/{},
                                        /*is_entry_computation=*/true);
  TF_RETURN_IF_ERROR(entry->Accept(&visitor));

  // Create empty buffer set for each non-visited instruction in the module
  for (HloComputation* comp : module_->computations()) {
    if (visited_computations_.contains(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->instructions()) {
      InstructionPoplarBufferSet instruction_set(inst->shape());
      SetInstructionBufferSet(inst, instruction_set);
    }
  }
  return Status::OK();
}

/* static */
StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>>
HloPoplarDataflowAnalysis::Run(const HloComputation* entry,
                               const CallGraph& call_graph,
                               const CompilerAnnotations* annotations) {
  const HloModule* module = entry->parent();
  VLOG(3) << "HloPoplarDataflowAnalysis::Run on module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());

  if (!call_graph.IsFlattened()) {
    return FailedPrecondition(
        "Cannot perform HloPoplarDataflowAnalysis as the module is not flat.");
  }

  auto dataflow_analysis =
      absl::WrapUnique(new HloPoplarDataflowAnalysis(module));

  TF_RETURN_IF_ERROR(
      dataflow_analysis->InitializeAndPropagate(annotations, entry));

  XLA_VLOG_LINES(3, dataflow_analysis->ToString());

  return std::move(dataflow_analysis);
}

/* static */
StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>>
HloPoplarDataflowAnalysis::Run(const HloComputation* entry,
                               const CompilerAnnotations* annotations) {
  auto call_graph = CallGraph::Build(entry->parent());
  return Run(entry, *call_graph, annotations);
}

/* static */
StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>>
HloPoplarDataflowAnalysis::Run(const HloModule* module,
                               const CompilerAnnotations& annotations) {
  auto call_graph = CallGraph::Build(module);
  return Run(module, annotations, *call_graph);
}

/* static */
StatusOr<std::unique_ptr<HloPoplarDataflowAnalysis>>
HloPoplarDataflowAnalysis::Run(const HloModule* module,
                               const CompilerAnnotations& annotations,
                               const CallGraph& call_graph) {
  return Run(module->entry_computation(), call_graph, &annotations);
}

}  // namespace poplarplugin
}  // namespace xla
