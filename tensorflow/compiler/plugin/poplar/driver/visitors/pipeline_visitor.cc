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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>

namespace xla {
namespace poplarplugin {

PipelineVisitor::PipelineVisitor(
    int64 stage_count, int64 iterations, CompilerResources& res,
    const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : iterations_(iterations),
      copy_sequences_(stage_count),
      program_sequences_(stage_count),
      SubComputationVisitor(res, inputs, dependent_subcomputations) {}

poplar::program::Sequence PipelineVisitor::GetPipelineSequence() const {
  poplar::program::Program ramp_up = GetPipelineRampUpSequence();
  poplar::program::Program ramp_down = GetPipelineRampDownSequence();
  poplar::program::Program repeat_block = GetPipelineRepeatBlockSequence();

  poplar::program::Sequence program;

  const auto half_seq_length = program_sequences_.size() / 2;

  program.add(ramp_up);
  program.add(
      poplar::program::Repeat(1 + iterations_ - half_seq_length, repeat_block));
  program.add(ramp_down);

  return program;
}

namespace {
/*
 * Creating a ramp sequence is just incrementally adding the poplar programs.
 *
 * @param a The even set of "pre-programs"
 * @param b The even set of "post-programs"
 * @param c The odd set of "pre-programs"
 * @param d The odd set of "post-programs"
 *
 * @returns a vector of poplar programs in the correct order for a ramp
 *          computation.
 *
 * Worked Example:
 *
 * Suppose our set of programs are:
 *  a = [A, C, E]
 *  b = [U, V, W]
 *  c = [B, D, F]
 *  d = [X, Y, Z]
 *
 * Suppose `a` and `c` are the set of compute stages, and `b` and `d` are the
 * associated set of post-compute inter-ipu-copies.
 *
 * In this example, our sequence length is 3
 * ramp_sequences := []
 *
 * We start adding poplar programs with i = 1, because 0 would do nothing
 * anyway.
 *
 * i := 1
 * ramp_sequences += [A] // Even pre-program
 * ramp_sequences += [U] // Even post-program
 * ramp_sequences += [B] // Odd pre-program
 * ramp_sequences += [X] // Odd post-program
 *
 * ramp_sequences = [A, U, B, X]
 *
 * i := 2
 * ramp_sequences += [A, C] // Even pre-programs
 * ramp_sequences += [U, V] // Even post-programs
 * ramp_sequences += [B, D] // Odd pre-programs
 * ramp_sequences += [X, Y] // Odd post-programs
 *
 * ramp_sequences = [A, U, B, X, A, C, U, V, B, D, X, Y]
 *
 * `ramp_sequences` is now a valid ramp up sequence.
 *
 * This matches the intuitive pipeline instruction sequence
 *    i ||  1  |  2  ||  n  || n+1 | n+2 ||
 * IPU0 ||AU|--|AU|--||AU|FZ||--|FZ|--|FZ||
 * IPU1 ||--|BX|--|BX||EW|BX||EW|--|EW|--||
 * IPU2 ||--|--|CV|DY||CV|DY||CV|DY|--|--||
 *      ||  RAMP UP  ||     || RAMP DOWN ||
 *
 * Interestingly, if all the inputs are reversed, `a` swapped with `d`, and `b`
 * swapped with `c`, then we get a valid (reversed) ramp down sequence.
 */
std::vector<poplar::program::Program> CreateRampSequences(
    const std::vector<poplar::program::Program>& a,
    const std::vector<poplar::program::Program>& b,
    const std::vector<poplar::program::Program>& c,
    const std::vector<poplar::program::Program>& d) {
  std::vector<poplar::program::Program> ramp_sequences;

  for (auto i = 1ul; i < a.size(); ++i) {
    for (auto k = 0ul; k < i; ++k) {
      ramp_sequences.push_back(a[k]);
    }

    for (auto k = 0ul; k < i; ++k) {
      ramp_sequences.push_back(b[k]);
    }

    for (auto k = 0ul; k < i; ++k) {
      ramp_sequences.push_back(c[k]);
    }

    for (auto k = 0ul; k < i; ++k) {
      ramp_sequences.push_back(d[k]);
    }
  }

  return ramp_sequences;
}

// Similar to the above, but with homogeneous "post-programs"
std::vector<poplar::program::Program> CreateRampSequences(
    const std::vector<poplar::program::Program>& a, poplar::program::Program b,
    const std::vector<poplar::program::Program>& c,
    poplar::program::Program d) {
  std::vector<poplar::program::Program> b_(a.size());
  b_.front() = b;

  std::vector<poplar::program::Program> d_(c.size());
  d_.front() = d;

  return CreateRampSequences(a, b_, c, d_);
}

// Similar to the above, but with homogeneous "pre-programs"
std::vector<poplar::program::Program> CreateRampSequences(
    poplar::program::Program a, const std::vector<poplar::program::Program>& b,
    poplar::program::Program c,
    const std::vector<poplar::program::Program>& d) {
  std::vector<poplar::program::Program> a_(b.size());
  a_.front() = a;

  std::vector<poplar::program::Program> c_(d.size());
  c_.front() = c;

  return CreateRampSequences(a_, b, c_, d);
}

// Return the pipeline stage index for the given hlo instruction
StatusOr<int> GetPipelineStage(const CompilerResources& res,
                               HloInstruction* hlo) {
  if (res.pipeline_stage_assignment.count(hlo) == 0) {
    return FailedPrecondition(
        "Hlo instruction \"%s\" does not have an assigned pipeline stage.",
        hlo->name());
  }

  return res.pipeline_stage_assignment.at(hlo);
}

}  // namespace

// Collect the pipeline stage programs and call CreateRampSequences
poplar::program::Program PipelineVisitor::GetPipelineRampUpSequence() const {
  std::vector<poplar::program::Program> program_sequences[2];
  poplar::program::Sequence copy_sequences[2];

  const auto half_seq_length = program_sequences_.size() / 2;

  for (auto i = 0ul; i < half_seq_length; ++i) {
    for (auto k = 0; k < 2; ++k) {
      program_sequences[k].push_back(program_sequences_[2 * i + k]);
      copy_sequences[k].add(copy_sequences_[2 * i + k]);
    }
  }

  auto ramp_up_sequences =
      CreateRampSequences(program_sequences[0], copy_sequences[0],
                          program_sequences[1], copy_sequences[1]);

  poplar::program::Sequence ramp_up;

  for (const auto& seq : ramp_up_sequences) {
    ramp_up.add(seq);
  }

  return ramp_up;
}

// Collect the pipeline stage programs and call CreateRampSequences
poplar::program::Program PipelineVisitor::GetPipelineRampDownSequence() const {
  std::vector<poplar::program::Program> program_sequences[2];
  poplar::program::Sequence copy_sequences[2];

  const auto half_seq_length = program_sequences_.size() / 2;

  for (auto i = 0ul; i < half_seq_length; ++i) {
    for (auto k = 0; k < 2; ++k) {
      program_sequences[k].push_back(program_sequences_[2 * i + k]);
      copy_sequences[k].add(copy_sequences_[2 * i + k]);
    }
  }

  // A ramp down is the mirror image of a ramp up
  absl::c_reverse(program_sequences[0]);
  absl::c_reverse(program_sequences[1]);

  auto ramp_down_sequences =
      CreateRampSequences(copy_sequences[1], program_sequences[1],
                          copy_sequences[0], program_sequences[0]);

  // Reverse back into the correct order
  absl::c_reverse(ramp_down_sequences);

  poplar::program::Sequence ramp_down;

  for (const auto& seq : ramp_down_sequences) {
    ramp_down.add(seq);
  }

  return ramp_down;
}

// Collect the pipeline stage programs and build the repeat block
poplar::program::Program PipelineVisitor::GetPipelineRepeatBlockSequence()
    const {
  std::vector<poplar::program::Program> repeat_block_sequences;
  std::vector<poplar::program::Program> program_sequences[2];
  std::vector<poplar::program::Program> copy_sequences[2];

  const auto half_seq_length = program_sequences_.size() / 2;

  for (auto i = 0ul; i < half_seq_length; ++i) {
    for (auto k = 0; k < 2; ++k) {
      program_sequences[k].push_back(program_sequences_[2 * i + k]);
      copy_sequences[k].push_back(copy_sequences_[2 * i + k]);
    }
  }

  repeat_block_sequences.insert(repeat_block_sequences.end(),
                                program_sequences[0].begin(),
                                program_sequences[0].end());
  repeat_block_sequences.insert(repeat_block_sequences.end(),
                                copy_sequences[0].begin(),
                                copy_sequences[0].end());
  repeat_block_sequences.insert(repeat_block_sequences.end(),
                                program_sequences[1].begin(),
                                program_sequences[1].end());
  repeat_block_sequences.insert(repeat_block_sequences.end(),
                                copy_sequences[1].begin(),
                                copy_sequences[1].end());

  poplar::program::Sequence repeat_block;

  for (const auto& seq : repeat_block_sequences) {
    repeat_block.add(seq);
  }

  return repeat_block;
}

Status PipelineVisitor::HandleNotImplemented(HloInstruction* hlo) {
  return xla::Unimplemented(
      "%s (%s) is not a valid pipeline stage hlo instruction",
      hlo->name().c_str(), HloOpcodeString(hlo->opcode()).c_str());
}

Status PipelineVisitor::HandleCall(HloInstruction* hlo) {
  HloComputation* comp = hlo->to_apply();
  VLOG(1) << "Processing " << hlo->name() << " : " << comp->name()
          << " as a pipeline stage";

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(resources_, hlo));
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateCallOp(resources_, hlo, hlo->shape(), tensor_map));

  program_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  const bool is_inter_ipu_copy_hlo =
      hlo->custom_call_target() ==
      GetPoplibsCustomOpTargetString(PoplibsOp::Poputil,
                                     PoplibsOp::IpuInterCopy);

  if (is_inter_ipu_copy_hlo) {
    return HandleInterIpuCopy(hlo);
  }

  const bool is_fifo_hlo =
      hlo->custom_call_target() ==
      GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::Fifo);

  if (is_fifo_hlo) {
    return HandleFifo(hlo);
  }

  return HandleNotImplemented(hlo);
}

Status PipelineVisitor::HandleFifo(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->ToString();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(resources_, hlo));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCustomCallOp(resources_, hlo, hlo->shape(), tensor_map));

  copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleInterIpuCopy(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();
  if (!IsPoplibsHloCustomOp(hlo)) {
    return HandleNotImplemented(hlo);
  }

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(resources_, hlo));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      CreateCustomCallOp(resources_, hlo, hlo->shape(), tensor_map));

  copy_sequences_[stage].add(prog);

  return Status::OK();
}

Status PipelineVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  VLOG(1) << "Processing " << hlo->name();

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(resources_, hlo));
  TF_ASSIGN_OR_RETURN(
      ArgVectors output_tensors,
      FindInplaceOutputTensors(tensor_map, resources_, hlo,
                               program_sequences_[stage], false));
  CHECK_EQ(output_tensors.size(), 1);
  CHECK_EQ(output_tensors[0].size(), CountShapes(hlo->shape()));
  for (int64 i = 0; i < output_tensors[0].size(); i++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, hlo, i, output_tensors[0][i]));
  }
  return Status::OK();
}

Status PipelineVisitor::FinishVisit(HloInstruction*) {
  auto seq = GetSequence();
  seq.add(program_sequences_[0]);
  program_sequences_[0] = seq;

  return Status::OK();
}

Status PipelineVisitor::HandleTuple(HloInstruction* hlo) {
  if (hlo->parent()->root_instruction() != hlo) {
    return FailedPrecondition(
        "Hlo tuple instructions are only allow in a pipeline when they are the "
        "root instruction. Hlo instruction \"%s\" is not.",
        hlo->name());
  }

  VLOG(1) << "Processing " << hlo->name();

  TF_ASSIGN_OR_RETURN(auto stage, GetPipelineStage(resources_, hlo));
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, resources_, hlo,
                                               program_sequences_[stage]));
  CHECK_EQ(inputs.size(), hlo->operand_count());
  uint64 n = 0;
  for (uint64 i = 0; i < inputs.size(); i++) {
    CHECK_EQ(inputs[i].size(), CountShapes(hlo->operand(i)->shape()));
    for (uint64 j = 0; j < inputs[i].size(); j++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, hlo, n, inputs[i][j]));
      n++;
    }
  }

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
