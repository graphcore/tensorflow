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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/repeat_loop_overlap_io_visitor.h"

#include <string>
#include <vector>

#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

RepeatLoopOverlapIOVisitor::RepeatLoopOverlapIOVisitor(
    CompilerResources& res, const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const ReallocateInputsInfo& reallocate_inputs_info,
    const poplar::DebugNameAndId& debug_name_and_id)
    : RepeatLoopVisitor(res, inputs, description, reallocate_inputs_info,
                        debug_name_and_id) {}

StatusOr<DriverProgramSequence*>
RepeatLoopOverlapIOVisitor::GetSequenceForInstruction(
    const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kOutfeed: {
      return &outfeed_sequence_;
    }
    case HloOpcode::kInfeed: {
      return &infeed_sequence_;
    }
    default: { return FailedPrecondition("Unsupported instruction type"); }
  }
}

Status RepeatLoopOverlapIOVisitor::AppendSequenceGroupedByInstruction(
    const HloInstruction* inst, const DriverProgramSequence& seq) {
  TF_ASSIGN_OR_RETURN(auto to_update, GetSequenceForInstruction(inst));
  to_update->add(seq);
  return Status::OK();
}

Status RepeatLoopOverlapIOVisitor::PrependSequenceGroupedByInstruction(
    const HloInstruction* inst, const DriverProgramSequence& seq) {
  TF_ASSIGN_OR_RETURN(auto to_update, GetSequenceForInstruction(inst));
  *to_update =
      DriverProgramSequence({seq, *to_update}, GetDebugNameAndId(inst));
  return Status::OK();
}

Status RepeatLoopOverlapIOVisitor::AddSequenceForInstruction(
    const HloInstruction* inst, const DriverProgramSequence& seq) {
  switch (inst->opcode()) {
    case HloOpcode::kInfeed:
      infeed_sequence_.add(seq);
      return Status::OK();
    case HloOpcode::kOutfeed:
      outfeed_sequence_.add(seq);
      return Status::OK();
    case HloOpcode::kGetTupleElement: {
      TF_ASSIGN_OR_RETURN(const auto tileset, GetTileset(inst));
      if (tileset == TILESET_IO_TILES) {
        auto ancestor = inst->LatestNonGteAncestor();

        if (ancestor->opcode() == HloOpcode::kInfeed) {
          io_tile_copy_in_sequence_.add(seq);
        } else {
          io_tile_copy_out_sequence_.add(seq);
        }

        return Status::OK();
      }
      break;
    }
    case HloOpcode::kCustomCall:
      if (IsPoplarInstruction(PoplarOp::InterTilesetCopy, inst)) {
        auto copy_inst = Cast<HloInterTilesetCopy>(inst);

        if (!copy_inst->IsCopyToIoTiles()) {
          io_tile_copy_in_sequence_.add(seq);
        } else {
          io_tile_copy_out_sequence_.add(seq);
        }

        return Status::OK();
      }
      break;
    case HloOpcode::kTuple: {
      TF_ASSIGN_OR_RETURN(const auto tileset, GetTileset(inst));
      if (tileset == TILESET_IO_TILES) {
        outfeed_sequence_.add(seq);
        return Status::OK();
      }
    } break;
    default:
      break;
  }

  return RepeatLoopVisitor::AddSequenceForInstruction(inst, seq);
}

DriverProgramSequence RepeatLoopOverlapIOVisitor::GetRepeatLoopSequence(
    const HloInstruction* inst) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(inst);
  const int64_t repeat_count = GetRepeatLoopCount(inst);

  auto& graph = GetGraph(resources_, inst);
  DriverProgramSequence seq(debug_name_and_id);
  seq.add(pre_loop_sequence_);

  DriverProgramSequence call_seq({debug_name_and_id, "call"});
  {
    DriverProgramSequence compute_seq({debug_name_and_id, "compute"});
    compute_seq.add(GetSequence(/*copy_execution_counters*/ false));
    // Increase the local execution counters at the end of each iteration.
    compute_seq.add(execution_counters_.IncrementLiveCounters());
    call_seq.add(poplar::program::Call(graph.addFunction(compute_seq)));
  }

  // The inner body of the loop. To allow the computa and IO to overlap we
  // construct the control program in the following way:
  // 1. Copy between the IO and compute tiles.
  // 2. Start the IO.
  // 3. Start the computation.
  // The breaking of the data dependency in step 1 allows poplar to overlap the
  // IO, assuming they are using non-overlapping tiles. Because we have 2
  // additional compute-batches in the device in any given iteration, we must
  // unroll the loop to fill and flush this.
  DriverProgramSequence repeat_seq({debug_name_and_id, "repeat"});
  // Copy from the IO tiles the nth compute-batch of data.
  repeat_seq.add(io_tile_copy_in_sequence_);
  // Copy to the IO tiles the n-1th compute-batch of data.
  repeat_seq.add(io_tile_copy_out_sequence_);
  // Start loading the n+1th compute-batch of data.
  repeat_seq.add(infeed_sequence_);
  // Start sending the n-1th compute-batch of data to the host.
  repeat_seq.add(outfeed_sequence_);
  // Compute on the nth compute-batch of data.
  repeat_seq.add(call_seq);

  if (has_resource_update_) {
    CHECK_GT(num_mini_batches_to_accumulate_, 0);
    CHECK_EQ(repeat_count % num_mini_batches_to_accumulate_, 0);
    // Create a double loop - the inner loop executes for
    // `num_mini_batches_to_accumulate_` iterations and then performs the
    // resource update.
    DriverProgramSequence inner_seq({debug_name_and_id, "inner"});
    // Zero the gradient accumulation buffers.
    inner_seq.add(tensors_zeroing_sequence_);
    // Load in initial data.
    inner_seq.add(infeed_sequence_);
    // Copy from the IO tiles.
    inner_seq.add(io_tile_copy_in_sequence_);
    // Start loading the next iteration's data.
    inner_seq.add(infeed_sequence_);
    // Compute on the first data elements.
    inner_seq.add(call_seq);

    // Run the inner loop defined above. We now have data ready to compute on
    // and ready to be sent to the host.
    inner_seq.add(poplar::program::Repeat(num_mini_batches_to_accumulate_ - 2,
                                          repeat_seq, {debug_name_and_id}));
    // Copy to/from the IO tiles.
    inner_seq.add(io_tile_copy_in_sequence_);
    inner_seq.add(io_tile_copy_out_sequence_);
    // Start sending results to the host.
    inner_seq.add(outfeed_sequence_);
    // Compute on the last data elements.
    inner_seq.add(call_seq);
    // Copy the final results to the IO tiles.
    inner_seq.add(io_tile_copy_out_sequence_);
    // Send the final results to the host.
    inner_seq.add(outfeed_sequence_);
    // Perform and resource updates.
    inner_seq.add(resource_update_sequence_);

    // Repeat the inner loop.
    seq.add(
        poplar::program::Repeat(repeat_count / num_mini_batches_to_accumulate_,
                                inner_seq, {debug_name_and_id}));
  } else {
    // Load in initial data.
    seq.add(infeed_sequence_);
    // Copy from the IO tiles.
    seq.add(io_tile_copy_in_sequence_);
    // Start loading the next iteration's data.
    seq.add(infeed_sequence_);
    // Compute on the first data elements.
    seq.add(call_seq);

    // Run the inner loop defined above. We now have data ready to compute on
    // and ready to be sent to the host.
    seq.add(poplar::program::Repeat(repeat_count - 2, repeat_seq,
                                    {debug_name_and_id}));
    // Copy to/from the IO tiles.
    seq.add(io_tile_copy_in_sequence_);
    seq.add(io_tile_copy_out_sequence_);
    // Start sending results to the host.
    seq.add(outfeed_sequence_);
    // Compute on the last data elements.
    seq.add(call_seq);
    // Copy the final results to the IO tiles.
    seq.add(io_tile_copy_out_sequence_);
    // Send the final results to the host.
    seq.add(outfeed_sequence_);
  }
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
