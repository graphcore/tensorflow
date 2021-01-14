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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/candidate_sampler.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_range_sampler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {

class CandidateSamplerOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    poplar::program::Sequence seq;

    // Retrieve attributes from the HloPoplarInstruction
    const HloCandidateSampler* sampler_inst = Cast<HloCandidateSampler>(inst);
    bool unique = sampler_inst->Unique();
    const uint64 range_max = sampler_inst->RangeMax();
    std::string distribution = sampler_inst->Distribution();
    std::string debug_name = GetDebugName(inst);

    // Try to place each op onto a different tile by using the next tile from
    // the linear mapping state
    const uint64 tile =
        MappingHelper::YieldNextTile(res.linear_mapping_state, graph);

    // Get inputs
    TF_ASSIGN_OR_RETURN(poplar::Tensor true_classes,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor seed,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    // Seed must be unsigned for the later call to poprand::setSeed
    poplar::Tensor seed_unsigned = seed.reinterpret(poplar::UNSIGNED_INT);
    const Shape sample_shape = output_shape.tuple_shapes()[0];
    // Note: Map each sample onto a single tile with grain size 1
    TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(sample_shape));
    poplar::Tensor samples =
        graph.addVariable(poplar_type, PoplarShapeFromXlaShape(sample_shape),
                          debug_name + "/samples");
    MappingHelper::MapTensorLinearly(res.linear_mapping_state, graph, samples,
                                     1, 1);
    const uint64 k = samples.numElements();

    // Create a sampler, sample from it and calculate expectations
    TF_ASSIGN_OR_RETURN(std::unique_ptr<RangeSampler> sampler,
                        RangeSamplerFactory(distribution, debug_name, range_max,
                                            tile, seed_unsigned, unique));
    sampler->Sample(graph, samples, seq);
    TF_ASSIGN_OR_RETURN(poplar::Tensor sampled_expectation,
                        sampler->Expectation(graph, samples, k, seq));
    TF_ASSIGN_OR_RETURN(poplar::Tensor true_expectation,
                        sampler->Expectation(graph, true_classes, k, seq));

    // Add the output tensors to the graph
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, samples));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, true_expectation));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, sampled_expectation));
    return seq;
  }
};

REGISTER_POPLAR_OP(CandidateSampler, CandidateSamplerOp)

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
