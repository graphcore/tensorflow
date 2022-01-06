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
#include "tensorflow/compiler/plugin/poplar/driver/tools/subcomputation_graph_caching.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/partitioned_elementwise_cluster_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
namespace subcomputation_graph_caching {
namespace {
RemoteBufferHandleVectors GetInputRemoteBufferHandles(
    DeferredArgRBVectors& inputs) {
  RemoteBufferHandleVectors handles(inputs.size());

  // Construct the structure for encoding remote buffer inputs and their
  // positions.
  for (int64 operand_idx = 0; operand_idx != inputs.size(); ++operand_idx) {
    auto& operand_inputs = inputs[operand_idx];
    RemoteBufferHandleVector& operand_handles = handles[operand_idx];
    operand_handles.resize(operand_inputs.size());
    for (int64 tuple_idx = 0; tuple_idx != operand_inputs.size(); ++tuple_idx) {
      // For remote buffer inputs store their handle.
      auto& operand_input = operand_inputs[tuple_idx];
      if (operand_input && operand_input->IsRemoteBuffer()) {
        operand_handles[tuple_idx] =
            operand_input->AsRemoteBufferHolder().GetHandle();
      }
    }
  }
  return handles;
}

ReallocateInputsInfo GetReallocateInputsInfo(const DeferredArgRBVectors& inputs,
                                             bool reallocate) {
  ReallocateInputsInfo reallocate_inputs;
  reallocate_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    reallocate_inputs.emplace_back(input.size(), reallocate);
    if (!reallocate) {
      // If there is a hint provided not to reallocate, any inputs which are not
      // parallel writeable are still reallocated.
      for (int64 i = 0; i != input.size(); ++i) {
        if (input[i] && input[i]->IsTensor()) {
          const poplar::Tensor& t = input[i]->AsTensor();
          reallocate_inputs.back()[i] = !t.isParallelWriteable();
        }
      }
    }
  }
  return reallocate_inputs;
}
}  // namespace

size_t SubcomputationGraphCacheKeyHash::operator()(
    const SubcomputationGraphCacheKey& key) const {
  size_t hash = HloComputationHash()(key.computation);
  for (auto& operand_handles : key.remote_buffer_handles) {
    for (auto& operand_handle : operand_handles) {
      hash = tensorflow::Hash64Combine(
          hash, std::hash<std::string>()(operand_handle.value_or("no_handle")));
    }
  }
  hash = tensorflow::Hash64Combine(hash, key.keep_input_layouts);
  return tensorflow::Hash64Combine(hash, key.partitioned_elementwise_cluster);
}

bool SubcomputationGraphCacheKeyEquals::operator()(
    const SubcomputationGraphCacheKey& a,
    const SubcomputationGraphCacheKey& b) const {
  if (a.keep_input_layouts != b.keep_input_layouts) {
    return false;
  }

  if (a.partitioned_elementwise_cluster != b.partitioned_elementwise_cluster) {
    return false;
  }

  if (a.remote_buffer_handles != b.remote_buffer_handles) {
    return false;
  }

  return HloComputationEquals()(a.computation, b.computation);
}

StatusOr<std::shared_ptr<DeferredVisitor>>
SubcomputationGraphCache::GetOrCompileSubcomputation(
    CompilerResources& res, TensorOrRemoteBufferVectors& inputs,
    const HloComputation* computation, bool keep_input_layouts,
    bool partitioned_elementwise_cluster) {
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  return GetOrCompileSubcomputation(res, deferred_inputs, computation,
                                    keep_input_layouts,
                                    partitioned_elementwise_cluster);
}

StatusOr<std::shared_ptr<DeferredVisitor>>
SubcomputationGraphCache::GetOrCompileSubcomputation(
    CompilerResources& res, DeferredArgRBVectors& inputs,
    const HloComputation* computation, bool keep_input_layouts,
    bool partitioned_elementwise_cluster) {
  SubcomputationGraphCacheKey key{computation, keep_input_layouts,
                                  partitioned_elementwise_cluster,
                                  GetInputRemoteBufferHandles(inputs)};

  auto itr = table_.find(key);
  if (itr == table_.end()) {
    VLOG(2) << "Compiling sub-computation " << computation->name();
    XLA_VLOG_LINES(2, computation->ToString());

    auto order =
        computation->parent()->schedule().sequence(computation).instructions();
    std::shared_ptr<DeferredVisitor> deferred_visitor;
    if (partitioned_elementwise_cluster) {
      CHECK(res.current_cluster_visitor == nullptr)
          << "Nested partitioned clusters are not allowed.";
      auto cluster_visitor =
          std::make_shared<PartitionedElementwiseClusterVisitor>(
              next_rearrangement_id_, res, inputs, computation->name(),
              /*allocate_all_input_tensors=*/true,
              /*dependent_computations=*/std::vector<const DeferredVisitor*>{},
              GetReallocateInputsInfo(inputs, false));
      res.current_cluster_visitor = cluster_visitor.get();
      TF_RETURN_IF_ERROR(
          computation->AcceptOrdered(cluster_visitor.get(), order));
      res.current_cluster_visitor = nullptr;
      next_rearrangement_id_ = cluster_visitor->GetNextRearrangementId();
      deferred_visitor = std::move(cluster_visitor);
    } else {
      deferred_visitor = std::make_shared<DeferredVisitor>(
          res, inputs, computation->name(),
          /*allocate_all_input_tensors=*/true,
          /*dependent_computations=*/std::vector<const DeferredVisitor*>{},
          GetReallocateInputsInfo(inputs, !keep_input_layouts));
      TF_RETURN_IF_ERROR(
          computation->AcceptOrdered(deferred_visitor.get(), order));
    }

    if (computation->HasSideEffect()) {
      return deferred_visitor;
    }
    itr = table_.emplace(key, deferred_visitor).first;
  } else {
    VLOG(1) << "Computation " << computation->name()
            << " has already been compiled, reusing the code.";
  }
  return itr->second;
}
}  // namespace subcomputation_graph_caching
}  // namespace poplarplugin
}  // namespace xla
