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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_DEFERRED_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_DEFERRED_VISITOR_H_

#include <functional>
#include <map>
#include <poplar/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_full.h"

namespace xla {
namespace poplarplugin {
struct CompilerResources;

using ReallocateInputsInfo = std::vector<std::vector<bool>>;
using TensorInputDescription = std::vector<std::vector<bool>>;

// Set of locations in which a tensor allocation is deferred for.
using DeferredAllocationsLocationsSet = absl::flat_hash_set<TensorLocation>;

// Function which is called when a deferred allocation is being allocated.
// Inputs is the allocation location.
// Returns the allocated tensor.
using DeferredAllocateFunction =
    std::function<StatusOr<poplar::Tensor>(TensorLocation)>;

// Function which is called on a allocated tensor to add any stream copies etc.
// Input is an allocated tensor.
// Returns a post processed Tensor.
using DeferredPostProcessFunction =
    std::function<StatusOr<poplar::Tensor>(poplar::Tensor)>;

// Class used by the DeferredVisitor to keep track of deferred
// allocations in the current visitor.
class DeferredAllocations {
 public:
  explicit DeferredAllocations(TensorMap& tensor_map)
      : tensor_map_(tensor_map) {}

  // Add a deferred allocation location.
  Status AddDeferredAllocation(bool allocate_now, TensorLocation location,
                               DeferredAllocateFunction allocate_fn,
                               DeferredPostProcessFunction post_process_fn);

  // Add a deferred allocation mapping from a given input location to an output
  // location.
  Status AddDeferredAllocationUser(TensorLocation user_input_location,
                                   TensorLocation user_output_location);

  // Make a tensor allocation and post process it.
  Status MakeDeferredAllocation(TensorLocation allocation_location,
                                TensorLocation input_to_allocation_location);

  // Post process a tensor from an allocation location.
  Status PostProcessAllocation(TensorLocation allocation_location,
                               poplar::Tensor tensor);

  // Returns whether the given location is a deferred allocation location, given
  // the current state.
  static bool IsDeferredAllocationLocation(CompilerResources& res,
                                           TensorLocation location);

  // Returns whether the given location is a deferred allocation location.
  bool IsDeferredAllocationLocation(TensorLocation location);

  // This is called by the tensor map when a tensor value for a location has
  // been requested. If any locations in the range are deferred, then allocate
  // them.
  static void AllocateIfExists(
      CompilerResources& res, const HloInstruction* inst,
      absl::optional<int64> opt_tensors_start = absl::nullopt,
      absl::optional<int64> opt_tensors_end = absl::nullopt);

  // Get all the input allocations which have not been allocated yet.
  const std::vector<TensorLocation> GetNotAllocatedLocations() const;

 private:
  // This is called by the tensor map when a tensor value for a location has
  // been requested. If any locations in the range are deferred, then allocate
  // them.
  Status AllocateIfExists(const HloInstruction* inst, int64 tensors_start,
                          int64 tensors_end);

  // Invokes the allocation function and propagates the allocation to any
  // locations which were deferred.
  Status MakeAllocation(TensorLocation input_location,
                        TensorLocation allocation_location);

  TensorMap& tensor_map_;

  // For each input location, store all the locations which are also deferring
  // the tensor.
  absl::flat_hash_map<TensorLocation, DeferredAllocationsLocationsSet>
      to_allocate_locations_;

  // For each of the input locations, store the function which is called for
  // allocations.
  absl::flat_hash_map<TensorLocation, DeferredAllocateFunction>
      allocation_functions_;

  // For each of the input locations, store the function which is called for
  // post processing.
  absl::flat_hash_map<TensorLocation, DeferredPostProcessFunction>
      post_process_functions_;

  // Set of locations that have been allocated.
  absl::flat_hash_set<TensorLocation> allocated_locations_;

  // Lookup map for what deferred allocation location set a location belongs to.
  std::map<TensorLocation, TensorLocation> location_lookup_table_;
};

/**
 * An Hlo instruction visitor which deferres allocation of tensors until they
 * can be allocated with an allocation target or it is used by an instruction
 * which requires the tensor to perform an operation.
 *
 * In order to postpone the allocation, handlers for the inputs (Parameters and
 * Infeeds) check whether they have an allocation target for that input tensor
 * and if they do not, they use DeferredAllocations structure to indicate that a
 * particular input is deferred.
 * When deferring and input the following information is provided:
 * - the location of the input to the graph (instruction and the flat tuple
 * index)
 * - the allocation function which is called to allocate the tensor in the
 * Poplar graph.
 * - the post processing function which can be used to add any stream copies to
 * a sequence.
 * These functions will be called when an allocation is being made.
 *
 * This visitor implements handlers for some instructions which do not need the
 * tensor be alloacted as they do not perform any calculations.
 *
 * Specialization of the handle for GetTupleElement instruction which checks if
 * any of its inputs are deferred, and if they are:
 * - and if there is an allocation target for this location (usually generated
 * by the ForwardAllocation pass) then allocate the tensor and propagate the
 * allocation information.
 * - otherwise if there is no allocation target, mark this location as deferred.
 * If the input to the GTE already has a tensor then just forward the tensor.
 *
 * Specialization of the handle for Tuple instructions, similarly checks all the
 * input locations, and if any of them are deferred inputs then the output is
 * marked as deferred too, otherwise forward the tensor.
 *
 * For instructions with callsites - While loops, Repeat loops, Pipelines and
 * Pipeline Stages, similarly get all the deferred inputs and recursively
 * create another DeferredVisitor. Any deferred inputs will also be deferred
 * inside of this evaluator until an allocation is made.
 *
 * Note that when any of the instructions has the same tensor as an input at two
 * different locations then the allocation of that tensor is forced.
 *
 * If a deferred tensor is used by any other instructions which actually
 * requires the tensor (for example to do calculations), then the TensorMap
 * functions for getting input/output tensors will automatically create allocate
 * the tensor and call the post processing function (note that in this scenario
 * the tensor is most likely to be linearly allocate which might not be
 * optimal).
 *
 * After the visitor has finished, any deferred allocations are propagated up
 * the scope to the callsite. Depending on whether the particular operand was
 * inplace at that callsite, an additional clone of the tensor might be added.
 *
 * Example:
 *
 * entry {
 *  p0 = param (0) <= allocation target convolution lhs
 *  p1 = param (1) <= allocation target convolution rhs
 *  p2 = param (2)
 *  inputs = tuple(p0, p1, p2)
 *  ROOT loop = repeat(inputs), to_apply=body
 * }
 *
 * body {
 *  args = param (0)
 *  a0 = gte(args), index=0
 *  a1 = gte(args), index=1
 *  acts = convolution(a0, a1)
 *  a2 = gte(args), index=2 <= allocation target bias add operand 1
 *  acts' = bias-add(acts, a2)
 *  ROOT t = tuple(a0, a1, acts')
 * }
 *
 * The evaluator will then first start at the entry computation and allocate p0
 * and p1 as they already have allocation targets.
 * Allocation of p2 is deferred throught the instructions p2, inputs (the tuple
 * instruction) and at the callsite to the repeat loop.
 * During the evaluation of the repeat loop, a2 eventually gets an allocation
 * target which is based on another tensor (bias add).
 * Once the evaluation the repeat loop is finished the tensor allocated for a2
 * is propagate up the scope and it is also used by p2 (as the repeat loop is
 * inplace) and the post processing function is called to add any operations for
 * that particular input.
 */
class DeferredVisitor : public FullVisitor {
 public:
  /**
   * Constructor for the DeferredVisitor.
   *
   * @param res Compiler resources
   * @param callsite_inputs The deferred inputs at the callsite - if any input
   * tensor is not yet allocated this visitor will allocate it, unless specified
   * by `allocate_all_input_tensors`, in which case if the input tensor is not
   * used inside the computation it will not be allocated.
   * @param allocate_all_input_tensors If there are any inputs which were not
   * used in the computation, this flag decides whether to allocate them anyway
   * or not.
   * @param dependent_computations When checking liveness of buffers, those
   * buffers might be also live in other dependent subcomputations.
   * @param reallocate_inputs When allocating the tensor for a parameter in the
   * computation, this flag indicates whether the parameter should be
   * reallocated given its uses in the computation or whether it should preserve
   * the layout of the input tensor.
   */
  DeferredVisitor(
      CompilerResources& res, const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id,
      bool allocate_all_input_tensors = true,
      const std::vector<const DeferredVisitor*>& dependent_computations = {},
      bool reallocate_inputs = true);

  DeferredVisitor(
      CompilerResources& res, const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id,
      bool allocate_all_input_tensors,
      const std::vector<const DeferredVisitor*>& dependent_computations,
      const ReallocateInputsInfo& reallocate_inputs_info);

  DeferredVisitor() = delete;

  // Returns the input tensors for this computation.
  const TensorOrRemoteBufferVectors& inputs() const;

  // Returns the output tensors of this computation.
  const TensorOrRemoteBufferVector& outputs() const;

  // Returns whether an input to the computation was allocated for this
  // computation.
  bool InputIsAllocated(int64 param, unsigned int index) const;

  // Returns whether an input to this computation is used.
  bool InputIsUsed(int64 param, unsigned int index) const;

  // Explicitly override all the handlers for deferred allocations as final so
  // that any inheriting visitor is aware of deferred allocations.
  Status HandleGetTupleElement(HloInstruction* inst) final;
  Status HandleInfeed(HloInstruction* inst) final;
  Status HandleParameter(HloInstruction* inst) final;
  Status HandleTuple(HloInstruction* inst) final;
  Status HandleCall(HloInstruction* inst) final;
  Status HandleWhile(HloInstruction* inst) final;
  Status HandleCustomCall(HloInstruction* inst) final;
  Status HandleConditional(HloInstruction* inst) final;
  Status HandleFusion(HloInstruction* inst) final;

  // Finish visit always sets the output tensors and moves the tensor map and
  // then calls FinishDeferedAllocationVisit.
  Status FinishScopedVisit(HloInstruction* inst) final;

  // A function which propagates any tensors which were not allocated at call
  // site but now have a tensor.
  virtual Status PropagateDeferredAllocations(
      const HloInstruction* callsite_inst,
      const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id);

  virtual Status PropagateDeferredAllocationsOperand(
      const HloInstruction* callsite_inst, int64 operand_idx,
      int64 parameter_idx,
      const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
      const poplar::DebugNameAndId& debug_name_and_id);

  poplar::program::Sequence GetSequence(
      bool copy_execution_counters = true) final;

  poplar::program::Sequence GetFunctionCall();

 protected:
  // Signal that we are entering a new variable scope, where zeroing and write
  // undef need to be tracked.
  void EnterVariableScope();

  // Signal that we are exiting a variable scope, where zeroing and write
  // undef need to be tracked. Zeroing and write undef are left to the user of
  // this function.
  Status ExitVariableScope();

  Status AddSequenceForInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  // Get the inputs for a deferred instruction.
  StatusOr<DeferredArgRBVectors> GetInputsForDeferredRBInstruction(
      const HloInstruction* inst);

  // Handlers which are aware of deferred allocations - can be overriden by
  // other handlers which are also deferred allocation aware.
  virtual Status HandleDeferredAllocationCall(HloInstruction* inst);
  virtual Status HandleDeferredAllocationTuple(HloInstruction* inst);
  virtual Status HandleDeferredAllocationWhile(HloInstruction* inst);

  // Handler of WideConst fusion which is aware of deferred
  // allocation.
  virtual Status HandleDeferredWideConst(HloInstruction* inst);

  // Handler of GradientAccumulatorCreate which is aware of deferred
  // allocation.
  virtual Status HandleGradientAccumulatorCreate(HloInstruction* inst);

  // Handler of CreateBuffer which is aware of deferred allocation.
  virtual Status HandleCreateBuffer(HloInstruction* inst);

  // Handler of RemoteParameterLoad which is aware of deferred allocation.
  virtual Status HandleRemoteParameterLoad(HloInstruction* inst);

  // Handler of BufferLoadSlice which is aware of deferred allocation.
  virtual Status HandleBufferLoadSlice(HloInstruction* inst);

  // Handler for all custom calls apart from those which support deferred
  // allocation.
  virtual Status HandleNonDeferredCustomCall(HloInstruction* inst);

  // Handler for all custom calls apart from those which support deferred
  // allocation.
  virtual Status HandleNonDeferredFusion(HloInstruction* inst);

  // FinishScopedVisit which is aware of deferred allocations.
  virtual Status FinishDeferedAllocationVisit(HloInstruction* inst);

  // Function called for each tensor in a parameter HloInstruction.
  // Input location is the location at which the tensor is an input to the
  // computation
  virtual Status HandleParameterTensor(TensorLocation input_location,
                                       const Shape shape);

  virtual Status PreProcessParameter(HloInstruction* parameter);

  // Default deferred allocation function.
  virtual DeferredAllocateFunction MakeParameterAllocationFunction(
      TensorLocation allocation_location, const Shape& shape,
      absl::optional<TensorOrRemoteBuffer> tensor_like,
      const poplar::DebugNameAndId& debug_name_and_id);
  // Default deferred post-processing function.
  virtual DeferredPostProcessFunction MakeParameterPostProcessFunction(
      TensorLocation input_location, int64 param_num, const Shape& shape,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Allocates the input by trying to find an allocation target, otherwise tries
  // to use the `tensor_like` argument to create an input tensor.
  // Allocation location is the location where the tensor is actually allocated.
  StatusOr<poplar::Tensor> AllocateInput(
      TensorLocation allocation_location, const Shape& shape,
      absl::optional<poplar::Tensor> tensor_like,
      const poplar::DebugNameAndId& debug_name_and_id);

  StatusOr<poplar::Tensor> AllocateInput(
      TensorLocation allocation_location, const Shape& shape,
      absl::optional<TensorOrRemoteBuffer> tensor_like,
      const poplar::DebugNameAndId& debug_name_and_id);

  StatusOr<poplar::Tensor> AllocateInput(
      TensorLocation allocation_location, const Shape& shape,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Function called for each input tensor into the computation.
  // Input location is the location at which the tensor is an input to the
  // computation (parameter/infeed).
  StatusOr<poplar::Tensor> PostProcessInputTensor(
      poplar::Tensor tensor, TensorLocation input_location, const Shape& shape,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Called by AllocateInput when allocating an input for an infeed.
  StatusOr<poplar::Tensor> PostProcessInfeedAllocation(
      TensorLocation location, const Shape& shape,
      poplar::program::Sequence& sequence, poplar::Tensor tensor,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Called by AllocateInput when allocating an input for a parameter.
  // By default, inplace evaluator does no post processing for parameters.
  virtual StatusOr<poplar::Tensor> PostProcessParameterAllocation(
      TensorLocation location, const Shape& shape,
      poplar::program::Sequence& sequence, poplar::Tensor tensor,
      const poplar::DebugNameAndId& debug_name_and_id) {
    return tensor;
  }

  // Get the top of the stack of deferred allocations scopes.
  StatusOr<DeferredAllocations*> GetDeferredAllocations();

  // Implementation of the PropagateDeferredAllocations which will add tensor
  // copies for any operand which requires it.
  Status PropagateDeferredAllocations(
      const HloInstruction* callsite_inst,
      const DeferredArgRBVectors& callsite_inputs, std::vector<bool> add_clone,
      const poplar::DebugNameAndId& debug_name_and_id);

  Status PropagateDeferredAllocationsOperand(
      const HloInstruction* callsite_inst, int64 operand_idx,
      int64 parameter_idx,
      const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
      bool add_clone, const poplar::DebugNameAndId& debug_name_and_id);

  // Returns true if the input is used in this computation and therefore it
  // needs to be allocated.
  bool InputIsUsedInThisComputation(const HloInstruction* inst,
                                    int64 tuple_index);

  // Returns true if the input is used in any dependent computation and
  // therefore it needs to be allocated.
  bool InputIsUsedInDependentComputations(TensorLocation location);

  // These are the inputs at the callsite and where the values will come from,
  // but the actual input in the computation will need it's own Tensor
  // (potentially with a different layout).
  DeferredArgRBVectors callsite_inputs_;
  // Actual inputs to the computation.
  TensorOrRemoteBufferVectors computation_inputs_;
  // Outputs of the computation.
  TensorOrRemoteBufferVector outputs_;

  // When checking livness of buffers, those buffers might be also live in other
  // dependent subcomputations.
  const std::vector<const DeferredVisitor*> dependent_computations_;

  // Allocated tensors for inputs which are used by this subcomputation only.
  TensorInputDescription used_tensors_;
  // Allocated tensors for inputs which are used by this or dependent
  // subcomputations.
  TensorInputDescription allocated_tensors_;

  // Whether to reallocate or keep the layout of the input tensors.
  const ReallocateInputsInfo reallocate_inputs_info_;

 private:
  absl::optional<poplar::Function> function_;

  const bool allocate_all_input_tensors_;
};

// Similar to DeferredVisitor, but the inputs are used inplace.
class InplaceDeferredVisitor : public DeferredVisitor {
 public:
  InplaceDeferredVisitor(
      CompilerResources& res, const DeferredArgRBVectors& inputs,
      const HloPoplarInplaceDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id,
      const std::vector<const DeferredVisitor*>& dependent_subcomputations = {},
      bool reallocate_inputs = false);

  InplaceDeferredVisitor(
      CompilerResources& res, const DeferredArgRBVectors& inputs,
      const HloPoplarInplaceDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id,
      const std::vector<const DeferredVisitor*>& dependent_subcomputations,
      const ReallocateInputsInfo& reallocate_inputs_info);

  // Function called for each tensor in a parameter HloInstruction.
  // Input location is the location at which the tensor is an input to the
  // computation.
  Status HandleParameterTensor(TensorLocation input_location,
                               const Shape shape) override;

  // If the subcomputation is used as a loop, then add input/output aliasing
  // copies. Returns the loop state (i.e. the output of the loop).
  StatusOr<TensorOrRemoteBufferVector> AddLoopInputOutputAliasingCopies(
      poplar::Graph& graph, const HloComputation* computation,
      const poplar::DebugNameAndId& debug_name_and_id);

  // A function which propagates any tensors which were not allocated at call
  // site but now have a layout.
  Status PropagateDeferredAllocations(
      const HloInstruction* callsite_inst,
      const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id) override;

  Status PropagateDeferredAllocationsOperand(
      const HloInstruction* callsite_inst, int64 operand_idx,
      int64 parameter_idx,
      const std::vector<absl::optional<TensorOrRemoteBuffer>>& callsite_input,
      const poplar::DebugNameAndId& debug_name_and_id) override;

  // If the visitor operator is allowed to reallocate inputs, then copies from
  // the callsite to computation inputs might be required as they are different
  // tensors.
  StatusOr<poplar::program::Sequence> GetPreambleCopies(
      const poplar::DebugNameAndId& debug_name_and_id);

 protected:
  // Add the given sequence to the correct sequence for aliasing copies.
  virtual void AddSequenceForAliasingCopy(const HloInstruction* inst,
                                          const poplar::program::Sequence& seq);

  // Given an output flat index get the corresponding parameter number and flat
  // index.
  std::pair<int64, int64> GetParameterNumberAndFlatIndex(
      int64 output_flat_index);

  const HloPoplarInplaceDescription& GetCallsiteDescription() const {
    return description_;
  }

 private:
  const HloPoplarInplaceDescription description_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_DEFERRED_VISITOR_H_
