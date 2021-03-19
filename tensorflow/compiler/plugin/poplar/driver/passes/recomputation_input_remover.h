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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTATION_INPUT_REMOVER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTATION_INPUT_REMOVER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * Pass which removes RecomputationInput instructions and inserts control
 * dependencies such that users of the checkpointed input are executed as soon
 * as possible.
 *
 * The RecomputationInput instructions has two inputs:
 * - the checkpointed recomputation input ('cri') - this is the checkpointed
 * value which has been inserted by the recomputation pass.
 * - the old input ('oi') - this is the original value in the
 * forward/recomputation graph which was checkpointed.
 *
 * For example take a following recomputation/backward graph:
 * // Recomp
 * a = op_x(in0, in1)
 * b = op_z(a)
 * c = op_y(b)
 * d = op_x(c, in2)
 * e = op_z(d)
 * f = op_y(e)
 *
 * // Backward
 * f' = op_y_grad(e, f)
 * e' = op_z_grad(d, f')
 * d_lhs' = op_x_grad_lhs(in2, e')
 * d_rhs' = op_x_grad_rhs(c, e')
 * c' = op_y_grad(b, d_lhs')
 * b' = op_z_grad(a, c')
 * a_rhs' = op_x_grad_rhs(in0, b')
 *
 * This means that all a, b, c, d, e and f are live until the corresponding
 * gradient operations are performed.
 *
 * With the RecomputationInput instructions (checkpointing output of c), we
 * can represnet this graphs as:
 * // Checkpointed parameters
 * chkp1 = param(0)
 * // Recomp
 * a = op_x(in0, in1)
 * b = op_z(a)
 * c = op_y(b)
 * ri1 = recomputation-input(chkp1, c)
 * d = op_x(ri1, in2)
 * e = op_z(d)
 * f = op_y(e)
 *
 * // Backward
 * f' = op_y_grad(e, f)
 * e' = op_z_grad(d, f')
 * d_lhs' = op_x_grad_lhs(in2, e')
 * d_rhs' = op_x_grad_rhs(ri1, e')
 * c' = op_y_grad(b, d_lhs')
 * b' = op_z_grad(a, c')
 * a_rhs' = op_x_grad_rhs(in0, b')
 *
 * This pass then:
 * - Removes the recomputation input instructions, and identifies 'cri' and 'oi'
 * instructions for each recomputation input instruction.
 * - Finds all the instructions which have a data path to 'oi' and makes sure
 * they are scheduled as late as possible
 *
 * This means that all the recomputation instructions and their backward
 * instructions can be executed together before other recomputation
 * instructions.
 *
 * This means that we can represent this graph as (where the order implied the
 * control dependencies were inserted into the graph):
 *
 * chkp1 = param(0)
 * d = op_x(chkp1, in2)
 * e = op_z(d)
 * f = op_y(e)
 * f' = op_y_grad(e, f)
 * e' = op_z_grad(d, f')
 * d_lhs' = op_x_grad_lhs(in2, e')
 * d_rhs' = op_x_grad_rhs(chkp1, e')
 *
 * a = op_x(in0, in1)
 * b = op_z(a)
 * c' = op_y_grad(b, d_lhs')
 * b' = op_z_grad(a, c')
 * a_rhs' = op_x_grad_rhs(in0, b')
 *
 * This means that all d, e and f are first live until the corresponding
 * gradient operations are performed and then a, b are (and c is the
 * checkpointed parameter to the computation).
 */
class RecomputationInputRemover : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "recomputation-input-remover";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTATION_INPUT_REMOVER_H_
