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

#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
namespace subcomputation_graph_caching {

StatusOr<std::shared_ptr<DeferredVisitor>>
SubcomputationGraphCache::GetOrCompileSubcomputation(
    CompilerResources& res, TensorOrRemoteBufferVectors& inputs,
    const HloComputation* computation) {
  DeferredArgRBVectors deferred_inputs = ConvertInputsToDeferredInputs(inputs);
  auto itr = table_.find(computation);
  if (itr == table_.end()) {
    VLOG(2) << "Compiling sub-computation " << computation->name();
    XLA_VLOG_LINES(2, computation->ToString());

    auto order =
        computation->parent()->schedule().sequence(computation).instructions();
    std::shared_ptr<DeferredVisitor> deferred_visitor =
        std::make_shared<DeferredVisitor>(res, deferred_inputs,
                                          computation->name());

    DeferredVisitor* def_visitor =
        const_cast<DeferredVisitor*>(deferred_visitor.get());
    TF_RETURN_IF_ERROR(computation->AcceptOrdered(def_visitor, order));

    if (computation->HasSideEffect()) {
      return deferred_visitor;
    }
    itr = table_.emplace(computation, deferred_visitor).first;
  } else {
    VLOG(1) << "Computation " << computation->name()
            << " has already been compiled, reusing the code.";
  }
  return itr->second;
}
}  // namespace subcomputation_graph_caching
}  // namespace poplarplugin
}  // namespace xla
