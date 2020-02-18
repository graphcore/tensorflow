/* Copyright 2017-2019 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <poplin/Convolution.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/conv_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor RemoveGroupsDimensionFromWeights(const poplin::ConvParams& p,
                                                const poplar::Tensor& t,
                                                bool flipped) {
  poplar::Tensor out = t;
  return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
}

// This function operates on the poplibs format weights (GOI...)
poplar::Tensor AddGroupsDimensionToWeights(const poplin::ConvParams& p,
                                           const poplar::Tensor& t,
                                           bool flipped) {
  poplar::Tensor out = t;

  unsigned int out_dim = flipped ? 1 : 0;
  unsigned int in_dim = 1 - out_dim;

  if (p.getNumConvGroups() == 1) {
    // Non-grouped case
    return out.reshapePartial(0, 0, {1});
  } else {
    unsigned int chan_div[2];
    chan_div[in_dim] = out.dim(in_dim) / p.getNumInputChansPerConvGroup();
    chan_div[out_dim] = out.dim(out_dim) / p.getNumOutputChansPerConvGroup();

    // OI... ->(GO)(GI)...
    out = out.reshapePartial(0, 2,
                             {chan_div[0], out.dim(0) / chan_div[0],
                              chan_div[1], out.dim(1) / chan_div[1]});

    // (GO)(GI)... -> (GG)OI...
    out = out.dimShufflePartial({2}, {1});

    // (GG)OI... -> GOI...
    return out.reshapePartial(0, 2, {out.dim(0) * out.dim(1)});
  }
}


}  // namespace poplarplugin
}  // namespace xla
