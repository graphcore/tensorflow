/* Copyright 2017 Graphcore Ltd
 */

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/conv_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/generic_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/mapping_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/subcomputation_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_subcomputation.h"

#include <poplar/OptionFlags.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/GraphFunction.hpp>

#include <memory>

namespace xla {
namespace poplarplugin {

// This structure contains additional information required to lower the graph
// from an XLA graph to a poplar graph.
struct CompilerResources {
  std::unique_ptr<poplar::Graph> main_graph;

  std::vector<poplar::Graph> shard_graphs;

  CompilerAnnotations annotations;

  CompilerInformation information;

  poplin::PlanningCache convolution_cache;

  poplin::matmul::PlanningCache dot_cache;

  const IpuOptions::FloatingPointBehaviour global_floating_point_behaviour;

  const poplar::OptionFlags default_conv_options;

  const poplar::OptionFlags default_matmul_options;

  const poplar::OptionFlags default_pooling_options;

  bool clear_matmul_pass_type;

  bool disable_graph_convolution_caching;

  uint32 replication_factor;

  bool merge_infeed_io_copies;

  bool always_rearrange_copies_on_host;

  std::map<std::string, TensorMap> tensor_maps;

  LinearMapperState linear_mapping_state;

  conv_graph_caching::ConvolutionGraphCache conv_graph_cache;

  conv_graph_caching::BwdWeightGraphCache bwd_weight_graph_cache;

  generic_graph_caching::GenericGraphCache graph_cache;

  subcomputation_graph_caching::SubcomputationGraphCache subcomputation_cache;

  CompilerResources(
      const poplar::OptionFlags& conv_options,
      const poplar::OptionFlags& matmul_options,
      const poplar::OptionFlags& pooling_options, bool clear_matmul_pass_type,
      bool disable_graph_convolution_caching, bool merge_infeed_io_copies,
      uint32 replication_factor, int64 max_all_reduce_buffer_size,
      int64 max_inter_ipu_copies_buffer_size,
      int64 max_scheduler_lookahead_depth,
      int64 max_scheduler_search_space_size, HloModule* module,
      const IpuOptions::FloatingPointBehaviour& floating_point_behaviour,
      bool always_rearrange_copies_on_host)
      : annotations(module),
        information(
            max_all_reduce_buffer_size, max_inter_ipu_copies_buffer_size,
            max_scheduler_lookahead_depth, max_scheduler_search_space_size),
        global_floating_point_behaviour(floating_point_behaviour),
        default_conv_options(conv_options),
        default_matmul_options(matmul_options),
        default_pooling_options(pooling_options),
        clear_matmul_pass_type(clear_matmul_pass_type),
        disable_graph_convolution_caching(disable_graph_convolution_caching),
        replication_factor(replication_factor),
        merge_infeed_io_copies(merge_infeed_io_copies),
        always_rearrange_copies_on_host(always_rearrange_copies_on_host) {}
};

}  // namespace poplarplugin
}  // namespace xla

#endif
