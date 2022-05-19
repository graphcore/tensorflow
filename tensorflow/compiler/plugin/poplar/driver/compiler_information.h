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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_INFORMATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_INFORMATION_H_

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

// This structure contains all information which is used during the
// modifications/optimisation of the XLA graph.
struct CompilerInformation {
  int64_t max_all_reduce_buffer_size = 0;

  int64_t max_reduce_scatter_buffer_size = 0;

  int64_t max_inter_ipu_copies_buffer_size = 0;

  int64_t max_reduce_many_buffer_size = 0;

  int64_t max_all_gather_buffer_size = 0;

  int64_t max_send_recv_cluster_size = 0;

  int64_t max_scheduler_lookahead_depth = 1;

  int64_t max_scheduler_search_space_size = 64;

  int64_t minimum_remote_tensor_size = 128;

  CompilerInformation& set_max_all_reduce_buffer_size(int64_t val) {
    max_all_reduce_buffer_size = val;
    return *this;
  }

  CompilerInformation& set_max_reduce_scatter_buffer_size(int64_t val) {
    max_reduce_scatter_buffer_size = val;
    return *this;
  }

  CompilerInformation& set_max_inter_ipu_copies_buffer_size(int64_t val) {
    max_inter_ipu_copies_buffer_size = val;
    return *this;
  }

  CompilerInformation& set_max_reduce_many_buffer_size(int64_t val) {
    max_reduce_many_buffer_size = val;
    return *this;
  }

  CompilerInformation& set_max_all_gather_buffer_size(int64_t val) {
    max_all_gather_buffer_size = val;
    return *this;
  }

  CompilerInformation& set_max_send_recv_cluster_size(int64_t val) {
    max_send_recv_cluster_size = val;
    return *this;
  }

  CompilerInformation& set_max_scheduler_lookahead_depth(int64_t val) {
    max_scheduler_lookahead_depth = val;
    return *this;
  }

  CompilerInformation& set_max_scheduler_search_space_size(int64_t val) {
    max_scheduler_search_space_size = val;
    return *this;
  }

  CompilerInformation& set_minimum_remote_tensor_size(int64_t val) {
    minimum_remote_tensor_size = val;
    return *this;
  }
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_INFORMATION_H_
