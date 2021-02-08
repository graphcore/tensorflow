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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::int64;

namespace xla {
namespace poplarplugin {

enum class SyntheticDataCategory {
  Seed,
  Infeed,
  Outfeed,
  HostEmbedding,
  Parameters,
  Unknown
};

class PoplarXlaFlags {
 public:
  static const PoplarXlaFlags& Get();
  // Getter for the flag usage string.
  static const std::string GetFlagUsageString();
  // Display all the flags infos.
  bool help = false;

  // If enabled, there will be no data transfers between the host and the
  // IPU(s).
  bool use_synthetic_data = false;

  // These values prevent a subset of the IPU/host data transfers from
  // happening.
  std::set<SyntheticDataCategory> synthetic_data_categories;

  // If enabled when using synthetic data, all the inputs to the graph will be
  // initialized to the value passed on the IPU.
  std::string synthetic_data_initializer = "";

  // If enabled, this computation will be executed on the IPU model.
  bool use_ipu_model = false;

  // If set to non-negative, the cycle count for the execution of the main graph
  // will be logged (on the specified tile).
  int log_cycle_count = -1;

  // When trying to convert a while loop to a repeat loop, we can try and use a
  // brute force method to simulate the conditional part of the while and find
  // the number of iterations. This flag sets how many iterations of the while
  // loop we should try and brute force it for (default 128).
  int64 while_loop_brute_force_max_trip_count = 128;

  // The maximum number of threads Poplar should use during compilation of the
  // graph.
  int64 max_compilation_threads = -1;

  // The maximum number of threads which each infeed queue is allowed to use
  // when accessing data from datasets.
  int64 max_infeed_threads = -1;

  // Path to a directory where the Poplar vertex graph should be saved to.
  std::string save_vertex_graph = "";

  // Path to a directory where the Poplar interval report should be saved to.
  std::string save_interval_report = "";

  // Path to the executable cache.
  std::string executable_cache_path = "";

  // Path for the tensormap files
  std::string tensor_map_file_path = "";

  // Dump the schedule graph as a dot to VLOG.
  bool dump_schedule_as_dot = false;

  // Use the fallback scheduler instead of the default one.
  bool fallback_scheduler = false;

  // Allow/disallow nans during graph construction.
  bool allow_nans = false;

  // When true, the infeed callback will return immediately without providing
  // any real data
  bool null_data_feed = false;

  // When set, and profiling is enabled, then a text summary of the profile will
  // be dumped into the standard output, in addition to the normal report
  // processing.
  bool dump_text_reports_to_stdio = false;

  // Whether to show the compilation progress bar.
  std::string show_progress_bar = "false";

  // Stores all the values as a string.
  std::string as_string = "";

  // Return the hash for all the flags which affect the graph generation and
  // compilation only.
  std::size_t hlo_hash;

 private:
  PoplarXlaFlags();

  // copy of the synthetic_data_categories argument, used for hashing.
  std::string raw_synthetic_data_categories = "";
};

;
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DIRVER_TOOLS_FALGS_H_
