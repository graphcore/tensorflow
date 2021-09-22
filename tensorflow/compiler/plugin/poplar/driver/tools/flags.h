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

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::int64;

namespace xla {
namespace poplarplugin {

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

  // Whether to show the compilation progress bar.
  std::string show_progress_bar = "auto";

  // When using 'ON_DEMAND' connection type, configure how often to poll for the
  // device (in milliseconds) when a device is not available - defaults to
  // 1000ms. Minimum is 100ms.
  int64 on_demand_device_poll_time = 1000;

  // When using 'ON_DEMAND' connection type, configure how long to wait (in
  // milliseconds) for a device before timing out - defaults to 3600000ms (1
  // hour).
  int64 on_demand_device_timeout = 3600000;

  // When specified and when using the Poplar IPUModel target, sets the number
  // of tiles for the IPUModel device created. This flag has no effect if the
  // ``--use_ipu_model`` flag is not used. This flag is ignored if the
  // ``IPUConfig.ipu_model.tiles_per_ipu`` is set.

  int64 ipu_model_tiles = -1;

  // Synchronise the starting point of each replica's main program.
  bool sync_replica_start = false;

  // Whether to run the HLO verifier as an invariant checker before and after
  // every HLO pass.
  bool enable_hlo_verifier = false;

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
