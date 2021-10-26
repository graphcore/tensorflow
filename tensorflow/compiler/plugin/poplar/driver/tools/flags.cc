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

#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <mutex>
#include <sstream>

namespace xla {
namespace poplarplugin {
namespace {
static const char* const kEnvVarName = "TF_POPLAR_FLAGS";

SyntheticDataCategory SyntheticDataCategoryFromName(absl::string_view name) {
#define RETURN_CATEGORY_FOR_NAME(NAME, CATEGORY) \
  if (name == #NAME) return CATEGORY;

  RETURN_CATEGORY_FOR_NAME(hostembedding, SyntheticDataCategory::HostEmbedding);
  RETURN_CATEGORY_FOR_NAME(infeed, SyntheticDataCategory::Infeed);
  RETURN_CATEGORY_FOR_NAME(outfeed, SyntheticDataCategory::Outfeed);
  RETURN_CATEGORY_FOR_NAME(parameters, SyntheticDataCategory::Parameters);
  RETURN_CATEGORY_FOR_NAME(seed, SyntheticDataCategory::Seed);

#undef RETURN_CATEGORY_FOR_NAME

  return SyntheticDataCategory::Unknown;
}

std::set<SyntheticDataCategory> ParseSyntheticDataCategoriesArg(
    const std::string& arg) {
  std::set<SyntheticDataCategory> categories;

  auto splitter = absl::StrSplit(arg, ',', absl::SkipWhitespace());
  for (auto split : splitter) {
    split = absl::StripAsciiWhitespace(split);

    categories.insert(SyntheticDataCategoryFromName(split));
  }

  return categories;
}

absl::flat_hash_map<std::string, std::string> GetFlagUsage() {
  static absl::flat_hash_map<std::string, std::string> flag_usage = {
      {"help", "Display all the flags infos. (bool)"},
      {"use_synthetic_data",
       "If enabled, there will be no data transfers between the host and the "
       "IPU(s). (bool)"},
      {"synthetic_data_categories",
       "A comma separated list of values that can be used for finer control "
       "over which kinds of data are made synthetic. Valid values are.. "
       "infeed/outfeed/seed/hostembedding/parameters."},
      {"synthetic_data_initializer",
       "If set when using synthetic data, all the inputs to the graph can be "
       "initialized directly on the IPU either randomly "
       "(synthetic_data_initializer=random, uniform, normal) or to a constant "
       "value X (synthetic_data_initializer=int)."},
      {"use_ipu_model",
       "If enabled, this computation will be executed on the IPU model. "
       "(bool)"},
      {"log_cycle_count",
       "The tile to log number of cycles for main graph. "
       "No logging will be done if negative. "
       "This may be used as an alternative to profiling for graphs with "
       "dynamic control flow. "
       "(int=-1)"},
      {"while_loop_brute_force_max_trip_count",
       "When trying to convert a while loop to a repeat loop, we can try and "
       "use a brute force method to simulate the conditional part of the while "
       "and find the number of iterations. This flag sets how many iterations "
       "of the while loop we should try and brute force it for. (int=128)"},
      {"max_compilation_threads",
       "The maximum number of threads Poplar should use during compilation of "
       "the graph. Negative value allows Poplar to pick the number of threads "
       "automatically. (int=-1)"},
      {"max_infeed_threads",
       "The maximum number of threads which each infeed queue is allowed to "
       "use when accessing data from datasets. Negative value allows the "
       "infeed to automatically pick the number of threads. (int=-1)"},
      {"save_vertex_graph",
       "Path to a directory where the Poplar vertex graphs should be saved to. "
       "(path)"},
      {"save_interval_report",
       "Path to a directory where the Poplar interval reports should be saved "
       "to. (path)"},
      {"executable_cache_path", "Path to the executable cache. (path)"},
      {"dump_schedule_as_dot", "Dumps the scheduler graph as a dot file."},
      {"tensor_map_file_path", "Directory for tensor map dump files."},
      {"null_data_feed",
       "Don't provide data to an infeed, for performance measurement."},
      {"fallback_scheduler",
       "Use the sync list scheduler rather than the default one."},
      {"show_progress_bar",
       "Whether to show the compilation progress bar. Either \"true\", "
       "\"false\" or \"auto\". Defaults to \"auto\"."},
      {"on_demand_device_poll_time",
       "When using \"ON_DEMAND\" connection type, configure how often to poll "
       "for the device (in miliseconds) when a device is not available - "
       "defaults to 1000ms. Minimum is 100ms."},
      {"on_demand_device_timeout",
       "When using 'ON_DEMAND' connection type, configure how long to wait (in "
       "milliseconds) for a device before timing out - defaults to 3600000ms "
       "(1 hour)."},
      {"ipu_model_tiles",
       "When specified and when using the Poplar IPUModel target, sets the "
       "number of tiles for the IPUModel device created. This flag has no "
       "effect if the ``--use_ipu_model`` flag is not used. This flag is "
       "ignored if the ``IPUConfig.ipu_model.tiles_per_ipu`` is set."},
      {"sync_replica_start",
       "Whether to synchronise the starting point of each replica's main "
       "program."},
      {"enable_hlo_verifier",
       "Whether to run the HLO verifier as an invariant checker before and "
       "after every HLO pass."},
      {"disable_poplar_version_check",
       "If set, the Poplar version check will be disabled."},
      {"allow_nans", "will allow NaNs."}};
  return flag_usage;
}
}  // namespace

PoplarXlaFlags::PoplarXlaFlags() {
  // Struct for deprecated flags.
  struct DeprecatedFlags {
    bool add_all_reduce_copies = false;
    bool force_replicated_mode = false;
    std::string save_oom_profiler = "";
    bool dump_text_reports_to_stdio = false;
  };

  DeprecatedFlags deprecated_flags;
  auto flag_usage = GetFlagUsage();

  std::vector<tensorflow::Flag> flag_list = {
#define ADD_FLAG(FLAG_NAME) \
  tensorflow::Flag(#FLAG_NAME, &FLAG_NAME, flag_usage.at(#FLAG_NAME)),
#define ADD_DEPRECATED_FLAG(FLAG_NAME) \
  tensorflow::Flag(#FLAG_NAME, &deprecated_flags.FLAG_NAME, ""),
      // clang-format off
    ADD_FLAG(help)
    ADD_FLAG(use_synthetic_data)
    ADD_FLAG(synthetic_data_initializer)
    ADD_FLAG(use_ipu_model)
    ADD_FLAG(log_cycle_count)
    ADD_FLAG(while_loop_brute_force_max_trip_count)
    ADD_FLAG(max_compilation_threads)
    ADD_FLAG(max_infeed_threads)
    ADD_FLAG(save_vertex_graph)
    ADD_FLAG(save_interval_report)
    ADD_FLAG(executable_cache_path)
    ADD_FLAG(dump_schedule_as_dot)
    ADD_FLAG(tensor_map_file_path)
    ADD_FLAG(fallback_scheduler)
    ADD_FLAG(allow_nans)
    ADD_FLAG(null_data_feed)
    ADD_FLAG(show_progress_bar)
    ADD_FLAG(on_demand_device_poll_time)
    ADD_FLAG(on_demand_device_timeout)
    ADD_FLAG(ipu_model_tiles)
    ADD_FLAG(sync_replica_start)
    ADD_FLAG(enable_hlo_verifier)
    ADD_FLAG(disable_poplar_version_check)

    // Deprecated flags.
    ADD_DEPRECATED_FLAG(dump_text_reports_to_stdio)
    ADD_DEPRECATED_FLAG(add_all_reduce_copies)
    ADD_DEPRECATED_FLAG(force_replicated_mode)
    ADD_DEPRECATED_FLAG(save_oom_profiler)

// clang-format on
#undef ADD_FLAG
#undef ADD_DEPRECATED_FLAG
  };

  const auto synthetic_data_category_hook = [this](const std::string& arg) {
    this->raw_synthetic_data_categories = arg;
    this->synthetic_data_categories = ParseSyntheticDataCategoriesArg(arg);
    return true;
  };
  flag_list.emplace_back(tensorflow::Flag(
      "synthetic_data_categories", synthetic_data_category_hook,
      /*default*/ "", flag_usage["synthetic_data_categories"]));

  xla::ParseFlagsFromEnvAndDieIfUnknown(kEnvVarName, flag_list);

  // Store all the flags as a string.
  as_string = "";
  if (const char* flag_buffer = std::getenv(kEnvVarName)) {
    as_string = flag_buffer;
    VLOG(1) << kEnvVarName << " is set to '" << as_string << "'";
  }

  const auto invalid_synthetic_data_category =
      synthetic_data_categories.count(SyntheticDataCategory::Unknown) != 0;
  if (invalid_synthetic_data_category) {
    LOG(FATAL) << "Using an unknown synthetic data category in: '"
               << raw_synthetic_data_categories << "'" << std::endl
               << "Usage: " << flag_usage["synthetic_data_categories"];
  }

  const auto not_using_synthetic_data =
      !use_synthetic_data && synthetic_data_categories.empty();
  if (not_using_synthetic_data && !synthetic_data_initializer.empty()) {
    LOG(FATAL) << "The flag \"synthetic_data_initializer\" can only be used "
                  "in combination with \"use_synthetic_data\" or "
                  "\"synthetic_data_categories\".";
  }

  if (deprecated_flags.add_all_reduce_copies) {
    LOG(INFO)
        << "The TensorFlow Poplar flag \"add_all_reduce_copies\" is "
           "deprecated, has no effect and it will be removed in the future.";
  }

  if (deprecated_flags.force_replicated_mode) {
    LOG(INFO)
        << "The TensorFlow Poplar flag \"force_replicated_mode\" is "
           "deprecated, has no effect and it will be removed in the future.";
  }

  if (!deprecated_flags.save_oom_profiler.empty()) {
    LOG(INFO)
        << "The TensorFlow Poplar flag \"save_oom_profiler\" is "
           "deprecated, has no effect and it will be removed in the future. "
           "Out of memory report will be saved in the report directory if "
           "`profiling` is enabled and `report_directory` is set.";
  }

  if (deprecated_flags.dump_text_reports_to_stdio) {
    LOG(INFO) << "The TensorFlow Poplar flag \"dump_text_reports_to_stdio\" is"
                 " deprecated and it will be removed in the future. Use the"
                 " PopVision Graph Analyser to manually inspect profiles.";
  }

  // Hash all the flags which affect the graph generation and compilation only.
  hlo_hash =
      hash_util::hash(use_synthetic_data, raw_synthetic_data_categories,
                      synthetic_data_initializer, use_ipu_model,
                      while_loop_brute_force_max_trip_count, fallback_scheduler,
                      allow_nans, log_cycle_count, sync_replica_start,
                      ipu_model_tiles, disable_poplar_version_check);
}

void PoplarXlaFlags::ReloadFlagsForTesting() {
  auto& flags = GetMutable();

  // TF flag parsing will remove parsed flags from the env,
  // so we have to create our new flags object after resetting
  // the env, otherwise the flag might get lost if it's parsed
  // before the assignment.
  int* argc;
  std::vector<char*>* argv;
  xla::ResetFlagsFromEnvForTesting(kEnvVarName, &argc, &argv);
  flags = PoplarXlaFlags();
}

const PoplarXlaFlags& PoplarXlaFlags::Get() { return GetMutable(); }

PoplarXlaFlags& PoplarXlaFlags::GetMutable() {
  static PoplarXlaFlags poplar_xla_flags;
  return poplar_xla_flags;
}

const std::string PoplarXlaFlags::GetFlagUsageString() {
  auto flag_usage = GetFlagUsage();
  std::stringstream usage_stream;
  usage_stream << "Usage for TF_POPLAR_FLAGS is:" << std::endl;
  for (auto pair : flag_usage) {
    usage_stream << "\t--" << pair.first << ": " << pair.second << std::endl;
  }
  return usage_stream.str();
}
}  // namespace poplarplugin
}  // namespace xla
