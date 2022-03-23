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
#include "ipu/poplar_executable_runner.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage_config.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "ipu/poplar_command_line_utils.h"

ABSL_FLAG(ipu::BinaryFiles, binaries, ipu::BinaryFiles(),
          "List of binary files containing metadata, binaries, weights,"
          " inputs, feeds, etc.");
ABSL_FLAG(bool, print_output, false,
          "Print the content of the tensors to stdout (Info + data)");
ABSL_FLAG(bool, print_data, false,
          "Print the content of the tensors to stdout (Data only)");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(bool, list_handles, false,
          "List the names of the handles present in --binaries");
ABSL_FLAG(bool, list_tensors, false,
          "List the names of the tensors present in --binaries");
ABSL_FLAG(bool, list_feeds, false,
          "List the names of the feeds present in --binaries");
ABSL_FLAG(bool, list, false, "List all the objects present in --binaries");
ABSL_FLAG(std::string, output_folder, "",
          "Where to export the content of the tensors");
ABSL_FLAG(std::string, tensor, "", "Name of the tensor to export");
ABSL_FLAG(std::string, feed, "", "Name of the feed to export");

bool HelpFilter(absl::string_view filename) {
  return filename.find(__FILE__) != absl::string_view::npos;
}

int main(int argc, char** argv) {
  // Setting a custom filter is required for the help to be displayed when
  // --help is passed.
  absl::FlagsUsageConfig config;
  config.contains_help_flags = &HelpFilter;
  absl::SetFlagsUsageConfig(config);

  absl::ParseCommandLine(argc, argv);

  const ipu::BinaryFiles binaries = absl::GetFlag(FLAGS_binaries);
  const bool print_output = absl::GetFlag(FLAGS_print_output);
  const bool print_data = absl::GetFlag(FLAGS_print_data);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const bool list_handles = absl::GetFlag(FLAGS_list_handles);
  const bool list_tensors = absl::GetFlag(FLAGS_list_tensors);
  const bool list_feeds = absl::GetFlag(FLAGS_list_feeds);
  const bool list_all = absl::GetFlag(FLAGS_list);
  const std::string output_folder = absl::GetFlag(FLAGS_output_folder);
  const std::string tensor_name = absl::GetFlag(FLAGS_tensor);
  const std::string feed_name = absl::GetFlag(FLAGS_feed);

  ipu::LogContext::EnableInfo(verbose);

  ERROR_ON_MSG(!output_folder.empty() && !ipu::CreateDirIfNeeded(output_folder),
               "Failed to create output folder '" << output_folder << "'");

  ERROR_ON_MSG(binaries.filenames.empty(),
               "--binaries needs to point at "
               "one or more folders or bin / ipu_bin files");

  ipu::BinaryLoader loader;
  for (auto file : binaries.filenames) {
    loader.LoadFile(file);
  }
  if (list_all || list_tensors || list_feeds || list_handles) {
    if (list_all || list_tensors) {
      std::cout << "List of tensors:\n"
                << absl::StrJoin(
                       loader.GetObjectSummaries(ipu::ObjectType::Tensor), "")
                << std::endl;
    }
    if (list_all || list_feeds) {
      std::cout << "List of feeds:\n"
                << absl::StrJoin(
                       loader.GetObjectSummaries(ipu::ObjectType::Feed), "")
                << std::endl;
    }
    if (list_all || list_handles) {
      for (auto name :
           loader.GetObjectNames(ipu::ObjectType::PoplarExecutable)) {
        std::cout << "List of handles in executable " << name << ":\n";
        auto exe = loader.CreateExecutable(name);
        std::cout << exe->StreamsList(true) << std::endl;
      }
    }
    return 0;
  }
  ERROR_ON_MSG(!tensor_name.empty() && !feed_name.empty(),
               "Only one of --tensor or --feed can be provided");
  ERROR_ON_MSG(tensor_name.empty() && feed_name.empty(),
               "At least one of --tensor or --feed must be provided");

  if (!tensor_name.empty()) {
    std::unique_ptr<ipu::StreamReader> reader =
        loader.GetTensorStream(tensor_name);
    ipu::Tensor out{*reader};
    if (print_data) {
      std::cout << tensor_name << " = ";
      out.SaveDataToJsonStream(&std::cout);
      std::cout << std::endl;
    } else if (print_output) {
      std::cout << "Value of " << tensor_name << ":\n";
      std::cout << out.ToString() << std::endl;
    }
    if (!output_folder.empty()) {
      out.SaveDataToJsonFile(absl::StrCat(
          output_folder, "/", ipu::SanitizeName(tensor_name), ".json"));
    }
  }
  if (!feed_name.empty()) {
    std::shared_ptr<ipu::StreamReader> reader =
        loader.CreateInfeedStreamReader(feed_name);
    ipu::InfeedStream feed{reader};
    ipu::Tensor tensor{feed.Info()};

    for (int64_t i = 0; i < feed.NumTensors(); i++) {
      feed.LoadTensor(tensor.Data());
      if (print_data) {
        std::cout << feed_name << "[" << i << "] = ";
        tensor.SaveDataToJsonStream(&std::cout);
        std::cout << std::endl;
      } else if (print_output) {
        std::cout << "Value of " << feed_name << " tensor=" << i << ":\n";
        std::cout << tensor.ToString() << std::endl;
      }
      if (!output_folder.empty()) {
        tensor.SaveDataToJsonFile(absl::StrCat(
            output_folder, "/", ipu::SanitizeName(feed_name), ".", i, ".json"));
      }
      feed.MoveToNextTensor();
    }
  }
  std::cout << "\nExport complete!\n";
  return 0;
}
