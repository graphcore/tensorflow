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
#include "tensorflow/compiler/plugin/poplar/tools/poplar_executable_runner.h"

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

struct MetadataFile {
  std::string filename;
};

struct ExecutableFile {
  std::string filename;
};

struct WeightsFolder {
  std::string folder;
};

struct InputFiles {
  bool Contains(const std::string name) const {
    return files.find(name) != files.end();
  }
  std::map<std::string, std::string> files;
};

struct InfeedFiles {
  bool Contains(const std::string name) const {
    return files.find(name) != files.end();
  }
  std::map<std::string, std::string> files;
};

std::string AbslUnparseFlag(ExecutableFile f) {
  return absl::UnparseFlag(f.filename);
}
std::string AbslUnparseFlag(MetadataFile f) {
  return absl::UnparseFlag(f.filename);
}

std::string AbslUnparseFlag(WeightsFolder f) {
  return absl::UnparseFlag(f.folder);
}

std::string AbslUnparseFlag(InputFiles f) {
  return absl::StrCat(
      "{", absl::StrJoin(f.files, ", ", absl::PairFormatter("=")), "}");
}

std::string AbslUnparseFlag(InfeedFiles f) {
  return absl::StrCat(
      "{", absl::StrJoin(f.files, ", ", absl::PairFormatter("=")), "}");
}

bool AbslParseFlag(absl::string_view text, InputFiles* f, std::string* error) {
  std::vector<std::string> inputs;
  if (!absl::ParseFlag(text, &inputs, error)) {
    return false;
  }
  for (auto& input : inputs) {
    std::vector<std::string> pair = absl::StrSplit(input, '=');
    if (pair.size() != 2) {
      *error = absl::StrCat("Invalid 'input_name=file' pair '", input,
                            "' in --input_data value '", text, "'");
      return false;
    }
    if (!std::ifstream(pair[1]).is_open()) {
      *error = absl::StrCat("Cannot open input_data file '", pair[1],
                            "' from --input_data '", text, "'");
      return false;
    }
    f->files[pair[0]] = pair[1];
  }

  return true;
}

bool AbslParseFlag(absl::string_view text, InfeedFiles* f, std::string* error) {
  std::vector<std::string> feeds;
  if (!absl::ParseFlag(text, &feeds, error)) {
    return false;
  }
  for (auto& feed : feeds) {
    std::vector<std::string> pair = absl::StrSplit(feed, '=');
    if (pair.size() != 2) {
      *error = absl::StrCat("Invalid 'infeed_name=file' pair '", feed,
                            "' in --infeed_data value '", text, "'");
      return false;
    }
    f->files[pair[0]] = pair[1];
  }

  return true;
}

bool AbslParseFlag(absl::string_view text, MetadataFile* f,
                   std::string* error) {
  if (!absl::ParseFlag(text, &f->filename, error)) {
    return false;
  }
  if (!std::ifstream(f->filename).is_open()) {
    *error = absl::StrCat("Could not open metadata file '", f->filename, "'.");
    return false;
  }
  return true;
}

bool AbslParseFlag(absl::string_view text, ExecutableFile* f,
                   std::string* error) {
  if (!absl::ParseFlag(text, &f->filename, error)) {
    return false;
  }
  if (!std::ifstream(f->filename).is_open()) {
    *error = absl::StrCat("Could not open Poplar Executable file '",
                          f->filename, "'");
    return false;
  }
  return true;
}

bool AbslParseFlag(absl::string_view text, WeightsFolder* f,
                   std::string* error) {
  if (!absl::ParseFlag(text, &f->folder, error)) {
    return false;
  }
  DIR* dp = opendir(f->folder.c_str());
  if (dp == NULL) {
    *error = absl::StrCat("'", f->folder, "' is not a valid directory");
    return false;
  }
  struct dirent* dirp;
  bool found_data = false;
  auto name_ends_with = [](const std::string& name,
                           const std::string& to_match) {
    return name.size() >= to_match.size() &&
           name.compare(name.size() - to_match.size(), to_match.size(),
                        to_match) == 0;
  };
  while ((dirp = readdir(dp)) != NULL) {
    std::string file(dirp->d_name);
    if (name_ends_with(file, ".data")) {
      found_data = true;
      break;
    }
  }
  if (!found_data) {
    *error = absl::StrCat("Couldn't find any .data file in the directory '",
                          f->folder, "'");
  }
  closedir(dp);
  return found_data;
}

ABSL_FLAG(MetadataFile, model_metadata, MetadataFile(),
          "Path to the json file containing the metadata of the model to run.");
ABSL_FLAG(ExecutableFile, model_executable, ExecutableFile(),
          "Path to the ipu_bin.poplar_exec file containing the Poplar binaries "
          "of the model to run");
ABSL_FLAG(WeightsFolder, weights_path, WeightsFolder(),
          "Path to the folder where the weights (.data files) can be found.");
ABSL_FLAG(InputFiles, input_data, InputFiles(),
          "List of input_name=input_file pairs for the given model. e.g "
          "--input_data=\"input_0=/tmp/data/input_0.json\".");
ABSL_FLAG(InfeedFiles, infeed_data, InfeedFiles(),
          "List of infeed_name=infeed_file pairs for the given model. e.g "
          "--infeed_data=\"training_feed=/tmp/data/training_feed.bin\". "
          "(Note: the tuple index will automatically be added by the Runner.)");
ABSL_FLAG(bool, print_output, false,
          "Print the content of the output buffers to stdout");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(bool, strict, false,
          "Enable strict mode: all the input data files must be provided by "
          "--input_data.");
ABSL_FLAG(std::string, output_folder, "",
          "Where to save the content of the output tensors");

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

  const std::string metadata_filename =
      absl::GetFlag(FLAGS_model_metadata).filename;
  const std::string poplar_executable_filename =
      absl::GetFlag(FLAGS_model_executable).filename;
  const std::string weights_path = absl::GetFlag(FLAGS_weights_path).folder;
  const InputFiles input_data = absl::GetFlag(FLAGS_input_data);
  const InfeedFiles infeed_data = absl::GetFlag(FLAGS_infeed_data);
  const bool print_output = absl::GetFlag(FLAGS_print_output);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const bool strict = absl::GetFlag(FLAGS_strict);
  const std::string output_folder = absl::GetFlag(FLAGS_output_folder);

  ipu::LogContext::EnableInfo(verbose);

  if (poplar_executable_filename.empty()) {
    std::cout << "ERROR: --model_executable needs to be set to a valid "
                 "ipu_bin.poplar_exec file\n";
    return -1;
  }
  if (metadata_filename.empty()) {
    std::cout << "ERROR: --model_metadata needs to be set to a valid "
                 "json file\n";
    return -1;
  }
  if (weights_path.empty()) {
    std::cout << "ERROR: --weights_path needs to be set to a valid folder\n";
    return -1;
  }

  std::cout << "\n[Parsing Graph's metadata]\n";
  ipu::JsonParser metadata{metadata_filename};
  ipu::TensorManager tensors{metadata};

  std::cout << "\n[Initialising IPU]\n";
  ipu::DeviceManager manager;
  poplar::Device device = manager.GetDevice(tensors.Config().NumIpus(),
                                            tensors.Config().OptionFlags());
  std::cout << "\n[Loading Poplar executable]\n";
  ipu::Executable exe{poplar_executable_filename};
  std::cout << "List of streams:\n";
  exe.PrintStreams();

  tensors.LoadParameters(weights_path);
  std::list<ipu::Tensor*> inputs = tensors.InputDataTensors();
  std::list<std::string> extra_inputs, inputs_missing;
  absl::c_transform(input_data.files, std::back_inserter(extra_inputs),
                    [](const std::pair<std::string, std::string>& pair) {
                      return pair.first;
                    });

  for (auto input : inputs) {
    if (!input_data.Contains(input->Info().Name())) {
      inputs_missing.push_back(input->Info().Name());
    } else {
      ipu::LogContext ctx{
          absl::StrCat("Loading input '", input->Info().Name(), "' from ",
                       input_data.files.at(input->Info().Name()))};
      extra_inputs.erase(absl::c_find(extra_inputs, input->Info().Name()));
      input->LoadDataFromJson(input_data.files.at(input->Info().Name()));
    }
  }
  if (strict && (!inputs_missing.empty() || !extra_inputs.empty())) {
    absl::c_for_each(inputs_missing, [](const std::string& input) {
      std::cout << "ERROR: No data provided for input_data '" << input
                << std::endl
                << std::flush;
    });
    absl::c_for_each(extra_inputs, [](const std::string& input) {
      std::cout << "ERROR: Provided a data file for the input '" << input
                << "' but this input cannot be found in the model provided.\n"
                << std::flush;
    });
    return -1;
  } else {
    absl::c_for_each(inputs_missing, [](const std::string& input) {
      std::cout << "WARNING: No data provided for input_data '" << input
                << "': the model will be run using a buffer of 0 values.\n"
                << std::flush;
    });
    absl::c_for_each(extra_inputs, [](const std::string& input) {
      std::cout << "WARNING: Provided a data file for the input '" << input
                << "' but this input cannot be found in the model provided.\n"
                << std::flush;
    });
  }

  std::list<std::string> extra_infeeds, infeeds_missing;
  absl::c_transform(infeed_data.files, std::back_inserter(extra_infeeds),
                    [](const std::pair<std::string, std::string>& pair) {
                      return pair.first;
                    });

  for (auto& infeed : tensors.MutableInfeeds()) {
    if (!infeed_data.Contains(infeed.Name())) {
      infeeds_missing.push_back(infeed.Name());
    } else {
      ipu::LogContext ctx{absl::StrCat("Loading infeed '", infeed.Name(),
                                       "' from ",
                                       infeed_data.files.at(infeed.Name()))};
      extra_infeeds.erase(absl::c_find(extra_infeeds, infeed.Name()));
      infeed.LoadDataFromBin(infeed_data.files.at(infeed.Name()));
    }
  }
  if (strict && (!infeeds_missing.empty() || !extra_infeeds.empty())) {
    absl::c_for_each(infeeds_missing, [](const std::string& infeed) {
      std::cout << "ERROR: No data provided for infeed_data '" << infeed
                << std::endl
                << std::flush;
    });
    absl::c_for_each(extra_infeeds, [](const std::string& infeed) {
      std::cout << "ERROR: Provided a data file for the infeed '" << infeed
                << "' but this infeed cannot be found in the model provided.\n"
                << std::flush;
    });
    return -1;
  } else {
    absl::c_for_each(infeeds_missing, [](const std::string& infeed) {
      std::cout << "WARNING: No data provided for infeed_data '" << infeed
                << "': the model will be run using a buffer of 0 values.\n"
                << std::flush;
    });
    absl::c_for_each(extra_infeeds, [](const std::string& infeed) {
      std::cout << "WARNING: Provided a data file for the infeed '" << infeed
                << "' but this infeed cannot be found in the model provided.\n"
                << std::flush;
    });
  }

  tensors.ConnectStreams(exe);

  ipu::SeedManager seeds{tensors.Config()};
  seeds.ConnectStreams(exe);

  std::cout << "\n[Executing]\n";
  exe.LoadAndRun(device);

  std::cout << "\nExecution complete!\n";
  if (print_output) {
    std::cout << "Outputs:\n";
    for (auto& output : tensors.Outputs()) {
      std::cout << output.ToString() << std::endl;
    }
  }
  if (!output_folder.empty()) {
    DIR* dp = opendir(output_folder.c_str());
    if (dp == NULL) {
      if (mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) !=
          0) {
        std::cout << "Failed to create output folder '" << output_folder
                  << "'\n";
        return -1;
      }
    } else {
      closedir(dp);
    }
    for (auto& output : tensors.Outputs()) {
      output.SaveDataToJsonFile(
          absl::StrCat(output_folder, "/", output.Info().Name(), ".data"));
    }
  }

  std::cout << "Run successful\n";
  return 0;
}
