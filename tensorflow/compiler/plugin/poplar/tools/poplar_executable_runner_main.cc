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

bool FileExists(const std::string& filename) {
  return std::ifstream(filename).is_open();
}

bool IsDir(const std::string& path) {
  DIR* dirp = opendir(path.c_str());
  if (dirp) {
    closedir(dirp);
  }
  return dirp != NULL;
}

std::string SecondsToTimeString(int64_t sec) {
  int hours = sec / 3600;
  int minutes = (sec - hours * 3600) / 60;
  sec -= hours * 3600 + minutes * 60;
  std::stringstream ss;
  if (hours > 0) {
    ss << hours << "h ";
  }
  if (minutes > 0) {
    ss << minutes << "m ";
  }
  ss << sec << "s";
  return ss.str();
}

std::string FileExtension(const std::string& filename,
                          bool no_extension_allowed = false) {
  size_t dot_pos = filename.rfind(".");
  if (dot_pos == std::string::npos) {
    ERROR_ON_MSG(!no_extension_allowed,
                 "Invalid filename '" << filename << "': no extension");
    return "";
  }
  return filename.substr(dot_pos + 1);
}

std::vector<std::string> ListFiles(const std::string& folder,
                                   std::string* error) {
  std::vector<std::string> files;
  DIR* dirp = opendir(folder.c_str());
  if (dirp == NULL) {
    *error = absl::StrCat("Can't open folder '", folder, "'");
    return {};
  }
  struct dirent* dp;
  while ((dp = readdir(dirp)) != NULL) {
    files.push_back(dp->d_name);
  }
  closedir(dirp);
  return files;
}

bool ContainsFilesWithExtension(const std::string& folder,
                                const std::string& extension,
                                std::string* error) {
  auto files = ListFiles(folder, error);
  if (!error->empty()) {
    return false;
  }
  for (auto file : files) {
    if (FileExtension(file, true) == extension) {
      return true;
    }
  }
  return false;
}

std::string GetOnlyFileWithExtension(const std::string& folder,
                                     const std::string& extension,
                                     std::string* error) {
  std::string retval;
  auto files = ListFiles(folder, error);
  if (!error->empty()) {
    return {};
  }
  for (auto file : files) {
    if (FileExtension(file, true) == extension) {
      if (!retval.empty()) {
        *error =
            absl::StrCat("More than one file have the extension '", extension,
                         "': [", absl::StrJoin(files, ", "), "]");
        return {};
      }
      retval = file;
    }
  }
  if (retval.empty()) {
    *error = absl::StrCat("No file was found with the extension '", extension,
                          "': [", absl::StrJoin(files, ", "), "]");
  }
  return retval;
}

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
    if (!FileExists(pair[1])) {
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
  if (IsDir(f->filename)) {
    ERROR_ON(!error->empty());
    const std::string folder = f->filename;
    const std::string filename =
        GetOnlyFileWithExtension(folder, "json", error);
    if (filename.empty()) {
      return false;
    }
    f->filename = absl::StrCat(folder, "/", filename);
  }
  if (!FileExists(f->filename)) {
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
  if (IsDir(f->filename)) {
    ERROR_ON(!error->empty());
    const std::string folder = f->filename;
    const std::string filename =
        GetOnlyFileWithExtension(folder, "ipu_bin", error);
    if (filename.empty()) {
      return false;
    }
    f->filename = absl::StrCat(folder, "/", filename);
  }
  if (!FileExists(f->filename)) {
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
  if (!IsDir(f->folder)) {
    *error = absl::StrCat("'", f->folder, "' is not a valid directory");
    return false;
  }
  if (!ContainsFilesWithExtension(f->folder, "data", error)) {
    if (error->empty()) {
      *error = absl::StrCat("Couldn't find any .data file in the directory '",
                            f->folder, "'");
    }
    return false;
  }
  return true;
}

ABSL_FLAG(MetadataFile, model_metadata, MetadataFile(),
          "Path to the json file containing the metadata of the model to run.");
ABSL_FLAG(ExecutableFile, model_executable, ExecutableFile(),
          "Path to the ipu_bin file containing the Poplar binaries "
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
ABSL_FLAG(int, iterations, 1, "Number of times to run the executable");
ABSL_FLAG(int, ckpt_frequency, 1, "Frequency at which to create checkpoints");
ABSL_FLAG(bool, print_output, false,
          "Print the content of the output buffers to stdout");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(bool, load_ckpt, false,
          "Load the checkpoint config from the weights folder");
ABSL_FLAG(bool, strict, false,
          "Enable strict mode: all the input data files must be provided by "
          "--input_data.");
ABSL_FLAG(std::string, output_folder, "",
          "Where to save the content of the output tensors");

bool HelpFilter(absl::string_view filename) {
  return filename.find(__FILE__) != absl::string_view::npos;
}

bool CreateDirIfNeeded(const std::string& dir) {
  DIR* dp = opendir(dir.c_str());
  if (dp == NULL) {
    if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
      return false;
    }
  } else {
    closedir(dp);
  }
  return true;
}

int main(int argc, char** argv) {
  using seconds = std::chrono::duration<float>;
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
  const bool load_ckpt = absl::GetFlag(FLAGS_load_ckpt);
  const bool strict = absl::GetFlag(FLAGS_strict);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  const int ckpt_frequency = absl::GetFlag(FLAGS_ckpt_frequency);
  const std::string output_folder = absl::GetFlag(FLAGS_output_folder);

  ipu::LogContext::EnableInfo(verbose);

  ERROR_ON_MSG(poplar_executable_filename.empty(),
               "--model_executable needs to be set to a valid "
               "ipu_bin file");
  ERROR_ON_MSG(metadata_filename.empty(),
               ": --model_metadata needs to be set to a valid "
               "json file");
  ERROR_ON_MSG(weights_path.empty(),
               "--weights_path needs to be set to a valid folder");
  ERROR_ON_MSG(!output_folder.empty() && !CreateDirIfNeeded(output_folder),
               "Failed to create output folder '" << output_folder << "'");

  auto init_start = std::chrono::high_resolution_clock::now();
  std::cout << "\n[Parsing Graph's metadata]\n";
  ipu::JsonParser metadata{metadata_filename};
  ipu::TensorManager tensors{metadata};

  std::cout << "\n[Initialising IPU]\n";
  ipu::DeviceManager manager;
  poplar::Device device = manager.GetDevice(tensors.Config().NumIpus(),
                                            tensors.Config().OptionFlags());
  std::cout << "\n[Loading Poplar executable]\n";
  ipu::Executable exe{poplar_executable_filename};
  PRINT_INFO("List of streams:\n" << exe.StreamsList());

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
      extra_inputs.erase(absl::c_find(extra_inputs, input->Info().Name()));
      const std::string filename = input_data.files.at(input->Info().Name());
      input->LoadDataFromJson(filename);
    }
  }
  if (strict && (!inputs_missing.empty() || !extra_inputs.empty())) {
    absl::c_for_each(inputs_missing, [](const std::string& input) {
      std::cout << "ERROR: No data provided for input_data '" << input << "'\n"
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
      ipu::LogContext ctx{absl::StrCat("Loading infeed '", infeed.Name(), "'")};
      extra_infeeds.erase(absl::c_find(extra_infeeds, infeed.Name()));
      infeed.InitializeDataSources(infeed_data.files.at(infeed.Name()));
    }
  }
  if (strict && (!infeeds_missing.empty() || !extra_infeeds.empty())) {
    absl::c_for_each(infeeds_missing, [](const std::string& infeed) {
      std::cout << "ERROR: No data provided for infeed_data '" << infeed
                << "'\n"
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

  if (load_ckpt) {
    tensors.LoadCheckpointMetadataFromJson(
        absl::StrCat(weights_path, "/ckpt.json"));
  }

  tensors.ConnectStreams(exe);

  ipu::SeedManager seeds{tensors.Config()};
  seeds.ConnectStreams(exe);

  auto init_end = std::chrono::high_resolution_clock::now();
  std::cout << "Done in "
            << SecondsToTimeString(seconds(init_end - init_start).count())
            << std::endl;
  std::cout << "\n[Executing]\n";
  exe.Load(device);
  for (int iteration = 0; iteration < iterations; iteration++) {
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = static_cast<float>(seconds(now - init_end).count());
    float remaining =
        iteration > 0 ? ((elapsed * iterations) / iteration) - elapsed : 0.0;
    std::cout << "Iteration " << iteration << "/" << iterations - 1
              << " Elapsed: " << SecondsToTimeString(elapsed)
              << ", Estimated remaining: " << SecondsToTimeString(remaining)
              << std::endl;
    std::string iteration_folder = output_folder;
    bool create_ckpt =
        !output_folder.empty() &&
        (iteration == (iterations - 1) || (iteration % ckpt_frequency == 0));
    if (create_ckpt && iterations > 1) {
      iteration_folder = absl::StrCat(iteration_folder, "/", iteration);
      ERROR_ON_MSG(!CreateDirIfNeeded(iteration_folder),
                   "Failed to create output folder '" << iteration_folder);
    }
    if (create_ckpt) {
      tensors.CreateCheckpointMetadataJson(
          absl::StrCat(iteration_folder, "/ckpt.json"));
      tensors.SetOutfeedsFolder(iteration_folder);
    } else {
      tensors.IgnoreOutfeeds();
    }

    exe.Run();

    if (print_output || create_ckpt) {
      exe.DeviceToHostCopy();
      if (print_output) {
        std::cout << "Outputs:\n";
        for (auto& output : tensors.Outputs()) {
          std::cout << output.ToString() << std::endl;
        }
      }
      if (create_ckpt) {
        tensors.SaveOutputsToJsonFile(iteration_folder);
      }
    }
  }
  std::cout << "\nExecution complete!\n";
  return 0;
}
