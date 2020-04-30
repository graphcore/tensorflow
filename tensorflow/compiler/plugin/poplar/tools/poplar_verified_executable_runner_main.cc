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
#include "popsec/popsec.hpp"

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

struct BinaryFiles {
  std::vector<std::string> filenames;
};

std::string AbslUnparseFlag(BinaryFiles f) {
  return absl::UnparseFlag(f.filenames);
}

bool AbslParseFlag(absl::string_view text, BinaryFiles* f, std::string* error) {
  std::vector<std::string> filenames;
  if (!absl::ParseFlag(text, &filenames, error)) {
    return false;
  }
  for (auto name : filenames) {
    if (IsDir(name)) {
      std::vector<std::string> files = ListFiles(name, error);
      if (!error->empty()) {
        return false;
      }
      for (auto file : files) {
        if (FileExtension(file, true) == "bin" ||
            FileExtension(file, true) == "ipu_bin") {
          f->filenames.push_back(absl::StrCat(name, "/", file));
        }
      }
    } else {
      if (!FileExists(name)) {
        *error = absl::StrCat("Could not open file '", name, "'.");
        return false;
      }
      f->filenames.push_back(name);
    }
  }
  return true;
}

ABSL_FLAG(BinaryFiles, binaries, BinaryFiles(),
          "List of binary files containing metadata, binaries, weights,"
          " inputs, feeds, etc.");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(bool, use_autoloader, false, "Enable the autoloader");
ABSL_FLAG(int, checkpoint_index, 0, "Index of the checkpoint");
ABSL_FLAG(int, device, -1, "Device to use (-1 for any)");
ABSL_FLAG(std::string, output_folder, ".",
          "Where to save the content of the output tensors");

bool HelpFilter(absl::string_view filename) {
  return filename.find(__FILE__) != absl::string_view::npos;
}

int main(int argc, char** argv) {
  using seconds = std::chrono::duration<float>;
  // Setting a custom filter is required for the help to be displayed when
  // --help is passed.
  absl::FlagsUsageConfig config;
  config.contains_help_flags = &HelpFilter;
  absl::SetFlagsUsageConfig(config);

  absl::ParseCommandLine(argc, argv);

  const BinaryFiles binaries = absl::GetFlag(FLAGS_binaries);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const bool use_autoloader = absl::GetFlag(FLAGS_use_autoloader);
  const int checkpoint_index = absl::GetFlag(FLAGS_checkpoint_index);
  const int requested_device_id = absl::GetFlag(FLAGS_device);
  const std::string output_folder = absl::GetFlag(FLAGS_output_folder);

  ipu::LogContext::EnableInfo(verbose);

  auto init_start = std::chrono::high_resolution_clock::now();
  ERROR_ON_MSG(output_folder.empty(), "You have to provide an --output_folder");
  ERROR_ON_MSG(!CreateDirIfNeeded(output_folder),
               "Failed to create output folder '" << output_folder << "'");

  ERROR_ON_MSG(binaries.filenames.empty(),
               "--binaries needs to point at "
               "one or more folders or bin / ipu_bin files");

  ipu::BinaryLoader loader;
  for (auto file : binaries.filenames) {
    loader.LoadFile(file);
  }

  std::unique_ptr<ipu::TensorManager> tensors =
      loader.CreateTensorManager(popsec::computeNonPayloadSize);
  std::unique_ptr<ipu::VerifiedExecutable> exe =
      loader.CreateVerifiedExecutable(use_autoloader);

  tensors->AssertAllTensorsProvided(loader);
  tensors->LoadInputsAndParameters(loader);
  tensors->LoadInfeeds(loader);
  if (tensors->ContainsCheckpoint()) {
    tensors->LoadVerifiedCheckpoint(loader, checkpoint_index);
  }

  PRINT_INFO("List of streams:\n" << exe->StreamsList());
  tensors->ConnectStreams(*exe);

  ipu::SeedManager seeds{tensors->Config()};
  seeds.ConnectStreams(*exe);

  std::cout << "\n[Initialising IPU]\n";
  ipu::DeviceManager manager;
  poplar::Device device;
  if (requested_device_id >= 0) {
    device = manager.GetSpecificDevice(requested_device_id,
                                       tensors->Config().OptionFlags());
  } else {
    device = manager.GetDevice(tensors->Config().NumIpus(),
                               tensors->Config().OptionFlags());
  }

  auto init_end = std::chrono::high_resolution_clock::now();
  std::cout << "Done in "
            << SecondsToTimeString(seconds(init_end - init_start).count())
            << std::endl;

  std::cout << "\n[Executing]\n";
  exe->Prepare(device);
  exe->Deploy();
  auto now = std::chrono::high_resolution_clock::now();
  float elapsed = static_cast<float>(seconds(now - init_end).count());
  tensors->SetOutfeedsFolder(output_folder);

  exe->Run();

  ipu::BinaryWriter parameters_writer(
      absl::StrCat(output_folder, "/parameters.bin"));
  tensors->SaveOutputs(ipu::TensorType::ParameterOut, parameters_writer);

  ipu::BinaryWriter outputs_writer(absl::StrCat(output_folder, "/outputs.bin"));
  tensors->SaveOutputs(ipu::TensorType::OutputData, outputs_writer);

  if (tensors->ContainsCheckpoint()) {
    ipu::BinaryWriter checkpoint_writer(
        absl::StrCat(output_folder, "/checkpoint.bin"));
    tensors->SaveCheckpoint(checkpoint_writer);
  }

  std::cout << "\nExecution complete!\n";
  return 0;
}
