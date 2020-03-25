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

struct BinaryFiles {
  std::vector<std::string> filenames;
};

struct CkptFile {
  std::string filename;
};

struct OutputFolder {
  std::string folder;
};

std::string AbslUnparseFlag(BinaryFiles f) {
  return absl::UnparseFlag(f.filenames);
}

std::string AbslUnparseFlag(CkptFile f) {
  return absl::UnparseFlag(f.filename);
}

std::string AbslUnparseFlag(OutputFolder f) {
  return absl::UnparseFlag(f.folder);
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

bool AbslParseFlag(absl::string_view text, CkptFile* f, std::string* error) {
  if (!absl::ParseFlag(text, &f->filename, error)) {
    return false;
  }
  if (IsDir(f->filename)) {
    ERROR_ON(!error->empty());
    f->filename = absl::StrCat(f->filename, "/ckpt.json");
  }

  if (!FileExists(f->filename)) {
    *error = absl::StrCat("Could not open checkpoint file '", f->filename, "'");
    return false;
  }
  return true;
}

bool AbslParseFlag(absl::string_view text, OutputFolder* f,
                   std::string* error) {
  if (!absl::ParseFlag(text, &f->folder, error)) {
    return false;
  }
  if (!IsDir(f->folder)) {
    *error = absl::StrCat("'", f->folder, "' is not a valid output directory");
    return false;
  }
  return true;
}

ABSL_FLAG(BinaryFiles, binaries, BinaryFiles(),
          "List of binary files containing metadata, binaries, weights,"
          " inputs, feeds, etc. Note if this flag is set then the flags "
          "model_metadata, model_executable, weights_path, input_data, "
          "infeed_data are ignored.");
ABSL_FLAG(int, iterations, 1, "Number of times to run the executable");
ABSL_FLAG(int, ckpt_frequency, 1, "Frequency at which to create checkpoints");
ABSL_FLAG(bool, print_output, false,
          "Print the content of the output buffers to stdout");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(CkptFile, ckpt, CkptFile(),
          "Load the checkpoint config from the given file");
ABSL_FLAG(bool, strict, false,
          "Enable strict mode: all the input data files must be provided by "
          "--input_data.");
ABSL_FLAG(OutputFolder, output_folder, OutputFolder(),
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

  const BinaryFiles binaries = absl::GetFlag(FLAGS_binaries);
  const bool print_output = absl::GetFlag(FLAGS_print_output);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const std::string ckpt_file = absl::GetFlag(FLAGS_ckpt).filename;
  const bool strict = absl::GetFlag(FLAGS_strict);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  const int ckpt_frequency = absl::GetFlag(FLAGS_ckpt_frequency);
  const std::string output_folder = absl::GetFlag(FLAGS_output_folder).folder;

  ipu::LogContext::EnableInfo(verbose);

  auto init_start = std::chrono::high_resolution_clock::now();
  ERROR_ON_MSG(!output_folder.empty() && !CreateDirIfNeeded(output_folder),
               "Failed to create output folder '" << output_folder << "'");

  ERROR_ON_MSG(binaries.filenames.empty(),
               "--binaries needs to point at "
               "one or more folders or bin / ipu_bin files");

  ipu::BinaryLoader loader;
  for (auto file : binaries.filenames) {
    loader.LoadFile(file);
  }

  std::unique_ptr<ipu::TensorManager> tensors = loader.CreateTensorManager();
  std::unique_ptr<ipu::Executable> exe = loader.CreateExecutable();

  if (strict) {
    tensors->AssertAllTensorsProvided(loader);
  }
  tensors->LoadInputsAndParameters(loader);
  tensors->LoadInfeeds(loader);

  if (!ckpt_file.empty()) {
    tensors->LoadCheckpointMetadataFromJson(ckpt_file);
  }

  PRINT_INFO("List of streams:\n" << exe->StreamsList());
  tensors->ConnectStreams(*exe);

  ipu::SeedManager seeds{tensors->Config()};
  seeds.ConnectStreams(*exe);

  std::cout << "\n[Initialising IPU]\n";
  ipu::DeviceManager manager;
  poplar::Device device = manager.GetDevice(tensors->Config().NumIpus(),
                                            tensors->Config().OptionFlags());
  auto init_end = std::chrono::high_resolution_clock::now();
  std::cout << "Done in "
            << SecondsToTimeString(seconds(init_end - init_start).count())
            << std::endl;
  std::cout << "\n[Executing]\n";
  exe->Load(device);
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
      tensors->CreateCheckpointMetadataJson(
          absl::StrCat(iteration_folder, "/ckpt.json"));
      tensors->SetOutfeedsFolder(iteration_folder);
    } else {
      tensors->IgnoreOutfeeds();
    }

    exe->Run();

    if (print_output || create_ckpt) {
      exe->DeviceToHostCopy();
      if (print_output) {
        std::cout << "Outputs:\n";
        for (auto& output : tensors->Outputs()) {
          std::cout << output.ToString() << std::endl;
        }
      }
      if (create_ckpt) {
        ipu::BinaryWriter parameters_writer(
            absl::StrCat(iteration_folder, "/parameters.bin"));
        tensors->SaveOutputs(ipu::TensorType::ParameterOut, parameters_writer);
        tensors->SaveOutputsToJson(ipu::TensorType::OutputData,
                                   iteration_folder);
      }
    }
  }
  std::cout << "\nExecution complete!\n";
  return 0;
}
