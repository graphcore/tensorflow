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
ABSL_FLAG(int, iterations, 1, "Number of times to run the executable");
ABSL_FLAG(int, ckpt_frequency, 1, "Frequency at which to create checkpoints");
ABSL_FLAG(int, device, -1, "Device to use (-1 for any)");
ABSL_FLAG(bool, print_output, false,
          "Print the content of the output buffers to stdout");
ABSL_FLAG(bool, verbose, false, "Enable verbose mode");
ABSL_FLAG(ipu::CheckpointFile, ckpt, ipu::CheckpointFile(),
          "Load the checkpoint config from the given file");
ABSL_FLAG(bool, strict, false,
          "Enable strict mode: all the input data files must be provided by "
          "--input_data.");
ABSL_FLAG(std::string, output_folder, "",
          "Where to save the content of the output tensors");

bool HelpFilter(absl::string_view filename) {
  return filename.find(__FILE__) != absl::string_view::npos;
}

int main(int argc, char** argv) {
  try {
    using seconds = std::chrono::duration<float>;
    // Setting a custom filter is required for the help to be displayed when
    // --help is passed.
    absl::FlagsUsageConfig config;
    config.contains_help_flags = &HelpFilter;
    absl::SetFlagsUsageConfig(config);

    absl::ParseCommandLine(argc, argv);

    const ipu::BinaryFiles binaries = absl::GetFlag(FLAGS_binaries);
    const bool print_output = absl::GetFlag(FLAGS_print_output);
    const bool verbose = absl::GetFlag(FLAGS_verbose);
    const std::string ckpt_file = absl::GetFlag(FLAGS_ckpt).filename;
    const bool strict = absl::GetFlag(FLAGS_strict);
    const int iterations = absl::GetFlag(FLAGS_iterations);
    const int ckpt_frequency = absl::GetFlag(FLAGS_ckpt_frequency);
    const int requested_device_id = absl::GetFlag(FLAGS_device);
    const std::string output_folder = absl::GetFlag(FLAGS_output_folder);

    ipu::LogContext::EnableInfo(verbose);

    auto init_start = std::chrono::high_resolution_clock::now();
    ERROR_ON_MSG(
        !output_folder.empty() && !ipu::CreateDirIfNeeded(output_folder),
        "Failed to create output folder '" << output_folder << "'");

    ERROR_ON_MSG(binaries.filenames.empty(),
                 "--binaries needs to point at "
                 "one or more folders or bin / ipu_bin files");

    ipu::BinaryLoader loader;
    for (auto file : binaries.filenames) {
      loader.LoadFile(file);
    }

    std::unique_ptr<ipu::Metadata> metadata = loader.ReadMetadata();
    std::unique_ptr<ipu::TensorManager> tensors =
        absl::make_unique<ipu::TensorManager>(*metadata);
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

    std::cout << "\n[Initialising IPU]\n";
    ipu::DeviceManager manager;
    poplar::Device device;
    if (requested_device_id >= 0) {
      device = manager.GetSpecificDevice(requested_device_id, *metadata);
    } else {
      device = manager.GetDevice(tensors->NumIpus(), *metadata);
    }
    auto init_end = std::chrono::high_resolution_clock::now();
    std::cout << "Done in "
              << ipu::SecondsToTimeString(
                     seconds(init_end - init_start).count())
              << std::endl;
    std::cout << "\n[Executing]\n";
    exe->Load(device);
    for (int iteration = 0; iteration < iterations; iteration++) {
      auto now = std::chrono::high_resolution_clock::now();
      float elapsed = static_cast<float>(seconds(now - init_end).count());
      float remaining =
          iteration > 0 ? ((elapsed * iterations) / iteration) - elapsed : 0.0;
      std::cout << "Iteration " << iteration << "/" << iterations - 1
                << " Elapsed: " << ipu::SecondsToTimeString(elapsed)
                << ", Estimated remaining: "
                << ipu::SecondsToTimeString(remaining) << std::endl;
      std::string iteration_folder = output_folder;
      bool create_ckpt =
          !output_folder.empty() &&
          (iteration == (iterations - 1) || (iteration % ckpt_frequency == 0));
      if (create_ckpt && iterations > 1) {
        iteration_folder = absl::StrCat(iteration_folder, "/", iteration);
        ERROR_ON_MSG(!ipu::CreateDirIfNeeded(iteration_folder),
                     "Failed to create output folder '" << iteration_folder);
      }
      if (create_ckpt) {
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
          tensors->CreateCheckpointMetadataJson(
              absl::StrCat(iteration_folder, "/ckpt.json"));
          ipu::BinaryWriter parameters_writer(
              absl::StrCat(iteration_folder, "/parameters.bin"));
          tensors->SaveOutputs(ipu::TensorType::ParameterOut,
                               parameters_writer);
          tensors->SaveOutputsToJson(ipu::TensorType::OutputData,
                                     iteration_folder);
        }
      }
    }
    std::cout << "\nExecution complete!\n";
    return 0;
  }
  CATCH_AND_RETHROW_AS_IPU_EXCEPTION
}
