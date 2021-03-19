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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

#include "include/json/json.h"

#include "ipu/poplar_executable_data.h"

namespace ipu {

class BinaryLoader;

/* Base class for all executables
 */
class IExecutable {
 public:
  IExecutable(StreamReader& stream, const Metadata& meta, int64_t length);
  poplar::Engine& Engine();
  std::string StreamsList(bool summary = false) const;
  virtual ~IExecutable() = default;

 protected:
  std::unique_ptr<poplar::Engine> engine_;
};

/* Class to use for executables which use verified data transfers.
 * is_verified should be set to true if the executable itself is verified too.
 *
 * This type of executables is expected to only contain one single program.
 */
class VerifiedExecutable : public IExecutable {
 public:
  VerifiedExecutable(StreamReader& stream, const Metadata& meta, int64_t length,
                     bool is_verified);
  void Prepare(poplar::Device& device);
  void Deploy();
  void Run();
  bool IsVerified() const;

 private:
  bool is_verified_;
};

/* Class to use for regular Poplar executables.
 *
 * This type of executable is expected to contain 3 distinct programs:
 * - HOST_TO_DEVICE
 * - MAIN_SEQUENCE
 * - DEVICE_TO_HOST
 */
class Executable : public IExecutable {
 public:
  explicit Executable(StreamReader& stream, const Metadata& meta,
                      int64_t length = 0);
  void Load(const poplar::Device& device);
  void Run();
  void DeviceToHostCopy();
};

/* Helper class to select which device to run on.
 */
class DeviceManager {
 public:
  DeviceManager();
  poplar::Device GetDevice(int64_t num_ipus, const Metadata& meta);
  poplar::Device GetSpecificDevice(int64_t device_id, const Metadata& meta);

 private:
  poplar::DeviceManager manager_;
};

/* Infeed instance (Made of one or more InfeedStream)
 */
class Infeed {
 public:
  explicit Infeed(const FeedInfo& infeed);
  void InitializeDataSources(const BinaryLoader& loader);
  const std::string& Name() const;
  std::vector<InfeedStream>& MutableStreams();
  const std::vector<InfeedStream>& Streams() const;

 private:
  const std::string name_;
  std::vector<InfeedStream> streams_;
};

/* Owns and keep track of all the tensors and feeds for a given graph.
 */
class TensorManager {
 public:
  explicit TensorManager(const Metadata& meta);
  const std::vector<Tensor>& Inputs() const;
  const std::vector<Tensor>& Outputs() const;
  const std::vector<Infeed>& Infeeds() const;
  // Create a non-verified checkpoint: save the position of all the infeeds so
  // that they can be fast forwarded to the same position at the beginning of
  // the next iteration.
  void CreateCheckpointMetadataJson(const std::string& filename) const;
  // Load a non-verified checkpoint.
  void LoadCheckpointMetadataFromJson(const std::string& filename);
  // Sets the folder where the outfeeds will save the tensors produced by the
  // graph.
  void SetOutfeedsFolder(const std::string& output_folder);
  // Don't save the outfeeds produced by the graph: just discard them.
  void IgnoreOutfeeds();
  std::vector<Infeed>& MutableInfeeds();
  // Allocate all the tensors.
  void AllocateTensors();
  std::list<Tensor*> InputDataTensors();
  // Ensure the loader contains data for all the tensors stored in thie manager.
  void AssertAllTensorsProvided(const BinaryLoader& loader);
  // Load data for all the inputs and parameters from the provided loader.
  void LoadInputsAndParameters(const BinaryLoader& loader);
  // Load a verified checkpoint from the loader with the given index.
  void LoadVerifiedCheckpoint(const BinaryLoader& loader,
                              int64_t checkpoint_index);
  // Create a verified checkpoint
  void SaveVerifiedCheckpoint(BinaryWriter& writer);
  // Connect the infeeds from the provided loader.
  void LoadInfeeds(const BinaryLoader& loader);
  // Export all the outputs using the provided binary writer.
  void SaveOutputs(TensorType type, BinaryWriter& writer,
                   bool allow_duplicates = false) const;
  // Export all the outputs to individual Json files in the specified folder
  void SaveOutputsToJson(TensorType type, const std::string& folder) const;
  // Connect all the tensors and feeds to the provided executable.
  void ConnectStreams(IExecutable& executable);
  // Does this manager contains a check point ? i.e does it contain any infeeds
  // whose position needs saving.
  bool ContainsCheckpoint() const;
  int64_t NumIpus() const;

 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<Infeed> infeeds_;
  std::vector<Outfeed> outfeeds_;
  std::vector<std::string> feeds_order_;
  std::vector<uint64_t> seeds_;
  int64_t num_ipus_;
  int64_t replication_count_;
  std::string random_number_seed_handle_;
};

/* TensorManager / Executable / VerifiedExecutable factories from
 * one or more binary files.
 */
class BinaryLoader : public BinaryReader {
 public:
  std::unique_ptr<TensorManager> CreateTensorManager(
      const std::string& metadata_name = "") const;
  std::unique_ptr<Executable> CreateExecutable(
      const std::string& executable_name = "") const;
  std::unique_ptr<VerifiedExecutable> CreateVerifiedExecutable(
      const std::string& executable_name = "") const;
};
}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_
