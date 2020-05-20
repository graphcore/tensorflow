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

#include "tensorflow/compiler/plugin/poplar/tools/poplar_executable_data.h"

namespace ipu {

class BinaryLoader;

class IExecutable {
 public:
  IExecutable(StreamReader& stream, int64_t length);
  poplar::Engine& Engine();
  std::string StreamsList(bool summmary = false) const;
  virtual ~IExecutable() = default;

 protected:
  std::unique_ptr<poplar::Engine> engine_;
};

class VerifiedExecutable : public IExecutable {
 public:
  VerifiedExecutable(StreamReader& stream, int64_t length, bool is_verified);
  void Prepare(poplar::Device& device);
  void Deploy();
  void Run();
  bool IsVerified() const;

 private:
  bool is_verified_;
};

class Executable : public IExecutable {
 public:
  explicit Executable(StreamReader& stream, int64_t length = 0);
  void Load(const poplar::Device& device);
  void Run();
  void DeviceToHostCopy();
};

class DeviceManager {
 public:
  DeviceManager();
  poplar::Device GetDevice(int64_t num_ipus, const poplar::OptionFlags& opts);
  poplar::Device GetSpecificDevice(int64_t device_id,
                                   const poplar::OptionFlags& opts);

 private:
  poplar::DeviceManager manager_;
};

class IpuConfig {
 public:
  explicit IpuConfig(const Metadata& meta);
  int64_t NumIpus() const;
  int64_t ReplicationCount() const;
  poplar::OptionFlags OptionFlags() const;

 private:
  int64_t replication_count_;
  int64_t num_ipus_;
  poplar::OptionFlags option_flags_;
};

class Infeed {
 public:
  explicit Infeed(const FeedInfo& infeed);
  void InitializeDataSources(const BinaryLoader& loader);
  const std::string& Name() const;
  std::vector<InfeedStream>& MutableStreams();
  const std::vector<InfeedStream>& Streams() const;
  static std::string StreamFilename(const std::string& filename,
                                    int64_t stream_idx);

 private:
  const std::string name_;
  std::vector<InfeedStream> streams_;
};

class TensorManager {
 public:
  explicit TensorManager(const Metadata& metadata);
  const std::vector<Tensor>& Inputs() const;
  const std::vector<Tensor>& Outputs() const;
  const std::vector<Infeed>& Infeeds() const;
  void CreateCheckpointMetadataJson(const std::string& filename) const;
  void LoadCheckpointMetadataFromJson(const std::string& filename);
  void SetOutfeedsFolder(const std::string& output_folder);
  void IgnoreOutfeeds();
  std::vector<Infeed>& MutableInfeeds();
  const IpuConfig& Config() const;
  void AllocateTensors();
  std::list<Tensor*> InputDataTensors();
  void AssertAllTensorsProvided(const BinaryLoader& loader);
  void LoadInputsAndParameters(const BinaryLoader& loader);
  void LoadVerifiedCheckpoint(const BinaryLoader& loader,
                              int64_t checkpoint_index);
  void SaveCheckpoint(BinaryWriter& writer);
  void LoadInfeeds(const BinaryLoader& loader);
  void SaveOutputs(TensorType type, BinaryWriter& writer,
                   bool allow_duplicates = false) const;
  void SaveOutputsToJson(TensorType type, const std::string& folder) const;
  void ConnectStreams(IExecutable& executable);
  bool ContainsCheckpoint() const;

 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<Infeed> infeeds_;
  std::vector<Outfeed> outfeeds_;
  std::vector<std::string> feeds_order_;
  IpuConfig config_;
};

class SeedManager {
 public:
  explicit SeedManager(const IpuConfig& config);
  void ConnectStreams(IExecutable& executable);

 private:
  std::vector<uint64_t> seeds_;
};

class BinaryLoader : public BinaryReader {
 public:
  std::unique_ptr<TensorManager> CreateTensorManager(
      const std::string metadata_name = "") const;
  std::unique_ptr<Executable> CreateExecutable(
      const std::string executable_name = "") const;
  std::unique_ptr<VerifiedExecutable> CreateVerifiedExecutable(
      const std::string executable_name = "") const;
};
}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_
