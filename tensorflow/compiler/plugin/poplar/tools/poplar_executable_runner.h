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

class Executable {
 public:
  explicit Executable(StreamReader& stream, int64_t length = 0);
  poplar::Engine& Engine();
  std::string StreamsList() const;
  void Load(const poplar::Device& device);
  void Run();
  void DeviceToHostCopy();

 private:
  std::unique_ptr<poplar::Engine> engine_;
};

class DeviceManager {
 public:
  DeviceManager();
  poplar::Device GetDevice(int64_t num_ipus, const poplar::OptionFlags& opts);

 private:
  poplar::DeviceManager manager_;
};

class IpuConfig {
 public:
  explicit IpuConfig(const Json::Value& config);
  int64_t NumIpus() const;
  int64_t ReplicationCount() const;
  poplar::OptionFlags OptionFlags() const;

 private:
  int64_t replication_count_;
  int64_t num_ipus_;
  poplar::OptionFlags option_flags_;
};

class InfeedStream {
 public:
  explicit InfeedStream(std::shared_ptr<StreamReader> in);
  explicit InfeedStream(const TensorInfo& info);
  const TensorInfo& Info() const;
  void InitializeDataSource(const std::string& filename);
  void InitializeDataSource(std::shared_ptr<StreamReader> in);
  void LoadTensor(void* dst);
  void ResetToFirstTensor();
  void MoveToNextTensor();
  void JumpToTensor(int64_t tensor_index);
  int64_t NumTensors() const;
  int64_t TensorIndex() const;
  std::string ToString();

 private:
  TensorInfo info_;
  bool current_tensor_loaded_;
  std::streampos first_tensor_pos_;
  int64_t num_tensors_;
  int64_t tensor_idx_;
  std::shared_ptr<StreamReader> reader_;
};

class Infeed {
 public:
  explicit Infeed(const Json::Value& infeed);
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
  explicit TensorManager(const Json::Value& root);
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
  void LoadInputs(const BinaryLoader& loader);
  void LoadInfeeds(const BinaryLoader& loader);
  void SaveOutputs(TensorType type, BinaryWriter& writer,
                   bool allow_duplicates = false) const;
  void SaveOutputsToJson(TensorType type, const std::string& folder) const;
  void ConnectStreams(Executable& executable);

 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<Infeed> infeeds_;
  std::vector<Outfeed> outfeeds_;
  IpuConfig config_;
};

class SeedManager {
 public:
  explicit SeedManager(const IpuConfig& config);
  void ConnectStreams(Executable& executable);

 private:
  std::vector<uint64_t> seeds_;
};

class BinaryLoader {
 public:
  void LoadFile(const std::string& filename);
  std::unique_ptr<TensorManager> CreateTensorManager(
      const std::string metadata_name = "") const;
  std::unique_ptr<Executable> CreateExecutable(
      const std::string executable_name = "") const;
  std::unique_ptr<StreamReader> CreateInfeedStreamReader(
      const std::string infeed_name) const;
  std::unique_ptr<StreamReader> GetTensorStream(const std::string& name) const;
  std::set<std::string> GetObjectNames(ObjectType type) const;
  bool ContainsObject(ObjectType type, const std::string& name) const;

 private:
  struct Object {
    std::string filename;
    std::streampos offset;
    std::streampos end;
  };
  const Object GetObject(ObjectType type, const std::string& name) const;
  std::map<ObjectType, std::map<std::string, Object>> objects_;
};
}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_
