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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

#include "include/json/json.h"

#define ERROR(msg)                                                            \
  do {                                                                        \
    std::stringstream __error_msg;                                            \
    __error_msg << "ERROR in " << __FILE__ << ":" << __LINE__ << ": " << msg; \
    if (!ipu::LogContext::Context().empty()) {                                \
      __error_msg << " Context:" << ipu::LogContext::Context();               \
    }                                                                         \
    throw std::runtime_error(__error_msg.str());                              \
  } while (0)

#define PRINT_INFO(msg)                                                    \
  if (ipu::LogContext::InfoEnabled()) {                                    \
    std::cout << "INFO in " << __FILE__ << ":" << __LINE__ << ": " << msg; \
    if (!ipu::LogContext::Context().empty()) {                             \
      std::cout << " Context:" << ipu::LogContext::Context();              \
    }                                                                      \
    std::cout << "\n" << std::flush;                                       \
  }

#define ERROR_ON_MSG(condition, msg) \
  do {                               \
    if (condition) {                 \
      ERROR(msg);                    \
    }                                \
  } while (0)

#define ERROR_ON(condition) ERROR_ON_MSG(condition, #condition)

namespace ipu {

class LogContext {
 public:
  static const std::string& Context();
  static bool InfoEnabled();
  static void EnableInfo(bool enabled);
  explicit LogContext(const std::string& context);
  ~LogContext();

 private:
  static std::string context_;
  static bool info_enabled_;
  const std::string saved_context_;
};

class StreamReader;
class StreamWriter;
class Infeed;
enum DataType {
  F32,
  F16,
  S32,
};

using ByteVector = std::vector<uint8_t>;

class TensorShape {
 public:
  explicit TensorShape(DataType type, const Json::Value& array);
  explicit TensorShape(StreamReader& in);
  TensorShape(const std::vector<int64_t>& shape, DataType type);
  int64_t NumElements() const;
  int64_t ElementSizeInBytes() const;
  int64_t DataSizeInBytes() const;
  int64_t NumDimensions() const;
  int64_t operator[](int64_t idx) const;
  std::string ToString() const;
  DataType Type() const;
  void ToStream(StreamWriter& out) const;
  bool operator==(const TensorShape& other) const;

 private:
  DataType type_;
  std::vector<int64_t> shape_;
};

struct Stream {
  const std::string name;
  bool is_input_stream;
};

class StreamList {
 public:
  explicit StreamList(const std::vector<std::string>& poplar_streams);
  const std::vector<Stream>& Streams() const;
  const Stream& operator[](int idx) const;

 private:
  const std::vector<Stream> streams_;
};

class Executable {
 public:
  explicit Executable(const std::string& executable_filename);
  poplar::Engine& Engine();
  void PrintStreams() const;
  StreamList GetStreams() const;
  void LoadAndRun(const poplar::Device& device);

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

/* Tensor types to connect to the Poplar binary:
 *
 * Parameter: Parameter to an op, e.g weights.
 * InputData: Data to feed to the input of the graph.
 * OutputData: Data produced by the graph, e.g label probabilities.
 * Infeed: Similar to InputData but represents a collection of inputs. Loops are
 * usually used with infeeds to feed data into a graph. Outfeed: Similar to
 * OutputData but if you use an infeed as an input you will usually get a feed
 * as an output.
 */
enum class TensorType { Parameter, InputData, OutputData, Infeed, Outfeed };

class TensorInfo {
 public:
  explicit TensorInfo(const Json::Value& info);
  TensorInfo(const Json::Value& info, TensorType type);
  TensorInfo(const std::string& name, const std::string& handle,
             const TensorShape& shape, TensorType type);
  explicit TensorInfo(StreamReader& in);

  /* Return the filename where the values for this parameter are stored.
   *
   * TensorType must be Parameter.
   */
  std::string ParameterFilename() const;

  const TensorShape& Shape() const;
  const std::string& Name() const;
  const std::string& Handle() const;
  TensorType Type() const;
  std::string ToString() const;
  void ToStream(StreamWriter& out) const;
  bool TypeAndShapeMatch(const TensorInfo& other) const;

 private:
  std::string name_;
  std::string handle_;
  TensorShape shape_;
  TensorType type_;
};

class Tensor {
 public:
  explicit Tensor(const TensorInfo& info);
  const TensorInfo& Info() const;
  void LoadDataFromJson(const std::string& data_filename);
  void SaveDataToJsonFile(const std::string& filename) const;
  void SaveDataToJsonStream(std::ostream* sout) const;
  void* Data();
  void* DataEnd();
  std::string ToString() const;

 private:
  TensorInfo info_;
  ByteVector data_;
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

class JsonParser {
 public:
  explicit JsonParser(const std::string& filename);
  const Json::Value& Root() const;

 private:
  Json::Value root_;
};

class TensorManager {
 public:
  explicit TensorManager(const JsonParser& metadata);
  const std::vector<Tensor>& Inputs() const;
  const std::vector<Tensor>& Outputs() const;
  const std::vector<Infeed>& Infeeds() const;
  std::vector<Infeed>& MutableInfeeds();
  const IpuConfig& Config() const;
  void AllocateTensors();
  std::list<Tensor*> InputDataTensors();
  void LoadParameters(const std::string& path);
  void ConnectStreams(Executable& executable);

 private:
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<Infeed> infeeds_;
  IpuConfig config_;
};

class SeedManager {
 public:
  explicit SeedManager(const IpuConfig& config);
  void ConnectStreams(Executable& executable);

 private:
  std::vector<uint64_t> seeds_;
};

class InfeedStream {
 public:
  explicit InfeedStream(const TensorInfo& info);
  const TensorInfo& Info() const;
  void LoadDataFromBin(const std::string& filename);
  const int8_t* TensorData(int64_t tensor_idx) const;
  int64_t NumTensors() const;

 private:
  TensorInfo info_;
  std::vector<int8_t> data_;
  int64_t num_tensors_;
};

class Infeed {
 public:
  explicit Infeed(const Json::Value& infeed);
  void LoadDataFromBin(const std::string& filename);
  const std::string& Name() const;
  const std::vector<InfeedStream>& Streams() const;
  static std::string StreamFilename(const std::string& filename,
                                    int64_t stream_idx);

 private:
  const std::string name_;
  std::vector<InfeedStream> streams_;
};

class StreamWriter {
 public:
  explicit StreamWriter(const std::string& filename);
  void WriteString(const std::string& value);
  void WriteInt64(int64_t value);
  void WriteInt64Array(const std::vector<int64_t>& values);
  void WriteData(const void* data, size_t size);
  void Close();

 private:
  std::ofstream fd_;
};

class StreamReader {
 public:
  explicit StreamReader(const std::string& filename);
  std::string ReadString();
  void ReadData(void* dst, int64_t length);
  int64_t ReadInt64();
  std::vector<int64_t> ReadInt64Array();

 private:
  std::ifstream fd_;
};

}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_RUNNER_H_
