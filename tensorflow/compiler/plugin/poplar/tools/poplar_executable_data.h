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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_DATA_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_DATA_H_

#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace poplar {
class Executable;
}  // namespace poplar

namespace ipu {

class LogContext {
 public:
  static const std::string& Context();
  static bool InfoEnabled();
  static void EnableInfo(bool enabled);
  explicit LogContext(const std::string& context);
  void UpdateContext(const std::string& new_context);
  ~LogContext();

 private:
  static std::string context_;
  static bool info_enabled_;
  const std::string saved_context_;
};

class BinaryWriter;
class Outfeed;
class StreamReader;
class StreamWriter;

enum DataType {
  F32,
  F16,
  S32,
};

using ByteVector = std::vector<uint8_t>;

class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(const TensorShape& shape, int64_t metadata_size);
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
  bool HasMetadata() const;
  int64_t MetadataSize() const;
  void SetMetadataSize(int64_t metadata_size);

 private:
  DataType type_;
  std::vector<int64_t> shape_;
  int64_t metadata_size_;
};

/* Tensor types to connect to the Poplar binary:
 *
 * NotSet: Used to detect uninitialized Tensors
 * Parameter: Parameter to an op, e.g weights.
 * ParameterOut: Parameters produced by the graph. e.g updated weights.
 * InputData: Data to feed to the input of the graph.
 * OutputData: Data produced by the graph, e.g label probabilities.
 * Infeed: Similar to InputData but represents a collection of inputs. Loops are
 * usually used with infeeds to feed data into a graph. Outfeed: Similar to
 * OutputData but if you use an infeed as an input you will usually get a feed
 * as an output.
 */
enum class TensorType {
  NotSet,
  Parameter,
  ParameterOut,
  InputData,
  OutputData,
  Infeed,
  Outfeed
};

std::string TensorTypeToString(TensorType type);

class TensorInfo {
 public:
  TensorInfo() = default;
  explicit TensorInfo(const Json::Value& info);
  TensorInfo(const Json::Value& info, TensorType type);
  TensorInfo(const std::string& name, const std::string& handle,
             const TensorShape& shape, TensorType type);
  explicit TensorInfo(StreamReader& in);

  /* Return the filename where the values for this Tensors are stored.
   */
  std::string Filename() const;

  const TensorShape& Shape() const;
  const std::string& Name() const;
  const std::string& Handle() const;
  TensorType Type() const;
  std::string ToString() const;
  void ToStream(StreamWriter& out) const;
  bool TypeAndShapeMatch(const TensorInfo& other) const;
  void SetMetadataSize(int64_t metadata_size);

 private:
  std::string name_;
  std::string handle_;
  TensorShape shape_;
  TensorType type_{TensorType::NotSet};
};

class Tensor {
 public:
  explicit Tensor(StreamReader& reader);
  explicit Tensor(const TensorInfo& info);
  explicit Tensor(const TensorInfo& info, const void* data);
  const TensorInfo& Info() const;
  void SaveDataToJsonFile(const std::string& filename) const;
  void SaveDataToJsonStream(std::ostream* sout) const;
  void LoadDataFromStream(StreamReader& in);
  void LoadDataFromJson(const std::string& data_filename);
  void* Data();
  std::string ToString() const;
  void ToStream(StreamWriter& out) const;
  void SetMetadataSize(int64_t metadata_size);

 private:
  TensorInfo info_;
  ByteVector data_;
};

Json::Value LoadJsonFromFile(const std::string& filename);
Json::Value LoadJsonFromString(const std::string& json_content);

class OutfeedStream {
 public:
  explicit OutfeedStream(const TensorInfo& info);
  const TensorInfo& Info() const;
  void WriteTensor(void* src, int64_t replication_count);
  void SetOutputFolder(const std::string& output_folder);
  void IgnoreOutput();
  ~OutfeedStream();

 private:
  void UpdateNumTensorsAndClose();
  TensorInfo info_;
  std::shared_ptr<StreamWriter> writer_;
  std::streampos num_tensors_pos_;
  std::streampos data_size_pos_;
};

class Outfeed {
 public:
  explicit Outfeed(const Json::Value& Outfeed,
                   std::function<size_t(size_t)> metadata_size_fn = {});
  const std::string& Name() const;
  std::vector<OutfeedStream>& Streams();
  void SetOutputFolder(const std::string& output_folder);
  void IgnoreOutput();

 private:
  const std::string name_;
  std::vector<OutfeedStream> streams_;
};

class StreamWriter {
 public:
  explicit StreamWriter(const std::string& filename);
  void WriteString(const std::string& value);
  void WriteInt64(int64_t value);
  void WriteInt64Array(const std::vector<int64_t>& values);
  void WriteData(const void* data, size_t size);
  void MoveAbsolute(std::streampos position);
  std::streampos CurrentPosition();
  void Close();
  std::fstream& Stream();

 private:
  std::fstream fd_;
};

class StreamReader {
 public:
  explicit StreamReader(const std::string& filename, bool is_versioned = true);
  StreamReader Clone();
  int64_t NumBytesLeft();
  std::string ReadString(int64_t max_len = 0);
  void ReadData(void* dst, int64_t length);
  void MoveRelative(std::streamoff offset);
  void MoveAbsolute(std::streampos position);
  std::streampos CurrentPosition();
  int64_t ReadInt64();
  std::vector<int64_t> ReadInt64Array();
  std::ifstream& Stream();
  const std::string& Filename() const;

 private:
  const std::string filename_;
  std::ifstream fd_;
  std::streampos end_;
};

enum class ObjectType { Feed, Tensor, PoplarExecutable, PoplarMetadata };

class FeedWriter {
 public:
  FeedWriter(std::shared_ptr<StreamWriter> writer, int64_t tensor_size,
             int64_t num_tensors);
  void AppendTensor(const void* data);

 private:
  std::shared_ptr<StreamWriter> writer_;
  int64_t tensor_size_;
  std::streampos end_pos_;
  std::streampos current_pos_;
};

class ExecutableWriter {
 public:
  ExecutableWriter(std::shared_ptr<StreamWriter> writer,
                   std::function<void()> on_write_complete);
  std::fstream& Stream();
  void WriteComplete();
  ~ExecutableWriter();

 private:
  std::shared_ptr<StreamWriter> writer_;
  std::function<void()> on_write_complete_;
};

class BinaryWriter {
 public:
  explicit BinaryWriter(const std::string& filename);
  FeedWriter CreateFeed(const std::string& name, const TensorInfo& info,
                        int64_t num_elements);
  ExecutableWriter CreateExecutable(const std::string& name);
  void WriteMetadata(const std::string& name, const std::string& json_metadata);
  void WriteTensor(const Tensor& tensor, const std::string override_name = "");
  void Close();
  ~BinaryWriter();

 private:
  std::shared_ptr<StreamWriter> writer_;
};

}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_DATA_H_
