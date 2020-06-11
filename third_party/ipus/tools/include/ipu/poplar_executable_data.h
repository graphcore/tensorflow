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
    throw ipu::Exception(__error_msg.str());                                  \
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

// This is needed to catch STL exceptions and attach some context to them.
#define CATCH_AND_RETHROW_AS_IPU_EXCEPTION                            \
  catch (const ipu::Exception& e) {                                   \
    throw e;                                                          \
  }                                                                   \
  catch (const std::out_of_range& e) {                                \
    ERROR(absl::StrCat("'std::out_of_range' exception: ", e.what())); \
  }                                                                   \
  catch (const std::exception& e) {                                   \
    ERROR(absl::StrCat("'std::exception' exception: ", e.what()));    \
  }

namespace ipu {

/* Custom exception type used throughout the library to raise exceptions
 * with some context attached to them if available.
 */
class Exception : public std::runtime_error {
 public:
  explicit Exception(const std::string& msg);
};

/* Context stack used to attach extra information to exceptions when they're
 * raised. All contexts changes can be printed by enabling the info mode.
 */
class LogContext {
 public:
  // Current context stack as a string
  static const std::string& Context();
  static bool InfoEnabled();
  static void EnableInfo(bool enabled);
  LogContext();
  // Push the context at the top of the context stack.
  explicit LogContext(const std::string& context);
  // Replace the top of the context stack with new_context.
  void UpdateContext(const std::string& new_context);
  // Pop the top of the context stack.
  void Clear();
  // Implicitly pop the top of the context stack if Clear() hasn't been
  // explicitly called.
  ~LogContext();

 private:
  static std::string context_;
  static bool info_enabled_;
  bool cleared_;
  std::string saved_context_;
};

class BinaryWriter;
struct FeedInfo;
class Outfeed;
class StreamReader;
class StreamWriter;

enum DataType {
  F32,
  F16,
  S32,
};

// Vector of bytes used to store binary data.
using ByteVector = std::vector<uint8_t>;

/* Represent the shape of a tensor.
 *
 * metadata_size is the extra space needed to store some extra metadata along
 * the tensor. (0 by default). DataSizeInBytes() is computed as NumElements() *
 * ElementSizeInBytes() + metadata_size
 */
class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(const TensorShape& shape, int64_t metadata_size);
  explicit TensorShape(const Json::Value& shape);
  explicit TensorShape(StreamReader& in);
  TensorShape(const std::vector<int64_t>& shape, DataType type);
  int64_t NumElements() const;
  int64_t ElementSizeInBytes() const;
  int64_t DataSizeInBytes() const;
  int64_t NumDimensions() const;
  int64_t operator[](int64_t idx) const;
  Json::Value ToJson() const;
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

/* Tensor description.
 *
 * Contains the shape of the tensor, its type, but also the Poplar `Handle()` it
 * is linked to and its `Name()` in the original graph.
 */
class TensorInfo {
 public:
  TensorInfo() = default;
  explicit TensorInfo(const Json::Value& info);
  TensorInfo(const Json::Value& info, TensorType type);
  TensorInfo(const std::string& name, const std::string& handle,
             const TensorShape& shape, TensorType type = TensorType::NotSet);
  explicit TensorInfo(StreamReader& in);

  const TensorShape& Shape() const;
  const std::string& Name() const;
  const std::string& Handle() const;
  TensorType Type() const;
  bool TypeAndShapeMatch(const TensorInfo& other) const;
  void SetMetadataSize(int64_t metadata_size);
  void SetShape(const TensorShape& shape);
  void SetName(const std::string& name);
  void SetHandle(const std::string& handle);
  void SetType(TensorType type);
  std::string ToString() const;
  void ToStream(StreamWriter& out) const;
  Json::Value ToJson() const;

 private:
  std::string name_;
  std::string handle_;
  TensorShape shape_;
  TensorType type_{TensorType::NotSet};
};

/* Tensor instance (Description + data)
 */
class Tensor {
 public:
  explicit Tensor(StreamReader& reader);
  explicit Tensor(const TensorInfo& info);
  explicit Tensor(const TensorInfo& info, const void* data);
  const TensorInfo& Info() const;
  void SaveDataToJsonFile(const std::string& filename) const;
  void SaveDataToJsonStream(std::ostream* sout) const;
  void LoadDataFromStream(StreamReader& in);
  void LoadDataFromJsonFile(const std::string& data_filename);
  void* Data();
  std::string ToString() const;
  void ToStream(StreamWriter& out) const;
  void SetMetadataSize(int64_t metadata_size);

 private:
  TensorInfo info_;
  ByteVector data_;
};

bool IsJsonFile(const std::string& filename);
Json::Value LoadJsonFromFile(const std::string& filename);
Json::Value LoadJsonFromString(const std::string& json_content);
// Turn the given string into a valid filename
// (Replace '/' with '_')
std::string SanitizeName(const std::string& filename);

/* Individual stream of an Outfeed
 */
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

/* Outfeed instance (Made of one or more OutfeedStream)
 */
class Outfeed {
 public:
  explicit Outfeed(const FeedInfo& info);
  const std::string& Name() const;
  std::vector<OutfeedStream>& Streams();
  void SetOutputFolder(const std::string& output_folder);
  void IgnoreOutput();

 private:
  const std::string name_;
  std::vector<OutfeedStream> streams_;
};

/* Metadata representing an Outfeed or Infeed
 */
struct FeedInfo {
  FeedInfo() = default;
  explicit FeedInfo(const Json::Value& info);
  Json::Value ToJson() const;
  std::string name;
  std::vector<TensorInfo> streams;
};

/* Opional verification information associated to a tensor or a feed.
 */
class VerificationInfo {
 public:
  VerificationInfo();
  explicit VerificationInfo(const Json::Value& info);
  Json::Value ToJson() const;
  VerificationInfo(int64_t key, int64_t id);
  void SetInfo(int64_t key, int64_t id);
  bool Initialised() const;
  int64_t Key() const;
  int64_t Id() const;

 private:
  int64_t key_;
  int64_t id_;
  bool initialised_;
};

/* Contains the metadata for all the tensors and feeds in the graph.
 */
struct Metadata {
 public:
  Metadata() = default;
  std::string ToJson() const;
  explicit Metadata(const Json::Value& root);
  static const std::string& CheckpointName();
  static const std::string& ClearCheckpointName();
  static const std::string& InputCheckpointIndexHandle();
  static const std::string& InputCheckpointIndexName();
  static const std::string& InputCheckpointHandle();
  static const std::string& OutputCheckpointHandle();
  static const std::string& OutputClearCheckpointHandle();

  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::vector<FeedInfo> infeeds;
  std::vector<FeedInfo> outfeeds;
  int64_t replication_count;
  int64_t num_ipus;
  std::string random_number_seed_handle;
  // PoplarOptions to pass to the poplar::Engine.
  std::map<std::string, std::string> engine_options;
  // PoplarOptions to pass to the poplar::GetDevice.
  std::map<std::string, std::string> device_options;
  // Verified mode only
  std::vector<std::string> feeds_order;
  std::map<std::string, VerificationInfo> verification_info;
};

/* Helper class to be used by the Frameworks to generate a Metadata object.
 */
class MetadataBuilder {
 public:
  MetadataBuilder() = default;
  void AddInput(const TensorInfo& tensor, const VerificationInfo& info = {});
  void AddInputParameter(const TensorInfo& tensor,
                         const VerificationInfo& info = {});
  void AddOutput(const TensorInfo& tensor, const VerificationInfo& info = {});
  void AddOutputParameter(const TensorInfo& tensor,
                          const VerificationInfo& info = {});
  void AddOutputModifiedParameter(const std::string& input_handle,
                                  const TensorInfo& tensor,
                                  const VerificationInfo& info = {});
  void CreateInfeed(const std::string& name);
  void AddInfeedStream(const std::string& infeed_name, const TensorInfo& tensor,
                       const VerificationInfo& info = {});
  void CreateOutfeed(const std::string& name);
  void AddOutfeedStream(const std::string& outfeed_name,
                        const TensorInfo& tensor,
                        const VerificationInfo& info = {});
  void AddDeviceOption(const std::string& key, const std::string& value);
  void AddEngineOption(const std::string& key, const std::string& value);
  void SetConfig(int64_t replication_count, int64_t num_ipus);

  void SetRandomNumberSeedHandle(const std::string& handle);
  void AddCheckpoint(const std::vector<std::string>& feeds_order,
                     const VerificationInfo& checkpointInInfo,
                     const VerificationInfo& checkpointOutInfo);
  Metadata BuildMetadata() const;

 private:
  void AddVerificationInfo(const std::string& handle,
                           const VerificationInfo& info);
  Metadata meta_;
  std::map<std::string, TensorInfo> input_parameters_;
  std::map<std::string, int64_t> infeeds_;
  std::map<std::string, int64_t> outfeeds_;
};

/* Utility class to create binary files.
 */
class StreamWriter {
 public:
  explicit StreamWriter(const std::string& filename);
  void WriteString(const std::string& value);
  void WriteInt64(int64_t value);
  void WriteInt64Array(const std::vector<int64_t>& values);
  void WriteData(const void* data, size_t size);
  void CopyFromStream(StreamReader& in, size_t size);
  void MoveAbsolute(std::streampos position);
  std::streampos CurrentPosition();
  void Close();
  std::ofstream& Stream();

 private:
  std::ofstream fd_;
};

/* Utility class to read binary files.
 */
class StreamReader {
 public:
  explicit StreamReader(const std::string& filename, bool is_versioned = true);
  StreamReader Clone();
  int64_t NumBytesLeft();
  std::string ReadString(int64_t max_len = 0);
  void ReadData(void* dst, int64_t length);
  void MoveRelative(std::streamoff offset);
  void MoveAbsolute(std::streampos position);
  void SetEnd(std::streampos end);
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

/* Utility class to populate the tensors of an infeed stream.
 */
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

/* Utility class to export to file Poplar executables.
 */
class ExecutableWriter {
 public:
  ExecutableWriter(std::shared_ptr<StreamWriter> writer,
                   std::function<void()> on_write_complete);
  StreamWriter& Writer();
  std::ofstream& Stream();
  void WriteComplete();
  ~ExecutableWriter();

 private:
  std::shared_ptr<StreamWriter> writer_;
  std::function<void()> on_write_complete_;
};

/* High level helper to export an entire graph to a binary file.
 */
class BinaryWriter {
 public:
  explicit BinaryWriter(const std::string& filename);
  FeedWriter CreateFeed(const std::string& name, const TensorInfo& info,
                        int64_t num_tensors);
  ExecutableWriter CreateExecutable(const std::string& name,
                                    bool is_verified = false);
  void WriteMetadata(const std::string& name, const Metadata& metadata);
  void WriteTensor(const Tensor& tensor, const std::string override_name = "");
  void Close();
  ~BinaryWriter();

 private:
  std::shared_ptr<StreamWriter> writer_;
};

/* Individual stream of an Infeed
 */
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

/* High level helper to load a serialised graph from one or more binary files
 */
class BinaryReader {
 public:
  void LoadFile(const std::string& filename);
  std::unique_ptr<Metadata> ReadMetadata(
      const std::string& metadata_name = "") const;
  std::unique_ptr<StreamReader> CreateExecutableReader(
      const std::string& executable_name = "") const;
  std::unique_ptr<StreamReader> CreateInfeedStreamReader(
      const std::string& infeed_name) const;
  std::unique_ptr<StreamReader> GetTensorStream(const std::string& name) const;
  std::set<std::string> GetObjectNames(ObjectType type) const;
  std::set<std::string> GetObjectSummaries(ObjectType type) const;
  bool ContainsObject(ObjectType type, const std::string& name) const;

 private:
  struct Object {
    std::string filename;
    std::streampos offset;
    std::streampos end;
  };
  std::unique_ptr<StreamReader> GetObjectReader(ObjectType type,
                                                const std::string& name) const;
  const Object GetObject(ObjectType type, const std::string& name) const;
  std::map<ObjectType, std::map<std::string, Object>> objects_;
};

}  // namespace ipu

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TOOLS_POPLAR_EXECUTABLE_DATA_H_
