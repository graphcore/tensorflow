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

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "third_party/eigen3/Eigen/Core"

namespace ipu {

namespace {

std::string GetRandomNumberSeedStream() { return "__seed_stream"; }

enum PoplarProgramType {
  HOST_TO_DEVICE,
  MAIN_SEQUENCE,
  DEVICE_TO_HOST,
};

class DataTypeInfo {
  struct TypeInfo {
    DataType type;
    int64_t element_size;
    std::string type_str;
  };

 public:
  static DataType FromString(const std::string& type_str);
  static const std::string& ToString(DataType type);
  static int64_t SizeInBytes(DataType type);
  static std::vector<DataTypeInfo::TypeInfo> CreateInfo();

 private:
  static const std::vector<TypeInfo> info_;
};

const std::vector<DataTypeInfo::TypeInfo> DataTypeInfo::info_ =
    DataTypeInfo::CreateInfo();

/* static */ std::vector<DataTypeInfo::TypeInfo> DataTypeInfo::CreateInfo() {
  std::vector<DataTypeInfo::TypeInfo> m;
#define ADD_MAPPING(type, elt_size) \
  m.emplace_back(DataTypeInfo::TypeInfo{type, elt_size, #type});

  ADD_MAPPING(F16, 2);
  ADD_MAPPING(S32, 4);
  ADD_MAPPING(F32, 4);
#undef ADD_MAPPING
  return m;
}

/* static */ DataType DataTypeInfo::FromString(const std::string& type_str) {
  auto it = absl::c_find_if(info_, [type_str](const TypeInfo& info) {
    return info.type_str == type_str;
  });
  ERROR_ON_MSG(it == info_.end(), "Unknown DataType '" << type_str << "'");
  return it->type;
}

/* static */ const std::string& DataTypeInfo::ToString(DataType type) {
  auto it = absl::c_find_if(
      info_, [type](const TypeInfo& info) { return info.type == type; });
  ERROR_ON_MSG(it == info_.end(), "Unknown DataType '" << type << "'");
  return it->type_str;
}
/* static */ int64_t DataTypeInfo::SizeInBytes(DataType type) {
  auto it = absl::c_find_if(
      info_, [type](const TypeInfo& info) { return info.type == type; });
  ERROR_ON_MSG(it == info_.end(), "Unknown DataType '" << type << "'");
  return it->element_size;
}

class DataIterator {
 public:
  DataIterator(DataType type, uint8_t* start, int64_t num_elements)
      : element_size_(DataTypeInfo::SizeInBytes(type)),
        current_(start),
        buffer_size_(num_elements * element_size_),
        end_(current_ + buffer_size_) {}
  template <typename T>
  T Get() const {
    ERROR_ON(sizeof(T) != element_size_);
    ERROR_ON_MSG(!IsValid(),
                 "Failed to get iterator's value: the iterator is pointing "
                 "past the end of the buffer of size "
                     << buffer_size_
                     << " bytes (Element size: " << element_size_ << ")");
    return *reinterpret_cast<T*>(current_);
  }
  template <typename T>
  void Set(T value) {
    ERROR_ON(sizeof(T) != element_size_);
    ERROR_ON_MSG(!IsValid(),
                 "Failed to set iterator's value: the iterator is pointing "
                 "past the end of the buffer of size "
                     << buffer_size_
                     << " bytes (Element size: " << element_size_ << ")");
    *reinterpret_cast<T*>(current_) = value;
  }
  void Increment() {
    ERROR_ON_MSG(
        !IsValid(),
        "Failed to increment iterator: reached the end of buffer of size "
            << buffer_size_ << " bytes (Element size: " << element_size_
            << ")");
    current_ += element_size_;
  }
  DataIterator& operator++(int) {
    Increment();
    return *this;
  }
  bool IsValid() const { return current_ < end_; }
  void ErrorIfIsValid() const {
    ERROR_ON_MSG(
        IsValid(),
        "Iterator should point at the end of the file but current position is "
            << end_ - current_ << " bytes from the end (Buffer size "
            << buffer_size_ << " Element size " << element_size_ << ")");
  }

 private:
  int64_t element_size_;
  int64_t buffer_size_;
  uint8_t* current_;
  uint8_t* end_;
};

Json::Value IteratorToJsonValue(DataType type, const DataIterator& current) {
  switch (type) {
    case S32: {
      return {current.Get<int32_t>()};
      break;
    }
    case F32: {
      return {current.Get<float>()};
      break;
    }
    case F16: {
      return {static_cast<float>(current.Get<Eigen::half>())};
      break;
    }
    default: { ERROR("DataType " << type << " not supported"); }
  }
}

void SetIteratorToJsonValue(const Json::Value& value, DataType type,
                            DataIterator& current) {
  switch (type) {
    case S32: {
      ERROR_ON(!value.isInt());
      current.Set(static_cast<int32_t>(value.asInt()));
      break;
    }
    case F32: {
      ERROR_ON(!value.isDouble());
      current.Set(value.asFloat());
      break;
    }
    case F16: {
      ERROR_ON(!value.isDouble());
      current.Set(Eigen::half(value.asFloat()));
      break;
    }
    default: { ERROR("DataType " << type << " not supported"); }
  }
}

TensorType ParseTensorType(const std::string& str) {
  if (str == "parameter") {
    return TensorType::Parameter;
  } else if (str == "input_data") {
    return TensorType::InputData;
  } else if (str == "output_data") {
    return TensorType::OutputData;
  } else if (str == "parameter_out") {
    return TensorType::ParameterOut;
  } else if (str == "infeed") {
    return TensorType::Infeed;
  } else if (str == "outfeed") {
    return TensorType::Outfeed;
  }
  ERROR("Unknown TensorType '" << str << "'");
}

std::string TensorTypeToString(TensorType type) {
  switch (type) {
    case TensorType::Parameter: {
      return "parameter";
    }
    case TensorType::ParameterOut: {
      return "output parameter";
    }
    case TensorType::InputData: {
      return "input";
    }
    case TensorType::OutputData: {
      return "output";
    }
    case TensorType::Infeed: {
      return "infeed";
    }
    case TensorType::Outfeed: {
      return "outfeed";
    }
    default: { ERROR("Unknown TensorType"); }
  }
}
poplar::OptionFlags ParseOptionFlags(const Json::Value& options) {
  poplar::OptionFlags opts;
  if (!options.isNull()) {
    for (auto key : options.getMemberNames()) {
      std::string value = options[key].asString();
      opts.set(key, value);
    }
  }
  return opts;
}

/* Recursively parse a JSON array of floats, validating the shape against the
 * expected shape and storing the values in a 1D array. */
class NDArrayParser {
 public:
  void operator()(const Json::Value& array, const TensorShape& expected_shape,
                  ByteVector& out);

 private:
  void ProcessDimension(int64_t dim, const Json::Value& array,
                        const TensorShape& expected_shape, DataIterator& out);
};

void NDArrayParser::operator()(const Json::Value& array,
                               const TensorShape& expected_shape,
                               ByteVector& out) {
  ERROR_ON(out.size() != 0);
  PRINT_INFO("Expected shape " << expected_shape.ToString() << " NumElements "
                               << expected_shape.NumElements()
                               << " SizeInBytes "
                               << expected_shape.DataSizeInBytes());
  out.resize(expected_shape.DataSizeInBytes());
  DataIterator it{expected_shape.Type(), out.data(),
                  expected_shape.NumElements()};
  // Special case if it's a scalar.
  if (expected_shape.NumDimensions() == 0) {
    SetIteratorToJsonValue(array, expected_shape.Type(), it);
    it++;
  } else {
    ProcessDimension(0, array, expected_shape, it);
  }
  it.ErrorIfIsValid();
}

void NDArrayParser::ProcessDimension(int64_t dim, const Json::Value& array,
                                     const TensorShape& expected_shape,
                                     DataIterator& out) {
  ERROR_ON(!array.isArray());
  ERROR_ON(array.size() != expected_shape[dim]);

  // Last dimension: parse the values
  if (dim == expected_shape.NumDimensions() - 1) {
    for (auto& value : array) {
      SetIteratorToJsonValue(value, expected_shape.Type(), out);
      out++;
    }
  } else {
    // Recursively process other dimensions.
    for (auto subarray : array) {
      ProcessDimension(dim + 1, subarray, expected_shape, out);
    }
  }
}

/* Create a ND array from a given tensor shape and an iterator over a 1D array
 * of float values. */
Json::Value DimensionToJson(int dimension, const TensorShape& shape,
                            DataIterator& current) {
  int64_t num_elements = shape[dimension];
  Json::Value values{Json::arrayValue};

  // Last dimension: parse the values
  if (dimension == shape.NumDimensions() - 1) {
    for (int64_t i = 0; i < num_elements; i++) {
      values.append(IteratorToJsonValue(shape.Type(), current));
      current++;
    }
    return values;
  } else {
    // Recursively process other dimensions.
    for (int64_t i = 0; i < num_elements; i++) {
      values.append(DimensionToJson(dimension + 1, shape, current));
    }
    return values;
  }
}

class InfeedCallback : public poplar::StreamCallback {
 public:
  explicit InfeedCallback(InfeedStream& stream) : stream_(stream) {}

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    return poplar::StreamCallback::Result::NotAvailable;
  }

  void fetch(void* dest) noexcept override {
    ERROR_ON_MSG(stream_.NumTensors() == 0,
                 "Infeed dataset does not contain any tensor");

    stream_.LoadTensor(dest);
  }

  void complete() noexcept override { stream_.MoveToNextTensor(); }

 private:
  InfeedStream& stream_;
};

void WriteJsonToStream(const Json::Value& root, std::ostream* sout) {
  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";

  std::unique_ptr<Json::StreamWriter> writer(json_builder.newStreamWriter());
  writer->write(root, sout);
}

}  // namespace

LogContext::LogContext(const std::string& context) : saved_context_(context_) {
  context_ = absl::StrCat(saved_context_, " ", context);
}
LogContext::~LogContext() { context_ = saved_context_; }
/* static */ const std::string& LogContext::Context() { return context_; }

/* static */ bool LogContext::InfoEnabled() { return info_enabled_; }

/* static */ void LogContext::EnableInfo(bool enabled) {
  info_enabled_ = enabled;
}

std::string LogContext::context_ = "";  // NOLINT
bool LogContext::info_enabled_ = false;

int64_t TensorShape::ElementSizeInBytes() const {
  return DataTypeInfo::SizeInBytes(type_);
}

int64_t TensorShape::NumDimensions() const { return shape_.size(); }

int64_t TensorShape::operator[](int64_t idx) const { return shape_.at(idx); }

int64_t TensorShape::NumElements() const {
  return absl::c_accumulate(shape_, 1, std::multiplies<int64_t>());
}

DataType TensorShape::Type() const { return type_; }

TensorShape::TensorShape(DataType type, const Json::Value& array)
    : type_(type) {
  ERROR_ON(!array.isArray());
  shape_.reserve(array.size());
  absl::c_transform(array, std::back_inserter(shape_),
                    [](const Json::Value& value) {
                      ERROR_ON(!value.isInt64());
                      return value.asInt64();
                    });
  ERROR_ON(shape_.size() != array.size());
}

std::string TensorShape::ToString() const {
  return absl::StrCat("[", absl::StrJoin(shape_, ", "),
                      "] type = ", DataTypeInfo::ToString(type_));
}

void TensorShape::ToStream(StreamWriter& out) const {
  out.WriteInt64(type_);
  out.WriteInt64Array(shape_);
}
TensorShape::TensorShape(StreamReader& in)
    : type_(static_cast<DataType>(in.ReadInt64())),
      shape_(in.ReadInt64Array()) {}

TensorShape::TensorShape(const std::vector<int64_t>& shape, DataType type)
    : shape_(shape), type_(type) {}

std::string Tensor::ToString() const {
  std::stringstream ss;
  SaveDataToJsonStream(&ss);
  return absl::StrCat(info_.ToString(), " data = ", ss.str());
}

void Tensor::SaveDataToJsonFile(const std::string& filename) const {
  std::ofstream out(filename);
  SaveDataToJsonStream(&out);
  out.close();
  PRINT_INFO("Saved content of " << info_.Name() << " to " << filename);
}

void Tensor::SaveDataToJsonStream(std::ostream* sout) const {
  // Use a const_cast here to avoid having to create a ConstDataIterator: it's
  // safe because the iterator will only be used for reading.
  DataIterator it(info_.Shape().Type(), const_cast<uint8_t*>(&data_[0]),
                  info_.Shape().NumElements());

  Json::Value root;
  if (info_.Shape().NumDimensions() == 0) {
    root = IteratorToJsonValue(info_.Shape().Type(), it);
  } else {
    root = DimensionToJson(0, info_.Shape(), it);
  }

  WriteJsonToStream(root, sout);
}

Executable::Executable(const std::string& executable_filename) {
  try {
    std::ifstream file(executable_filename, std::ios::binary);
    poplar::Executable poplar_executable =
        poplar::Executable::deserialize(file);
    engine_.reset(new poplar::Engine(std::move(poplar_executable)));
  } catch (const std::exception& e) {
    ERROR("Failed to deserialize " << executable_filename << " : " << e.what());
  }
}

poplar::Engine& Executable::Engine() { return *engine_; }
StreamList::StreamList(const std::vector<std::string>& poplar_streams)
    : streams_([&poplar_streams]() {
        std::vector<Stream> streams;
        absl::c_transform(poplar_streams, std::back_inserter(streams),
                          [](const std::string& stream) {
                            char last_char = stream[stream.size() - 1];
                            const std::string name =
                                stream.substr(0, stream.size() - 1);
                            ERROR_ON(last_char != '+' && last_char != '-');
                            return Stream{name, last_char == '+'};
                          });
        return streams;
      }()) {}
const std::vector<Stream>& StreamList::Streams() const { return streams_; }

const Stream& StreamList::operator[](int idx) const {
  ERROR_ON(idx >= streams_.size());
  return streams_.at(idx);
}

StreamList Executable::GetStreams() const {
  return StreamList{engine_->listStreams()};
}

std::string Executable::StreamsList() const {
  int idx = 0;
  std::stringstream ss;
  for (auto stream : engine_->listStreams()) {
    ss << "[" << idx++ << "] " << stream << std::endl;
  }
  return ss.str();
}

void Executable::Load(const poplar::Device& device) {
  std::cout << "Loading program onto the device\n";
  engine_->load(device);
  PRINT_INFO("Running HOST_TO_DEVICE");
  engine_->run(PoplarProgramType::HOST_TO_DEVICE);
}

void Executable::Run() {
  PRINT_INFO("Running MAIN_SEQUENCE");
  engine_->run(PoplarProgramType::MAIN_SEQUENCE);
}

void Executable::DeviceToHostCopy() {
  PRINT_INFO("Running DEVICE_TO_HOST");
  engine_->run(PoplarProgramType::DEVICE_TO_HOST);
}

DeviceManager::DeviceManager()
    : manager_(poplar::DeviceManager::createDeviceManager()) {
  ERROR_ON_MSG(absl::c_none_of(manager_.getDevices(),
                               [](const poplar::Device& d) {
                                 return d.getTarget().getTargetType() ==
                                        poplar::TargetType::IPU;
                               }),
               "No physical IPU detected on this host");
}

poplar::Device DeviceManager::GetDevice(int64_t num_ipus,
                                        const poplar::OptionFlags& opts) {
  auto device_list =
      manager_.getDevices(poplar::TargetType::IPU, num_ipus, opts);
  ERROR_ON_MSG(
      device_list.size() == 0,
      "Failed to find any IPU device that match the requested config: num_ipus="
          << num_ipus << " OptionFlags="
          << absl::StrJoin(opts, ", ", absl::PairFormatter("=")));

  std::cout << "Found " << device_list.size()
            << " devices matching the requested configuration.\n";
  int attempt = 1;
  for (auto& d : device_list) {
    std::cout << "[" << attempt << "/" << device_list.size()
              << "]Trying to attach...";
    if (d.attach()) {
      std::cout << " OK!\n";
      unsigned mj, mn, pt;
      d.getDriverVersion(mj, mn, pt);
      const auto& ids = d.getDriverIDs();
      std::cout << "Poplar driver: " << mj << "." << mn << "." << pt
                << std::endl;
      std::cout << "Successfully attached to IPU" << (ids.size() > 1 ? "s" : "")
                << ": " << absl::StrJoin(ids, ",") << std::endl;
      return std::move(d);
    }
    std::cout << " Failed.\n";
    attempt++;
  }
  ERROR("Failed to attach to any of the IPU devices");
}

TensorInfo::TensorInfo(const Json::Value& info, TensorType type)
    : TensorInfo(
          info["name"].asString(), info["handle"].asString(),
          TensorShape{DataTypeInfo::FromString(info["data_type"].asString()),
                      info["shape"]},
          type) {
  PRINT_INFO("Found " << ToString());
}

TensorInfo::TensorInfo(const Json::Value& info)
    : TensorInfo(info, ParseTensorType(info["type"].asString())) {}

TensorInfo::TensorInfo(StreamReader& in)
    : name_(in.ReadString()),
      handle_(in.ReadString()),
      shape_(in),
      type_(static_cast<TensorType>(in.ReadInt64())) {}

TensorInfo::TensorInfo(const std::string& name, const std::string& handle,
                       const TensorShape& shape, TensorType type)
    : name_(name), handle_(handle), shape_(shape), type_(type) {}

bool TensorInfo::TypeAndShapeMatch(const TensorInfo& other) const {
  return type_ == other.type_ && shape_ == other.shape_;
}
void TensorInfo::ToStream(StreamWriter& out) const {
  out.WriteString(name_);
  out.WriteString(handle_);
  shape_.ToStream(out);
  out.WriteInt64(static_cast<int64_t>(type_));
}

std::string TensorInfo::ToString() const {
  return absl::StrCat("{ name = '", name_, "' type = '",
                      TensorTypeToString(type_), "' shape = '",
                      shape_.ToString(), "' handle='", handle_, "' }");
}

int64_t TensorShape::DataSizeInBytes() const {
  return NumElements() * ElementSizeInBytes();
}
const TensorShape& TensorInfo::Shape() const { return shape_; }

bool TensorShape::operator==(const TensorShape& other) const {
  return type_ == other.type_ && shape_ == other.shape_;
}

const std::string& TensorInfo::Name() const { return name_; }
const std::string& TensorInfo::Handle() const { return handle_; }
TensorType TensorInfo::Type() const { return type_; }

Tensor::Tensor(const TensorInfo& info) : info_(info) {
  data_.resize(info_.Shape().DataSizeInBytes());
}

std::string TensorInfo::Filename() const {
  std::string data_filename = name_;
  std::replace(data_filename.begin(), data_filename.end(), '/', '_');
  return absl::StrCat(data_filename, ".data");
}

IpuConfig::IpuConfig(const Json::Value& config)
    : replication_count_(config["replication_count"].asInt64()),
      num_ipus_(config["num_ipus"].asInt64()),
      option_flags_(ParseOptionFlags(config["options"])) {}

int64_t IpuConfig::NumIpus() const { return num_ipus_; }
int64_t IpuConfig::ReplicationCount() const { return replication_count_; }

poplar::OptionFlags IpuConfig::OptionFlags() const { return option_flags_; }

JsonParser::JsonParser(const std::string& filename) {
  std::ifstream json_file(filename);
  ERROR_ON_MSG(!json_file.is_open(), "Failed to open file " << filename);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;
  ERROR_ON_MSG(!parseFromStream(builder, json_file, &root_, &errs), errs);
}

const Json::Value& JsonParser::Root() const { return root_; }

TensorManager::TensorManager(const JsonParser& metadata)
    : config_(metadata.Root()["config"]) {
  config_ = IpuConfig(metadata.Root()["config"]);
  absl::c_transform(
      metadata.Root()["inputs"], std::back_inserter(inputs_),
      [](const Json::Value& input) { return Tensor{TensorInfo{input}}; });
  absl::c_transform(
      metadata.Root()["outputs"], std::back_inserter(outputs_),
      [](const Json::Value& output) { return Tensor{TensorInfo{output}}; });
  absl::c_transform(metadata.Root()["infeeds"], std::back_inserter(infeeds_),
                    [](const Json::Value& infeed) { return Infeed{infeed}; });
  absl::c_transform(
      metadata.Root()["outfeeds"], std::back_inserter(outfeeds_),
      [](const Json::Value& outfeed) { return Outfeed{outfeed}; });
}

void TensorManager::LoadCheckpointMetadataFromJson(
    const std::string& filename) {
  LogContext ctx(
      absl::StrCat("Loading checkpoint metadata from JSON file ", filename));
  JsonParser parser(filename);
  const Json::Value& infeed_positions = parser.Root()["infeeds"];
  absl::c_for_each(infeeds_, [&infeed_positions](Infeed& infeed) {
    absl::c_for_each(
        infeed.MutableStreams(), [&infeed_positions](InfeedStream& stream) {
          auto index = infeed_positions[stream.Info().Handle()];
          ERROR_ON_MSG(index.isNull(), "Can't find any information for stream '"
                                           << stream.Info().Handle()
                                           << "'in the checkpoint metadata");
          stream.JumpToTensor(index.asInt64());
        });
  });
}

void TensorManager::CreateCheckpointMetadataJson(
    const std::string& filename) const {
  std::ofstream out(filename);
  Json::Value root;
  Json::Value infeeds;
  absl::c_for_each(infeeds_, [&infeeds](const Infeed& infeed) {
    absl::c_for_each(infeed.Streams(), [&infeeds](const InfeedStream& stream) {
      infeeds[stream.Info().Handle()] = stream.TensorIndex();
    });
  });
  if (!infeeds.empty()) {
    root["infeeds"] = infeeds;
  }
  WriteJsonToStream(root, &out);
  out.close();
}
void TensorManager::SetOutfeedsFolder(const std::string& output_folder) {
  absl::c_for_each(outfeeds_, [&output_folder](Outfeed& outfeed) {
    outfeed.SetOutputFolder(output_folder);
  });
}

void TensorManager::IgnoreOutfeeds() {
  absl::c_for_each(outfeeds_, [](Outfeed& outfeed) { outfeed.IgnoreOutput(); });
}

const TensorInfo& Tensor::Info() const { return info_; }

const std::vector<Tensor>& TensorManager::Inputs() const { return inputs_; }
const std::vector<Tensor>& TensorManager::Outputs() const { return outputs_; }
const std::vector<Infeed>& TensorManager::Infeeds() const { return infeeds_; }
std::vector<Infeed>& TensorManager::MutableInfeeds() { return infeeds_; }

void Tensor::LoadDataFromJson(const std::string& data_filename) {
  LogContext ctx(absl::StrCat("Loading ", TensorTypeToString(info_.Type()), " ",
                              info_.Name(), " from JSON file '", data_filename,
                              "'"));
  try {
    NDArrayParser parser;
    data_.clear();
    JsonParser js(data_filename);
    parser(js.Root(), info_.Shape(), data_);
  } catch (const std::out_of_range& error) {
    ERROR(error.what());
  }
}

void TensorManager::LoadParameters(const std::string& path) {
  for (auto& input : inputs_) {
    if (input.Info().Type() == TensorType::Parameter) {
      LogContext ctx(
          absl::StrCat("Loading parameter '", input.Info().Name(), "'"));
      input.LoadDataFromJson(absl::StrCat(path, "/", input.Info().Filename()));
    }
  }
}

void TensorManager::SaveOutputsToJsonFile(const std::string& path) {
  std::map<std::string, int> duplicates;
  for (auto& output : outputs_) {
    std::string filename = output.Info().Filename();
    auto& occurrences = duplicates[output.Info().Name()];
    if (occurrences > 0) {
      filename = Infeed::StreamFilename(filename, occurrences);
    }
    occurrences++;
    output.SaveDataToJsonFile(absl::StrCat(path, "/", filename));
  }
}

std::list<Tensor*> TensorManager::InputDataTensors() {
  std::list<Tensor*> out;
  for (auto& input : inputs_) {
    if (input.Info().Type() == TensorType::InputData) {
      out.push_back(&input);
    }
  }
  return out;
}

const IpuConfig& TensorManager::Config() const { return config_; }

void* Tensor::Data() {
  ERROR_ON(data_.size() == 0);
  ERROR_ON_MSG(data_.size() != info_.Shape().DataSizeInBytes(),
               "Buffer is of size " << data_.size() << " but the shape "
                                    << info_.Shape().ToString() << " requires "
                                    << info_.Shape().DataSizeInBytes());
  return data_.data();
}
void* Tensor::DataEnd() { return data_.data() + data_.size(); }

void TensorManager::ConnectStreams(Executable& executable) {
  auto& engine = executable.Engine();

  for (auto& input : inputs_) {
    PRINT_INFO("Connecting " << input.Info().Name() << " to handle "
                             << input.Info().Handle());
    engine.connectStream(input.Info().Handle(), input.Data());
  }

  for (auto& output : outputs_) {
    for (int replica_id = 0; replica_id < config_.ReplicationCount();
         replica_id++) {
      auto callback = [&output, replica_id](void* ptr) {
        if (replica_id == 0) {
          std::memcpy(output.Data(), ptr,
                      output.Info().Shape().DataSizeInBytes());
        }
      };
      engine.connectStreamToCallback(output.Info().Handle(), replica_id,
                                     callback);
    }
  }

  for (auto& infeed : infeeds_) {
    for (auto& infeed_stream : infeed.MutableStreams()) {
      for (int replica_id = 0; replica_id < config_.ReplicationCount();
           replica_id++) {
        auto callback = absl::make_unique<InfeedCallback>(infeed_stream);
        engine.connectStreamToCallback(infeed_stream.Info().Handle(),
                                       replica_id, std::move(callback));
      }
    }
  }
  for (auto& outfeed : outfeeds_) {
    for (auto& outfeed_stream : outfeed.Streams()) {
      for (int replica_id = 0; replica_id < config_.ReplicationCount();
           replica_id++) {
        engine.connectStreamToCallback(
            outfeed_stream.Info().Handle(), replica_id,
            [&outfeed_stream, this](void* src) {
              outfeed_stream.WriteTensor(src, this->config_.ReplicationCount());
            });
      }
    }
  }
}

SeedManager::SeedManager(const IpuConfig& config) {
  int replication_count = config.ReplicationCount();
  seeds_.resize(replication_count);
  std::mt19937_64 seed_generator;
  for (auto& seed : seeds_) {
    seed = seed_generator();
  }
}

void SeedManager::ConnectStreams(Executable& executable) {
  for (int replica_id = 0; replica_id < seeds_.size(); replica_id++) {
    auto callback = [this, replica_id](void* ptr) mutable {
      reinterpret_cast<uint64_t*>(ptr)[0] = this->seeds_[replica_id];
    };

    executable.Engine().connectStreamToCallback(GetRandomNumberSeedStream(),
                                                replica_id, callback);
  }
}

void StreamWriter::WriteInt64(int64_t value) {
  WriteData(&value, sizeof(value));
}
void StreamWriter::WriteInt64Array(const std::vector<int64_t>& values) {
  WriteInt64(values.size());
  if (values.size() > 0) {
    WriteData(values.data(), sizeof(int64_t) * values.size());
  }
}
void StreamWriter::WriteData(const void* data, size_t size) {
  ERROR_ON(size == 0);
  fd_.write(reinterpret_cast<const char*>(data), size);
}
void StreamWriter::WriteString(const std::string& value) {
  WriteInt64(value.size());
  if (value.size() > 0) {
    WriteData(value.c_str(), value.size());
  }
}
std::string StreamReader::ReadString() {
  int64_t len = ReadInt64();
  if (len == 0) {
    return "";
  }
  std::string out;
  out.resize(len);
  ReadData(const_cast<char*>(out.c_str()), len);
  return out;
}
StreamWriter::StreamWriter(const std::string& filename)
    : fd_(filename, std::ostream::binary) {
  ERROR_ON_MSG(!fd_.is_open(), "Failed to open file '" << filename << "'");
}
void StreamWriter::Close() { fd_.close(); }

StreamReader::StreamReader(const std::string& filename)
    : fd_(filename, std::istream::binary) {
  ERROR_ON_MSG(!fd_.is_open(), "Failed to open file '" << filename << "'");
  std::streampos begin = fd_.tellg();
  fd_.seekg(0, std::ios::end);
  end_ = fd_.tellg();
  fd_.seekg(0, std::ios::beg);
}

int64_t StreamReader::NumBytesLeft() { return end_ - fd_.tellg(); }

void StreamReader::ReadData(void* dst, int64_t length) {
  ERROR_ON(length <= 0);
  fd_.read(reinterpret_cast<char*>(dst), length);
}

int64_t StreamReader::ReadInt64() {
  int64_t value;
  ReadData(&value, sizeof(value));
  return value;
}

std::vector<int64_t> StreamReader::ReadInt64Array() {
  int64_t len = ReadInt64();
  if (len == 0) {
    return {};
  }
  std::vector<int64_t> out;
  out.resize(len);
  ReadData(out.data(), len * sizeof(int64_t));
  return out;
}

void StreamReader::MoveRelative(std::ios::streamoff offset) {
  fd_.seekg(offset, std::ios::cur);
}

void StreamReader::MoveAbsolute(std::ios::streampos position) {
  fd_.seekg(position);
}
std::ios::streampos StreamReader::CurrentPosition() { return fd_.tellg(); }

InfeedStream::InfeedStream(const TensorInfo& info) : info_(info) {
}

InfeedStream::InfeedStream(const std::string& filename) {
  InitializeDataSource(filename);
}
const TensorInfo& InfeedStream::Info() const { return info_; }

std::string InfeedStream::ToString() {
  // Backup current position
  int64_t tensor_idx = tensor_idx_;
  std::ios::streampos current_pos = reader_->CurrentPosition();

  ResetToFirstTensor();
  Tensor tmp{info_};
  std::list<std::string> tensors;
  for (; TensorIndex() < NumTensors(); MoveToNextTensor()) {
    LoadTensor(tmp.Data());
    tensors.push_back(tmp.ToString());
  }

  reader_->MoveAbsolute(current_pos);
  tensor_idx_ = tensor_idx;
  return absl::StrCat("[", absl::StrJoin(tensors, "\n"), "]");
}
void InfeedStream::InitializeDataSource(const std::string& filename) {
  LogContext ctx{absl::StrCat("from file ", filename)};
  reader_ = std::make_shared<StreamReader>(filename);
  TensorInfo info{*reader_};
  // If there is no metadata then keep these ones.
  if (info_.Type() == TensorType::NotSet) {
    info_ = std::move(info);
  } else {
    ERROR_ON_MSG(
        !info_.TypeAndShapeMatch(info),
        "Type and Shape from metadata JSON file: "
            << info_.ToString()
            << " don't match those from binary file: " << info.ToString());
  }
  num_tensors_ = reader_->ReadInt64();
  // Num tensors not provided: deduce it from the file size.
  if (num_tensors_ == 0) {
    num_tensors_ = reader_->NumBytesLeft() / info_.Shape().DataSizeInBytes();
  }
  ERROR_ON(num_tensors_ == 0);
  ERROR_ON(reader_->NumBytesLeft() !=
           num_tensors_ * info_.Shape().DataSizeInBytes());
  first_tensor_pos_ = reader_->CurrentPosition();
  current_tensor_loaded_ = false;
  tensor_idx_ = 0;
}

void InfeedStream::LoadTensor(void* dst) {
  if (tensor_idx_ >= num_tensors_) {
    ResetToFirstTensor();
  }
  if (current_tensor_loaded_) {
    // Current tensor has already been read so move back the file descriptor
    // position.
    reader_->MoveRelative(-info_.Shape().DataSizeInBytes());
  }
  reader_->ReadData(dst, info_.Shape().DataSizeInBytes());
  current_tensor_loaded_ = true;
}

void InfeedStream::JumpToTensor(int64_t tensor_index) {
  ERROR_ON(tensor_index >= num_tensors_);
  reader_->MoveAbsolute(first_tensor_pos_ +
                        info_.Shape().DataSizeInBytes() * tensor_index);
  tensor_idx_ = tensor_index;
  current_tensor_loaded_ = false;
}

void InfeedStream::MoveToNextTensor() {
  if (!current_tensor_loaded_) {
    reader_->MoveRelative(info_.Shape().DataSizeInBytes());
  }
  current_tensor_loaded_ = false;
  tensor_idx_++;
}

void InfeedStream::ResetToFirstTensor() {
  PRINT_INFO("Infeed " << info_.Name() << " reached the end of its "
                       << num_tensors_ << " elements: resetting to the start");
  current_tensor_loaded_ = false;
  tensor_idx_ = 0;
  reader_->MoveAbsolute(first_tensor_pos_);
}

int64_t InfeedStream::NumTensors() const { return num_tensors_; }
int64_t InfeedStream::TensorIndex() const { return tensor_idx_; }

OutfeedStream::OutfeedStream(const TensorInfo& info) : info_(info), writer_() {}

void OutfeedStream::SetOutputFolder(const std::string& output_folder) {
  const std::string filename =
      absl::StrCat(output_folder, "/", info_.Name(), ".bin");
  if (writer_) {
    writer_->Close();
  }
  writer_ = std::make_shared<StreamWriter>(filename);
  info_.ToStream(*writer_);
  // Set num_tensors to 0: we don't know how many elements we're going to write.
  writer_->WriteInt64(0);
}

void OutfeedStream::IgnoreOutput() {
  if (writer_) {
    writer_->Close();
  }
  writer_ = nullptr;
}

const TensorInfo& OutfeedStream::Info() const { return info_; }

void OutfeedStream::WriteTensor(void* src, int64_t replication_count) {
  if (writer_) {
    writer_->WriteData(src,
                       info_.Shape().DataSizeInBytes() / replication_count);
  }
}

std::vector<OutfeedStream>& Outfeed::Streams() { return streams_; }

Outfeed::Outfeed(const Json::Value& outfeed)
    : name_(outfeed["name"].asString()) {
  absl::c_transform(
      outfeed["streams"], std::back_inserter(streams_),
      [](const Json::Value& stream) {
        return OutfeedStream{TensorInfo{stream, TensorType::Outfeed}};
      });
}

void Outfeed::SetOutputFolder(const std::string& output_folder) {
  absl::c_for_each(streams_, [&output_folder](OutfeedStream& stream) {
    stream.SetOutputFolder(output_folder);
  });
}

void Outfeed::IgnoreOutput() {
  absl::c_for_each(streams_,
                   [](OutfeedStream& stream) { stream.IgnoreOutput(); });
}

std::vector<InfeedStream>& Infeed::MutableStreams() { return streams_; }
const std::vector<InfeedStream>& Infeed::Streams() const { return streams_; }

const std::string& Infeed::Name() const { return name_; }

Infeed::Infeed(const Json::Value& infeed) : name_(infeed["name"].asString()) {
  absl::c_transform(
      infeed["streams"], std::back_inserter(streams_),
      [](const Json::Value& stream) {
        return InfeedStream{TensorInfo{stream, TensorType::Infeed}};
      });
}

void Infeed::InitializeDataSources(const std::string& filename) {
  int64_t prev_num_elements;
  for (int i = 0; i < streams_.size(); i++) {
    LogContext ctx{absl::StrCat("stream ", i)};
    streams_[i].InitializeDataSource(Infeed::StreamFilename(filename, i));
  }
  ERROR_ON(!absl::c_all_of(streams_, [this](const InfeedStream& stream) {
    return stream.NumTensors() == this->streams_[0].NumTensors();
  }));
}

/* static */ std::string Infeed::StreamFilename(const std::string& filename,
                                                int64_t stream_idx) {
  size_t dot_pos = filename.rfind(".");
  ERROR_ON_MSG(dot_pos == std::string::npos, "Invalid filename: no extension");
  std::string basename = filename.substr(0, dot_pos);
  std::string extension = filename.substr(dot_pos);
  return absl::StrCat(basename, ".", stream_idx, extension);
}
}  // namespace ipu
