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
#include "ipu/poplar_executable_data.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <streambuf>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "third_party/eigen3/Eigen/Core"

namespace ipu {

namespace {

class BinaryVersion {
 public:
  void ToStream(StreamWriter& out);
  /* A stream is compatible with the current binary version if the major
   * versions are identical and the stream's minor version is less or equal to
   * the current version. */
  void ErrorIfNotCompatible(StreamReader& in);

 private:
  // Increment minor only when backward compatibility is maintained (e.g new
  // features are added) Increment major and reset minor to 0 otherwise.
  int major{4};
  int minor{0};
};

class TensorTypeInfo {
  struct TypeInfo {
    TensorType type;
    std::string type_str;
  };

 public:
  static TensorType FromString(const std::string& type_str);
  static const std::string& ToString(TensorType type);
  static std::vector<TensorTypeInfo::TypeInfo> CreateInfo();

 private:
  static const std::vector<TypeInfo> info_;
};

const std::vector<TensorTypeInfo::TypeInfo> TensorTypeInfo::info_ =
    TensorTypeInfo::CreateInfo();

/* static */ std::vector<TensorTypeInfo::TypeInfo>
TensorTypeInfo::CreateInfo() {
  std::vector<TensorTypeInfo::TypeInfo> m;
  m.emplace_back(TensorTypeInfo::TypeInfo{TensorType::NotSet, "not_set"});
  m.emplace_back(TensorTypeInfo::TypeInfo{TensorType::Parameter, "parameter"});
  m.emplace_back(TensorTypeInfo::TypeInfo{TensorType::InputData, "input_data"});
  m.emplace_back(
      TensorTypeInfo::TypeInfo{TensorType::OutputData, "output_data"});
  m.emplace_back(
      TensorTypeInfo::TypeInfo{TensorType::ParameterOut, "parameter_out"});
  m.emplace_back(TensorTypeInfo::TypeInfo{TensorType::Infeed, "infeed"});
  m.emplace_back(TensorTypeInfo::TypeInfo{TensorType::Outfeed, "outfeed"});

  return m;
}

/* static */ TensorType TensorTypeInfo::FromString(
    const std::string& type_str) {
  auto it = absl::c_find_if(info_, [type_str](const TypeInfo& info) {
    return info.type_str == type_str;
  });
  ERROR_ON_MSG(it == info_.end(), "Unknown TensorType '" << type_str << "'");
  return it->type;
}

/* static */ const std::string& TensorTypeInfo::ToString(TensorType type) {
  auto it = absl::c_find_if(
      info_, [type](const TypeInfo& info) { return info.type == type; });
  ERROR_ON_MSG(it == info_.end(),
               "Unknown TensorType '" << static_cast<int64_t>(type) << "'");
  return it->type_str;
}

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
  DataIterator& operator++() {
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
  ERROR_ON(!out.empty());
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
    ++it;
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
      ++out;
    }
  } else {
    // Recursively process other dimensions.
    for (const auto& subarray : array) {
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
      ++current;
    }
    return values;
  }
  // Recursively process other dimensions.
  for (int64_t i = 0; i < num_elements; i++) {
    values.append(DimensionToJson(dimension + 1, shape, current));
  }
  return values;
}

void WriteJsonToStream(const Json::Value& root, std::ostream* sout) {
  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";

  std::unique_ptr<Json::StreamWriter> writer(json_builder.newStreamWriter());
  writer->write(root, sout);
}

void BinaryVersion::ToStream(StreamWriter& out) {
  std::stringstream ss;
  ss << "v" << major << "." << minor;
  out.WriteString(ss.str());
}

void BinaryVersion::ErrorIfNotCompatible(StreamReader& in) {
  LogContext ctx("checking version number");
  std::string version_str = in.ReadString(10);
  const std::regex expr("v([0-9]+)\\.([0-9]+)");
  std::smatch matches;
  ERROR_ON(!std::regex_match(version_str, matches, expr));
  ERROR_ON(matches.size() != 3);
  int stream_major = std::stoi(matches[1].str());
  int stream_minor = std::stoi(matches[2].str());
  ERROR_ON_MSG(stream_major != major,
               "Binary file format version mismatch: file uses "
                   << version_str << " whereas the parser is v" << major << "."
                   << minor);
  ERROR_ON(stream_minor > minor);
}

class DeferredSizeWriter {
 public:
  explicit DeferredSizeWriter(const std::shared_ptr<StreamWriter>& writer)
      : writer_(writer), start_(writer->CurrentPosition()) {
    writer_->WriteInt64(0);
  }
  void WriteSize() {
    if (writer_) {
      std::streampos end = writer_->CurrentPosition();
      ERROR_ON(static_cast<int64_t>(start_) + sizeof(int64_t) > end);
      writer_->MoveAbsolute(start_);
      writer_->WriteInt64(static_cast<int64_t>(end) -
                          static_cast<int64_t>(start_) - sizeof(int64_t));
      writer_->MoveAbsolute(end);
      writer_ = nullptr;
    }
  }

 private:
  std::shared_ptr<StreamWriter> writer_;
  std::streampos start_;
};

DeferredSizeWriter CreateObject(ObjectType type, const std::string& name,
                                const std::shared_ptr<StreamWriter>& writer) {
  writer->WriteInt64(static_cast<int64_t>(type));
  writer->WriteString(name);
  return DeferredSizeWriter{writer};
}

std::string ObjectTypeToString(ObjectType type) {
  switch (type) {
    case ObjectType::Feed: {
      return "feed";
    }
    case ObjectType::Tensor: {
      return "tensor";
    }
    case ObjectType::PoplarExecutable: {
      return "Poplar executable";
    }
    case ObjectType::PoplarMetadata: {
      return "Poplar metadata";
    }
    default:
      ERROR("Unknown ObjectType " << static_cast<int64_t>(type));
  }
}

template <typename Key, typename Value>
std::list<Key> GetMapKeys(const std::map<Key, Value>& m) {
  std::list<Key> keys;
  absl::c_transform(
      m, std::back_inserter(keys),
      [](const std::pair<Key, Value>& pair) { return pair.first; });
  return keys;
}

}  // namespace

std::string TensorTypeToString(TensorType type) {
  return TensorTypeInfo::ToString(type);
}

Exception::Exception(const std::string& msg) : std::runtime_error(msg) {}

LogContext::LogContext() : cleared_(true) {}

LogContext::LogContext(const std::string& context) : saved_context_(context_) {
  UpdateContext(context);
}

void LogContext::UpdateContext(const std::string& new_context) {
  context_ = absl::StrCat(saved_context_, " ", new_context);
  cleared_ = false;
  if (InfoEnabled()) {
    std::cout << "[" << context_ << "]" << std::endl;
  }
}

void LogContext::Clear() {
  if (!cleared_) {
    // Don't restore the saved context if we're handling an exception
    // we might want to recover the context later.
    if (!std::uncaught_exception()) {
      context_ = saved_context_;
    }
    cleared_ = true;
  }
}

LogContext::~LogContext() { Clear(); }
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

int64_t TensorShape::DataSizeInBytes() const {
  return NumElements() * ElementSizeInBytes() + metadata_size_;
}

bool TensorShape::HasMetadata() const { return metadata_size_ != 0; }

int64_t TensorShape::MetadataSize() const { return metadata_size_; }

void TensorShape::SetMetadataSize(int64_t metadata_size) {
  metadata_size_ = metadata_size;
}

DataType TensorShape::Type() const { return type_; }

TensorShape::TensorShape(const TensorShape& shape, int64_t metadata_size)
    : TensorShape(shape) {
  metadata_size_ = metadata_size;
}

TensorShape::TensorShape(const Json::Value& shape) {
  type_ = DataTypeInfo::FromString(shape["data_type"].asString());
  metadata_size_ = shape["metadata"].asInt64();
  Json::Value array = shape["shape"];
  ERROR_ON(!array.isArray());
  shape_.reserve(array.size());
  for (const auto& dim : array) {
    shape_.push_back(dim.asInt64());
  }
}

Json::Value TensorShape::ToJson() const {
  Json::Value shape;
  shape["data_type"] = DataTypeInfo::ToString(type_);
  Json::Value dims{Json::arrayValue};
  for (auto dim : shape_) {
    dims.append(Json::Value::Int64(dim));
  }
  shape["shape"] = dims;
  shape["metadata"] = Json::Value::Int64(metadata_size_);
  return shape;
}

std::string TensorShape::ToString() const { return ToJson().toStyledString(); }

void TensorShape::ToStream(StreamWriter& out) const {
  out.WriteInt64(type_);
  out.WriteInt64Array(shape_);
  out.WriteInt64(metadata_size_);
}

TensorShape::TensorShape(StreamReader& in)
    : type_(static_cast<DataType>(in.ReadInt64())),
      shape_(in.ReadInt64Array()),
      metadata_size_(in.ReadInt64()) {}

TensorShape::TensorShape(const std::vector<int64_t>& shape, DataType type)
    : shape_(shape), type_(type), metadata_size_(0) {}

bool TensorShape::operator==(const TensorShape& other) const {
  return type_ == other.type_ && shape_ == other.shape_;
}

std::string Tensor::ToString() const {
  std::stringstream ss;
  if (info_.Shape().HasMetadata()) {
    ss << "<not available>";
  } else {
    SaveDataToJsonStream(&ss);
  }
  return absl::StrCat(info_.ToString(), " data = ", ss.str());
}

void Tensor::SaveDataToJsonFile(const std::string& filename) const {
  LogContext ctx(
      absl::StrCat("Saving content of ", info_.Name(), " in ", filename));
  ERROR_ON_MSG(info_.Shape().HasMetadata(), "Cannot access verified data");
  std::ofstream out(filename);
  ERROR_ON_MSG(!out.is_open(), "Failed to open file " << filename);
  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  SaveDataToJsonStream(&out);
  out.close();
  PRINT_INFO("Saved content of " << info_.Name() << " to " << filename);
}

void Tensor::ToStream(StreamWriter& out) const {
  Info().ToStream(out);
  out.WriteData(data_.data(), Info().Shape().DataSizeInBytes());
}

void Tensor::LoadDataFromStream(StreamReader& in) {
  TensorInfo info{in};
  if (info_.Type() == TensorType::NotSet) {
    info_ = std::move(info);
  } else {
    ERROR_ON_MSG(
        !info_.TypeAndShapeMatch(info),
        "Type and Shape from metadata JSON file: "
            << info_.ToString()
            << " don't match those from binary file: " << info.ToString());
    info_.SetMetadataSize(info.Shape().MetadataSize());
  }
  data_.resize(info_.Shape().DataSizeInBytes());
  in.ReadData(data_.data(), info_.Shape().DataSizeInBytes());
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

TensorInfo::TensorInfo(const Json::Value& info, TensorType type)
    : TensorInfo(info["name"].asString(), info["handle"].asString(),
                 TensorShape{info["shape"]}, type) {
  PRINT_INFO("Found " << ToString());
}

Json::Value TensorInfo::ToJson() const {
  Json::Value info;
  info["name"] = name_;
  info["handle"] = handle_;
  info["shape"] = shape_.ToJson();
  info["type"] = TensorTypeInfo::ToString(type_);
  return info;
}

TensorInfo::TensorInfo(const Json::Value& info)
    : TensorInfo(info, TensorTypeInfo::FromString(info["type"].asString())) {}

TensorInfo::TensorInfo(StreamReader& in)
    : name_(in.ReadString()),
      handle_(in.ReadString()),
      shape_(in),
      type_(static_cast<TensorType>(in.ReadInt64())) {}

TensorInfo::TensorInfo(const std::string& name, const std::string& handle,
                       const TensorShape& shape, TensorType type)
    : name_(name), handle_(handle), shape_(shape), type_(type) {}

void TensorInfo::SetMetadataSize(int64_t metadata_size) {
  shape_.SetMetadataSize(metadata_size);
}

void TensorInfo::SetShape(const TensorShape& shape) { shape_ = shape; }

void TensorInfo::SetName(const std::string& name) { name_ = name; }

void TensorInfo::SetHandle(const std::string& handle) { handle_ = handle; }

void TensorInfo::SetType(const TensorType type) { type_ = type; }

bool TensorInfo::TypeAndShapeMatch(const TensorInfo& other) const {
  bool type_match = type_ == other.type_;
  // Types don't need to be an exact match: they just need to
  // be compatible.
  if (!type_match && type_ == TensorType::Parameter) {
    type_match = other.type_ == TensorType::ParameterOut;
  }
  if (!type_match && type_ == TensorType::InputData) {
    type_match = other.type_ == TensorType::OutputData;
  }
  if (!type_match && type_ == TensorType::Infeed) {
    type_match = other.type_ == TensorType::Outfeed;
  }
  return type_match && shape_ == other.shape_;
}

void TensorInfo::ToStream(StreamWriter& out) const {
  out.WriteString(name_);
  out.WriteString(handle_);
  shape_.ToStream(out);
  out.WriteInt64(static_cast<int64_t>(type_));
}

std::string TensorInfo::ToString() const { return ToJson().toStyledString(); }

const TensorShape& TensorInfo::Shape() const { return shape_; }

const std::string& TensorInfo::Name() const { return name_; }
const std::string& TensorInfo::Handle() const { return handle_; }
TensorType TensorInfo::Type() const { return type_; }

Tensor::Tensor(StreamReader& reader) { LoadDataFromStream(reader); }

Tensor::Tensor(const TensorInfo& info) : info_(info) {
  data_.resize(info_.Shape().DataSizeInBytes());
}

Tensor::Tensor(const TensorInfo& info, const void* data) : Tensor(info) {
  memcpy(data_.data(), data, info_.Shape().DataSizeInBytes());
}

const TensorInfo& Tensor::Info() const { return info_; }

void Tensor::LoadDataFromJsonFile(const std::string& data_filename) {
  LogContext ctx(
      absl::StrCat("Loading ", TensorTypeInfo::ToString(info_.Type()), " ",
                   info_.Name(), " from JSON file '", data_filename, "'"));
  try {
    NDArrayParser parser;
    data_.clear();
    parser(LoadJsonFromFile(data_filename), info_.Shape(), data_);
  } catch (const std::out_of_range& error) {
    ERROR(error.what());
  }
}

void* Tensor::Data() {
  ERROR_ON(data_.empty());
  ERROR_ON_MSG(data_.size() != info_.Shape().DataSizeInBytes(),
               "Buffer is of size " << data_.size() << " but the shape "
                                    << info_.Shape().ToString() << " requires "
                                    << info_.Shape().DataSizeInBytes());
  return data_.data();
}

std::string SanitizeName(const std::string& filename) {
  std::string out = filename;
  absl::c_replace(out, '/', '_');
  return out;
}

bool IsJsonFile(const std::string& filename) {
  std::ifstream json_file(filename);
  if (!json_file.is_open()) {
    return false;
  }
  char first_char;
  json_file >> first_char;
  // Note: This is only intended as a quick check to differentiate binary files
  // from text json files. This relies on the fact that binary files start by a
  // BinaryVersion::ToStream()(i.e vXX.YY in binary) while Json files are text
  // files starting by an ascii '{'
  return first_char == '{';
}

Json::Value LoadJsonFromFile(const std::string& filename) {
  std::ifstream json_file(filename);
  ERROR_ON_MSG(!json_file.is_open(), "Failed to open file " << filename);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;
  Json::Value root;
  ERROR_ON_MSG(!parseFromStream(builder, json_file, &root, &errs), errs);
  return root;
}

Json::Value LoadJsonFromString(const std::string& json_content) {
  Json::CharReaderBuilder builder;
  const char* start = json_content.c_str();
  const char* end = start + json_content.size();
  JSONCPP_STRING errs;
  Json::Value root;
  std::unique_ptr<Json::CharReader> const reader(builder.newCharReader());
  ERROR_ON_MSG(!reader->parse(start, end, &root, &errs), errs);
  return root;
}

std::ofstream& StreamWriter::Stream() { return fd_; }

void StreamWriter::WriteInt64(int64_t value) {
  WriteData(&value, sizeof(value));
}

void StreamWriter::WriteInt64Array(const std::vector<int64_t>& values) {
  WriteInt64(values.size());
  if (!values.empty()) {
    WriteData(values.data(), sizeof(int64_t) * values.size());
  }
}

void StreamWriter::WriteData(const void* data, size_t size) {
  ERROR_ON(size == 0);
  fd_.write(reinterpret_cast<const char*>(data), size);
}

void StreamWriter::WriteString(const std::string& value) {
  WriteInt64(value.size());
  if (!value.empty()) {
    WriteData(value.c_str(), value.size());
  }
}

void StreamReader::SetEnd(std::streampos end) { end_ = end; }

const std::string& StreamReader::Filename() const { return filename_; }

std::string StreamReader::ReadString(int64_t max_len) {
  int64_t len = ReadInt64();
  ERROR_ON_MSG(max_len > 0 && len > max_len,
               "Invalid string (Len " << len << "> max " << max_len << ")");
  ERROR_ON_MSG(len > 1048576, "File corrupted: string of length " << len);
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
  fd_.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  BinaryVersion().ToStream(*this);
}

void StreamWriter::Close() {
  if (fd_.is_open()) {
    fd_.close();
  }
}

void StreamWriter::CopyFromStream(StreamReader& in, size_t size) {
  ERROR_ON(size > in.NumBytesLeft());
  std::vector<char> buffer;
  buffer.resize(size);
  in.ReadData(buffer.data(), buffer.size());
  WriteData(buffer.data(), buffer.size());
}

void StreamWriter::MoveAbsolute(std::streampos position) {
  fd_.seekp(position);
}

std::streampos StreamWriter::CurrentPosition() { return fd_.tellp(); }

StreamReader::StreamReader(const std::string& filename, bool is_versioned)
    : fd_(filename, std::ifstream::binary), filename_(filename) {
  ERROR_ON_MSG(!fd_.is_open(), "Failed to open file '" << filename << "'");
  fd_.exceptions(std::ifstream::eofbit | std::ifstream::failbit |
                 std::ifstream::badbit);
  std::streampos begin = fd_.tellg();
  fd_.seekg(0, std::ifstream::end);
  end_ = fd_.tellg();
  fd_.seekg(0, std::ifstream::beg);
  if (is_versioned) {
    BinaryVersion().ErrorIfNotCompatible(*this);
  }
}

StreamReader StreamReader::Clone() {
  StreamReader clone{filename_};
  clone.MoveAbsolute(CurrentPosition());
  return clone;
}

int64_t StreamReader::NumBytesLeft() { return end_ - fd_.tellg(); }

void StreamReader::ReadData(void* dst, int64_t length) {
  ERROR_ON(length <= 0);
  ERROR_ON_MSG(fd_.tellg() + length > end_,
               "Trying to read past the end of the stream");
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

void StreamReader::MoveRelative(std::streamoff offset) {
  fd_.seekg(offset, std::ifstream::cur);
}

void StreamReader::MoveAbsolute(std::streampos position) {
  fd_.seekg(position);
}

std::streampos StreamReader::CurrentPosition() { return fd_.tellg(); }

std::ifstream& StreamReader::Stream() { return fd_; }

OutfeedStream::OutfeedStream(const TensorInfo& info) : info_(info) {}

void OutfeedStream::UpdateNumTensorsAndClose() {
  auto end = writer_->CurrentPosition();
  writer_->MoveAbsolute(data_size_pos_);
  writer_->WriteInt64(static_cast<int64_t>(end) -
                      static_cast<int64_t>(data_size_pos_) - sizeof(int64_t));
  writer_->MoveAbsolute(num_tensors_pos_);
  int64_t num_bytes = end - num_tensors_pos_ - sizeof(int64_t);
  ERROR_ON(num_bytes % info_.Shape().DataSizeInBytes() != 0);
  writer_->WriteInt64(num_bytes / info_.Shape().DataSizeInBytes());
  writer_->MoveAbsolute(end);
  writer_->Close();
}

void OutfeedStream::SetOutputFolder(const std::string& output_folder) {
  const std::string filename =
      absl::StrCat(output_folder, "/", SanitizeName(info_.Name()), ".bin");
  if (writer_) {
    UpdateNumTensorsAndClose();
  }
  writer_ = std::make_shared<StreamWriter>(filename);
  // Must match the information in BinaryWriter::CreateFeed
  writer_->WriteInt64(static_cast<int64_t>(ObjectType::Feed));
  writer_->WriteString(info_.Name());
  // Set data size to 0: we don't know how many elements we're going to write.
  data_size_pos_ = writer_->CurrentPosition();
  writer_->WriteInt64(0);
  info_.ToStream(*writer_);
  // Set num_tensors to 0: we don't know how many elements we're going to write.
  num_tensors_pos_ = writer_->CurrentPosition();
  writer_->WriteInt64(0);
}

void OutfeedStream::IgnoreOutput() {
  if (writer_) {
    UpdateNumTensorsAndClose();
  }
  writer_ = nullptr;
}

OutfeedStream::~OutfeedStream() {
  // Close the file.
  IgnoreOutput();
}

const TensorInfo& OutfeedStream::Info() const { return info_; }

void OutfeedStream::WriteTensor(void* src, int64_t replication_count) {
  if (writer_) {
    writer_->WriteData(src,
                       info_.Shape().DataSizeInBytes() / replication_count);
  }
}

std::vector<OutfeedStream>& Outfeed::Streams() { return streams_; }

Outfeed::Outfeed(const FeedInfo& info) : name_(info.name) {
  for (auto stream : info.streams) {
    streams_.emplace_back(stream);
  }
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

FeedWriter::FeedWriter(std::shared_ptr<StreamWriter> writer,
                       int64_t tensor_size, int64_t num_tensors)
    : writer_(writer),
      tensor_size_(tensor_size),
      current_pos_(writer->CurrentPosition()) {
  std::vector<char> dummy_buffer;
  dummy_buffer.resize(tensor_size);
  // Reserve the space:
  for (int i = 0; i < num_tensors; i++) {
    writer_->WriteData(dummy_buffer.data(), tensor_size);
  }
  end_pos_ = writer_->CurrentPosition();
}

void FeedWriter::AppendTensor(const void* data) {
  ERROR_ON(current_pos_ >= end_pos_);
  // Save the current writer's position
  std::streampos saved_pos = writer_->CurrentPosition();
  // Jump to the feed's location in the file
  writer_->MoveAbsolute(current_pos_);
  // Write the tensor
  writer_->WriteData(data, tensor_size_);
  // Move current_pos_ to the next slot
  current_pos_ = writer_->CurrentPosition();
  // Restore the writer to its current position
  writer_->MoveAbsolute(saved_pos);
}

BinaryWriter::BinaryWriter(const std::string& filename)
    : writer_(std::make_shared<StreamWriter>(filename)) {}

FeedWriter BinaryWriter::CreateFeed(const std::string& name,
                                    const TensorInfo& info,
                                    int64_t num_tensors) {
  DeferredSizeWriter size_writer{CreateObject(ObjectType::Feed, name, writer_)};
  info.ToStream(*writer_);
  writer_->WriteInt64(num_tensors);
  FeedWriter feed{writer_, info.Shape().DataSizeInBytes(), num_tensors};
  size_writer.WriteSize();
  return feed;
}

ExecutableWriter BinaryWriter::CreateExecutable(const std::string& name,
                                                bool is_verified) {
  DeferredSizeWriter size_writer{
      CreateObject(ObjectType::PoplarExecutable, name, writer_)};
  writer_->WriteInt64(static_cast<int64_t>(is_verified));
  std::function<void()> on_write_complete =
      std::bind(&DeferredSizeWriter::WriteSize, size_writer);
  return ExecutableWriter(writer_, on_write_complete);
}

void BinaryWriter::WriteMetadata(const std::string& name,
                                 const Metadata& metadata) {
  DeferredSizeWriter size_writer{
      CreateObject(ObjectType::PoplarMetadata, name, writer_)};
  writer_->WriteString(metadata.ToJson());
  size_writer.WriteSize();
}

void BinaryWriter::WriteTensor(const Tensor& tensor,
                               const std::string override_name) {
  DeferredSizeWriter size_writer{CreateObject(
      ObjectType::Tensor,
      override_name.empty() ? tensor.Info().Name() : override_name, writer_)};
  tensor.ToStream(*writer_);
  size_writer.WriteSize();
}

void BinaryWriter::Close() {
  if (writer_) {
    writer_->Close();
    writer_ = nullptr;
  }
}

BinaryWriter::~BinaryWriter() { Close(); }

std::ofstream& ExecutableWriter::Stream() { return writer_->Stream(); }

void ExecutableWriter::WriteComplete() {
  if (writer_) {
    on_write_complete_();
    writer_ = nullptr;
  }
}

StreamWriter& ExecutableWriter::Writer() { return *writer_; }

ExecutableWriter::~ExecutableWriter() { WriteComplete(); }

ExecutableWriter::ExecutableWriter(std::shared_ptr<StreamWriter> writer,
                                   std::function<void()> on_write_complete)
    : writer_(std::move(writer)),
      on_write_complete_(std::move(on_write_complete)) {}

void BinaryReader::LoadFile(const std::string& filename) {
  LogContext ctx{"Loading binary file '" + filename + "'"};
  StreamReader reader{filename};
  while (reader.NumBytesLeft() > 0) {
    int64_t type_int = reader.ReadInt64();
    ERROR_ON_MSG(type_int < 0 || type_int > static_cast<int64_t>(
                                                ObjectType::PoplarMetadata),
                 "Invalid ObjectType '" << type_int << "'");
    ObjectType type = static_cast<ObjectType>(type_int);
    std::string name = reader.ReadString();
    Object obj;
    int64_t size = reader.ReadInt64();
    ERROR_ON_MSG(size <= 0, "Invalid object size " << size << " for "
                                                   << ObjectTypeToString(type)
                                                   << " '" << name << "': ");
    obj.filename = filename;
    obj.offset = reader.CurrentPosition();
    reader.MoveRelative(size);
    obj.end = reader.CurrentPosition();
    PRINT_INFO("Obj " << obj.filename << " off " << obj.offset << " end "
                      << obj.end << " Name " << name << " Type "
                      << ObjectTypeToString(type));
    objects_[type][name] = obj;
  }
}

std::unique_ptr<StreamReader> BinaryReader::GetObjectReader(
    ObjectType type, const std::string& name) const {
  const Object& obj = GetObject(type, name);
  std::unique_ptr<StreamReader> in =
      absl::make_unique<StreamReader>(obj.filename);
  in->MoveAbsolute(obj.offset);
  in->SetEnd(obj.end);
  return in;
}

std::unique_ptr<Metadata> BinaryReader::ReadMetadata(
    const std::string& metadata_name) const {
  LogContext ctx{"BinaryReader::CreateMetadataReader" + metadata_name};
  auto reader = GetObjectReader(ObjectType::PoplarMetadata, metadata_name);
  return absl::make_unique<Metadata>(LoadJsonFromString(reader->ReadString()));
}

std::unique_ptr<StreamReader> BinaryReader::CreateExecutableReader(
    const std::string& executable_name) const {
  LogContext ctx{"BinaryReader::CreateExecutableReader " + executable_name};
  return GetObjectReader(ObjectType::PoplarExecutable, executable_name);
}

const BinaryReader::Object BinaryReader::GetObject(
    ObjectType type, const std::string& name) const {
  auto objects_it = objects_.find(type);
  ERROR_ON_MSG(objects_it == objects_.end(),
               "No object of type " << ObjectTypeToString(type) << " loaded");
  auto objects = objects_it->second;
  ERROR_ON_MSG(name.empty() && objects.size() > 1,
               "BinaryLoader contains more than one "
                   << ObjectTypeToString(type)
                   << " and "
                      "you did not specify which one to load: ["
                   << absl::StrJoin(GetMapKeys(objects), ", ") << "]");
  if (name.empty()) {
    return objects.begin()->second;
  }
  auto object_it = objects.find(name);
  ERROR_ON_MSG(object_it == objects.end(),
               "Could not find "
                   << ObjectTypeToString(type) << " named '" << name << "': ["
                   << absl::StrJoin(GetMapKeys(objects), ", ") << "]");
  return object_it->second;
}

std::unique_ptr<StreamReader> BinaryReader::CreateInfeedStreamReader(
    const std::string& infeed_name) const {
  LogContext ctx{"BinaryReader::CreateInfeedStreamReader " + infeed_name};
  return GetObjectReader(ObjectType::Feed, infeed_name);
}

bool BinaryReader::ContainsObject(ObjectType type,
                                  const std::string& name) const {
  auto objects_it = objects_.find(type);
  if (objects_it == objects_.end()) {
    return false;
  }
  return objects_it->second.find(name) != objects_it->second.end();
}

std::unique_ptr<StreamReader> BinaryReader::GetTensorStream(
    const std::string& name) const {
  LogContext ctx{"BinaryLoader::GetTensorStream " + name};
  return GetObjectReader(ObjectType::Tensor, name);
}

std::set<std::string> BinaryReader::GetObjectNames(ObjectType type) const {
  std::set<std::string> names;
  auto objects_it = objects_.find(type);
  if (objects_it != objects_.end()) {
    for (auto obj : objects_it->second) {
      names.insert(obj.first);
    }
  }
  return names;
}

std::set<std::string> BinaryReader::GetObjectSummaries(ObjectType type) const {
  ERROR_ON_MSG(type != ObjectType::Feed && type != ObjectType::Tensor,
               "Summaries only supported for feeds and tensors");
  std::set<std::string> summaries;
  auto objects_it = objects_.find(type);
  if (objects_it != objects_.end()) {
    Json::FastWriter writer;
    for (auto obj : objects_it->second) {
      switch (type) {
        case ObjectType::Tensor: {
          Tensor tmp{*GetTensorStream(obj.first)};
          summaries.insert(writer.write(tmp.Info().ToJson()));
          break;
        }
        case ObjectType::Feed: {
          InfeedStream tmp{CreateInfeedStreamReader(obj.first)};
          Json::Value json = tmp.Info().ToJson();
          json["num_tensors"] = Json::Value::Int64(tmp.NumTensors());
          summaries.insert(writer.write(json));
          break;
        }
      }
    }
  }
  return summaries;
}

InfeedStream::InfeedStream(const TensorInfo& info) : info_(info) {}

const TensorInfo& InfeedStream::Info() const { return info_; }

std::string InfeedStream::ToString() {
  // Backup current position
  int64_t tensor_idx = tensor_idx_;
  std::streampos current_pos = reader_->CurrentPosition();

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

InfeedStream::InfeedStream(std::shared_ptr<StreamReader> in) {
  InitializeDataSource(std::move(in));
}

void InfeedStream::InitializeDataSource(std::shared_ptr<StreamReader> in) {
  reader_ = std::move(in);
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
    info_.SetMetadataSize(info.Shape().MetadataSize());
  }
  num_tensors_ = reader_->ReadInt64();
  ERROR_ON(num_tensors_ <= 0);
  ERROR_ON(reader_->NumBytesLeft() <
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

/* static */ const std::string& Metadata::CheckpointName() {
  static const std::string value = "checkpoint";
  return value;
}

/* static */ const std::string& Metadata::ClearCheckpointName() {
  static const std::string value = "checkpointClear";
  return value;
}

/* static */ const std::string& Metadata::InputCheckpointIndexHandle() {
  static const std::string value = "checkpointIndex";
  return value;
}

/* static */ const std::string& Metadata::InputCheckpointIndexName() {
  return InputCheckpointIndexHandle();
}

/* static */ const std::string& Metadata::InputCheckpointHandle() {
  static const std::string value = "checkpointIn";
  return value;
}

/* static */ const std::string& Metadata::OutputCheckpointHandle() {
  static const std::string value = "checkpointOut";
  return value;
}

/* static */ const std::string& Metadata::OutputClearCheckpointHandle() {
  static const std::string value = "checkpointOutClear";
  return value;
}

FeedInfo::FeedInfo(const Json::Value& info) : name(info["name"].asString()) {
  for (const auto& stream : info["streams"]) {
    streams.emplace_back(TensorInfo{stream});
  }
  ERROR_ON(streams.empty());
}

Json::Value FeedInfo::ToJson() const {
  ERROR_ON(streams.empty());
  Json::Value feed;
  Json::Value json_streams;
  feed["name"] = name;
  for (const auto& stream : streams) {
    json_streams.append(stream.ToJson());
  }
  feed["streams"] = json_streams;
  return feed;
}

std::string Metadata::ToJson() const {
  Json::Value json_inputs;
  for (const auto& input : inputs) {
    json_inputs.append(input.ToJson());
  }

  Json::Value json_outputs;
  for (const auto& output : outputs) {
    json_outputs.append(output.ToJson());
  }

  Json::Value json_infeeds;
  for (const auto& infeed : infeeds) {
    json_infeeds.append(infeed.ToJson());
  }

  Json::Value json_outfeeds;
  for (const auto& outfeed : outfeeds) {
    json_outfeeds.append(outfeed.ToJson());
  }

  Json::Value config;
  Json::Value json_device_options;
  for (auto opt : device_options) {
    json_device_options[opt.first] = opt.second;
  }
  if (!json_device_options.empty()) {
    config["device_options"] = json_device_options;
  }

  Json::Value json_engine_options;
  for (auto opt : engine_options) {
    json_engine_options[opt.first] = opt.second;
  }
  if (!json_engine_options.empty()) {
    config["engine_options"] = json_engine_options;
  }

  config["replication_count"] = Json::Value::Int64(replication_count);
  config["num_ipus"] = Json::Value::Int64(num_ipus);
  config["random_number_seed_handle"] = random_number_seed_handle;

  Json::Value checkpoint;
  if (!feeds_order.empty()) {
    Json::Value streams;
    for (const auto& feed : feeds_order) {
      streams.append(feed);
    }
    checkpoint["feeds"] = streams;
  }

  Json::Value verification;
  for (auto info : verification_info) {
    verification[info.first] = info.second.ToJson();
  }

  Json::Value root;
  if (!json_inputs.empty()) {
    root["inputs"] = json_inputs;
  }
  if (!json_outputs.empty()) {
    root["outputs"] = json_outputs;
  }
  if (!json_infeeds.empty()) {
    root["infeeds"] = json_infeeds;
  }
  if (!json_outfeeds.empty()) {
    root["outfeeds"] = json_outfeeds;
  }
  if (!feeds_order.empty()) {
    root["checkpoint"] = checkpoint;
  }
  if (!verification.empty()) {
    root["verification_info"] = verification;
  }
  root["config"] = config;

  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";

  return Json::writeString(json_builder, root);
}

Metadata::Metadata(const Json::Value& root) {
  for (const auto& input : root["inputs"]) {
    inputs.emplace_back(TensorInfo{input});
  }
  for (const auto& output : root["outputs"]) {
    outputs.emplace_back(TensorInfo{output});
  }
  for (const auto& infeed : root["infeeds"]) {
    infeeds.emplace_back(FeedInfo{infeed});
  }
  for (const auto& outfeed : root["outfeeds"]) {
    outfeeds.emplace_back(FeedInfo{outfeed});
  }
  Json::Value config = root["config"];
  replication_count = config["replication_count"].asInt64();
  num_ipus = config["num_ipus"].asInt64();
  random_number_seed_handle = config["random_number_seed_handle"].asString();

  Json::Value json_device_options = config["device_options"];
  if (!json_device_options.isNull()) {
    for (const auto& key : json_device_options.getMemberNames()) {
      device_options[key] = json_device_options[key].asString();
    }
  }

  Json::Value json_engine_options = config["engine_options"];
  if (!json_engine_options.isNull()) {
    for (const auto& key : json_engine_options.getMemberNames()) {
      engine_options[key] = json_engine_options[key].asString();
    }
  }

  Json::Value checkpoint = root["checkpoint"];
  for (const auto& feed : checkpoint["feeds"]) {
    feeds_order.push_back(feed.asString());
  }

  Json::Value verification = root["verification_info"];
  if (!verification.isNull()) {
    for (const auto& handle : verification.getMemberNames()) {
      verification_info[handle] = VerificationInfo(verification[handle]);
    }
  }
}

VerificationInfo::VerificationInfo() : initialised_(false), key_(-1), id_(-1) {}

VerificationInfo::VerificationInfo(int64_t key, int64_t id) {
  SetInfo(key, id);
}

bool VerificationInfo::Initialised() const { return initialised_; }

int64_t VerificationInfo::Key() const { return key_; }

int64_t VerificationInfo::Id() const { return id_; }

void VerificationInfo::SetInfo(int64_t key, int64_t id) {
  initialised_ = true;
  key_ = key;
  id_ = id;
}

VerificationInfo::VerificationInfo(const Json::Value& info)
    : key_(info["key"].asInt64()),
      id_(info["id"].asInt64()),
      initialised_(true) {}

Json::Value VerificationInfo::ToJson() const {
  ERROR_ON(!Initialised());
  Json::Value out;
  out["key"] = Json::Value::Int64(key_);
  out["id"] = Json::Value::Int64(id_);
  return out;
}

void MetadataBuilder::AddVerificationInfo(const std::string& handle,
                                          const VerificationInfo& info) {
  ERROR_ON_MSG(
      meta_.verification_info.find(handle) != meta_.verification_info.end(),
      "Handle " << handle << " already in use");
  if (info.Initialised()) {
    meta_.verification_info.emplace(handle, info);
  }
}

void MetadataBuilder::AddInput(const TensorInfo& tensor,
                               const VerificationInfo& info) {
  AddVerificationInfo(tensor.Handle(), info);
  meta_.inputs.push_back(tensor);
  meta_.inputs.back().SetType(TensorType::InputData);
}

void MetadataBuilder::AddInputParameter(const TensorInfo& tensor,
                                        const VerificationInfo& info) {
  AddVerificationInfo(tensor.Handle(), info);
  meta_.inputs.push_back(tensor);
  meta_.inputs.back().SetType(TensorType::Parameter);
  ERROR_ON_MSG(
      !input_parameters_.emplace(tensor.Handle(), meta_.inputs.back()).second,
      "Already contains an input parameter with handle '" << tensor.Handle()
                                                          << "'");
}

void MetadataBuilder::AddOutput(const TensorInfo& tensor,
                                const VerificationInfo& info) {
  AddVerificationInfo(tensor.Handle(), info);
  meta_.outputs.push_back(tensor);
  meta_.outputs.back().SetType(TensorType::OutputData);
}

void MetadataBuilder::AddOutputParameter(const TensorInfo& tensor,
                                         const VerificationInfo& info) {
  AddVerificationInfo(tensor.Handle(), info);
  meta_.outputs.push_back(tensor);
  meta_.outputs.back().SetType(TensorType::ParameterOut);
}

void MetadataBuilder::AddOutputModifiedParameter(
    const std::string& input_handle, const TensorInfo& tensor,
    const VerificationInfo& info) {
  TensorInfo to_add = tensor;
  to_add.SetType(TensorType::ParameterOut);
  ERROR_ON_MSG(input_parameters_.find(input_handle) == input_parameters_.end(),
               "Input parameter " << input_handle << " not found");
  auto input = input_parameters_.at(input_handle);
  ERROR_ON_MSG(!input.TypeAndShapeMatch(to_add),
               "Type and Shape from tensor: "
                   << to_add.ToString()
                   << " don't match those from input: " << input.ToString());
  ERROR_ON_MSG(input.Name() != to_add.Name(),
               "Output parameters name "
                   << to_add.Name() << " doesn't match the name of the input "
                   << input.Name());
  if (info.Initialised()) {
    ERROR_ON_MSG(meta_.verification_info.at(input_handle).Id() != info.Id(),
                 "Input/Output VerificationInfo's ID mismatch");
  } else {
    ERROR_ON_MSG(meta_.verification_info.find(input_handle) !=
                     meta_.verification_info.end(),
                 "Input has some VerificationInfo but the output doesn't");
  }
  AddVerificationInfo(to_add.Handle(), info);
  meta_.outputs.push_back(to_add);
  meta_.outputs.back().SetType(TensorType::ParameterOut);
}

void MetadataBuilder::CreateInfeed(const std::string& name) {
  ERROR_ON_MSG(infeeds_.find(name) != infeeds_.end(),
               "Infeed " << name << " already exists");
  meta_.infeeds.emplace_back();
  meta_.infeeds.back().name = name;
  infeeds_[name] = meta_.infeeds.size() - 1;
}

void MetadataBuilder::AddInfeedStream(const std::string& infeed_name,
                                      const TensorInfo& tensor,
                                      const VerificationInfo& info) {
  auto& streams = meta_.infeeds.at(infeeds_.at(infeed_name)).streams;
  AddVerificationInfo(tensor.Handle(), info);
  streams.push_back(tensor);
  streams.back().SetType(TensorType::Infeed);
}

void MetadataBuilder::CreateOutfeed(const std::string& name) {
  ERROR_ON_MSG(outfeeds_.find(name) != outfeeds_.end(),
               "Outfeed " << name << " exists already");
  meta_.outfeeds.emplace_back();
  meta_.outfeeds.back().name = name;
  outfeeds_[name] = meta_.outfeeds.size() - 1;
}

void MetadataBuilder::AddOutfeedStream(const std::string& outfeed_name,
                                       const TensorInfo& tensor,
                                       const VerificationInfo& info) {
  auto& streams = meta_.outfeeds.at(outfeeds_.at(outfeed_name)).streams;
  AddVerificationInfo(tensor.Handle(), info);
  streams.push_back(tensor);
  streams.back().SetType(TensorType::Outfeed);
}

void MetadataBuilder::AddDeviceOption(const std::string& key,
                                      const std::string& value) {
  ERROR_ON_MSG(!meta_.device_options.emplace(key, value).second,
               "DeviceOption " << key << " already added");
}

void MetadataBuilder::AddEngineOption(const std::string& key,
                                      const std::string& value) {
  ERROR_ON_MSG(!meta_.engine_options.emplace(key, value).second,
               "EngineOption " << key << " already added");
}

void MetadataBuilder::SetConfig(int64_t replication_count, int64_t num_ipus) {
  meta_.replication_count = replication_count;
  meta_.num_ipus = num_ipus;
}

void MetadataBuilder::SetRandomNumberSeedHandle(const std::string& handle) {
  meta_.random_number_seed_handle = handle;
}

void MetadataBuilder::AddCheckpoint(const std::vector<std::string>& feeds_order,
                                    const VerificationInfo& checkpointInInfo,
                                    const VerificationInfo& checkpointOutInfo) {
  ERROR_ON_MSG(!meta_.feeds_order.empty(), "Checkpoint already added");
  // Indices are 64 bits but Poplar only supports S32, so multiply by 2:
  TensorInfo inputCheckpoint{
      Metadata::CheckpointName(), Metadata::InputCheckpointHandle(),
      TensorShape({static_cast<int64_t>(2 * feeds_order.size())},
                  DataType::S32),
      TensorType::Parameter};
  AddVerificationInfo(inputCheckpoint.Handle(), checkpointInInfo);
  meta_.inputs.push_back(inputCheckpoint);

  TensorInfo inputCheckpointIndex{Metadata::InputCheckpointIndexName(),
                                  Metadata::InputCheckpointIndexHandle(),
                                  TensorShape({2}, DataType::S32),
                                  TensorType::InputData};
  meta_.inputs.push_back(inputCheckpointIndex);

  TensorInfo outputCheckpoint = inputCheckpoint;
  outputCheckpoint.SetHandle(Metadata::OutputCheckpointHandle());
  outputCheckpoint.SetType(TensorType::ParameterOut);
  AddVerificationInfo(outputCheckpoint.Handle(), checkpointOutInfo);
  meta_.outputs.push_back(outputCheckpoint);

  TensorInfo outputClearCheckpoint = outputCheckpoint;
  outputClearCheckpoint.SetHandle(Metadata::OutputClearCheckpointHandle());
  outputClearCheckpoint.SetName(Metadata::ClearCheckpointName());
  outputClearCheckpoint.SetType(TensorType::OutputData);
  meta_.outputs.push_back(outputClearCheckpoint);

  meta_.feeds_order = feeds_order;
}

Metadata MetadataBuilder::BuildMetadata() const { return meta_; }

}  // namespace ipu
