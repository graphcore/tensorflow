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
#include "tensorflow/compiler/plugin/poplar/tools/poplar_executable_data.h"

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
  int major{2};
  int minor{0};
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
  explicit DeferredSizeWriter(std::shared_ptr<StreamWriter> writer)
      : writer_(writer), start_(writer->CurrentPosition()) {
    writer->WriteInt64(0);
  }
  void WriteSize() {
    if (writer_) {
      std::ios::streampos end = writer_->CurrentPosition();
      ERROR_ON(static_cast<int64_t>(start_) + sizeof(int64_t) > end);
      writer_->MoveAbsolute(start_);
      writer_->WriteInt64(static_cast<int64_t>(end) -
                          static_cast<int64_t>(start_) - sizeof(int64_t));
      writer_->MoveAbsolute(end);
      writer_ = nullptr;
    }
  }
  ~DeferredSizeWriter() { WriteSize(); }

 private:
  std::shared_ptr<StreamWriter> writer_;
  std::ios::streampos start_;
};

}  // namespace

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

LogContext::LogContext(const std::string& context) : saved_context_(context_) {
  UpdateContext(context);
}

void LogContext::UpdateContext(const std::string& new_context) {
  context_ = absl::StrCat(saved_context_, " ", new_context);
  if (InfoEnabled()) {
    std::cout << "[" << context_ << "]" << std::endl;
  }
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

TensorShape::TensorShape(DataType type, const Json::Value& array)
    : type_(type), metadata_size_(0) {
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
  out.WriteInt64(metadata_size_);
}

TensorShape::TensorShape(StreamReader& in)
    : type_(static_cast<DataType>(in.ReadInt64())),
      shape_(in.ReadInt64Array()),
      metadata_size_(in.ReadInt64()) {}

TensorShape::TensorShape(const std::vector<int64_t>& shape, DataType type)
    : shape_(shape), type_(type), metadata_size_(0) {}

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

void TensorInfo::SetMetadataSize(int64_t metadata_size) {
  shape_.SetMetadataSize(metadata_size);
}

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

std::string TensorInfo::ToString() const {
  return absl::StrCat("{ name = '", name_, "' type = '",
                      TensorTypeToString(type_), "' shape = '",
                      shape_.ToString(), "' handle='", handle_, "' }");
}

std::string TensorInfo::Filename() const {
  std::string data_filename = name_;
  std::replace(data_filename.begin(), data_filename.end(), '/', '_');
  return absl::StrCat(data_filename, ".data");
}

const TensorShape& TensorInfo::Shape() const { return shape_; }

bool TensorShape::operator==(const TensorShape& other) const {
  return type_ == other.type_ && shape_ == other.shape_;
}

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

void Tensor::LoadDataFromJson(const std::string& data_filename) {
  LogContext ctx(absl::StrCat("Loading ", TensorTypeToString(info_.Type()), " ",
                              info_.Name(), " from JSON file '", data_filename,
                              "'"));
  try {
    NDArrayParser parser;
    data_.clear();
    parser(LoadJsonFromFile(data_filename), info_.Shape(), data_);
  } catch (const std::out_of_range& error) {
    ERROR(error.what());
  }
}

void* Tensor::Data() {
  ERROR_ON(data_.size() == 0);
  ERROR_ON_MSG(data_.size() != info_.Shape().DataSizeInBytes(),
               "Buffer is of size " << data_.size() << " but the shape "
                                    << info_.Shape().ToString() << " requires "
                                    << info_.Shape().DataSizeInBytes());
  return data_.data();
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

std::fstream& StreamWriter::Stream() { return fd_; }

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
    : fd_(filename, std::ostream::binary | std::ostream::out) {
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

void StreamWriter::MoveAbsolute(std::ios::streampos position) {
  fd_.seekg(position);
}

std::ios::streampos StreamWriter::CurrentPosition() { return fd_.tellg(); }

StreamReader::StreamReader(const std::string& filename, bool is_versioned)
    : fd_(filename, std::ios::binary), filename_(filename) {
  ERROR_ON_MSG(!fd_.is_open(), "Failed to open file '" << filename << "'");
  fd_.exceptions(std::ifstream::eofbit | std::ifstream::failbit |
                 std::ifstream::badbit);
  std::streampos begin = fd_.tellg();
  fd_.seekg(0, std::ios::end);
  end_ = fd_.tellg();
  fd_.seekg(0, std::ios::beg);
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

void StreamReader::MoveRelative(std::ios::streamoff offset) {
  fd_.seekg(offset, std::ios::cur);
}

void StreamReader::MoveAbsolute(std::ios::streampos position) {
  fd_.seekg(position);
}

std::ios::streampos StreamReader::CurrentPosition() { return fd_.tellg(); }

std::ifstream& StreamReader::Stream() { return fd_; }

OutfeedStream::OutfeedStream(const TensorInfo& info) : info_(info), writer_() {}

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
      absl::StrCat(output_folder, "/", info_.Name(), ".bin");
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

Outfeed::Outfeed(const Json::Value& outfeed,
                 std::function<size_t(size_t)> metadata_size_fn)
    : name_(outfeed["name"].asString()) {
  absl::c_transform(outfeed["streams"], std::back_inserter(streams_),
                    [&](const Json::Value& stream) {
                      TensorInfo info{stream, TensorType::Outfeed};
                      if (metadata_size_fn) {
                        info.SetMetadataSize(
                            metadata_size_fn(info.Shape().DataSizeInBytes()));
                      }
                      return OutfeedStream{info};
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
  std::ios::streampos saved_pos = writer_->CurrentPosition();
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
  writer_->WriteInt64(static_cast<int64_t>(ObjectType::Feed));
  writer_->WriteString(name);
  DeferredSizeWriter object_size{writer_};
  info.ToStream(*writer_);
  writer_->WriteInt64(num_tensors);
  return FeedWriter{writer_, info.Shape().DataSizeInBytes(), num_tensors};
}

ExecutableWriter BinaryWriter::CreateExecutable(const std::string& name) {
  writer_->WriteInt64(static_cast<int64_t>(ObjectType::PoplarExecutable));
  writer_->WriteString(name);
  DeferredSizeWriter object_size{writer_};
  std::function<void()> on_write_complete =
      std::bind(&DeferredSizeWriter::WriteSize, object_size);
  return ExecutableWriter(writer_, on_write_complete);
}

void BinaryWriter::WriteMetadata(const std::string& name,
                                 const std::string& json_metadata) {
  writer_->WriteInt64(static_cast<int64_t>(ObjectType::PoplarMetadata));
  writer_->WriteString(name);
  DeferredSizeWriter object_size{writer_};
  writer_->WriteString(json_metadata);
}

void BinaryWriter::WriteTensor(const Tensor& tensor,
                               const std::string override_name) {
  writer_->WriteInt64(static_cast<int64_t>(ObjectType::Tensor));
  if (!override_name.empty()) {
    writer_->WriteString(override_name);
  } else {
    writer_->WriteString(tensor.Info().Name());
  }
  DeferredSizeWriter object_size{writer_};
  tensor.ToStream(*writer_);
}

void BinaryWriter::Close() {
  if (writer_) {
    writer_->Close();
    writer_ = nullptr;
  }
}

BinaryWriter::~BinaryWriter() { Close(); }

std::fstream& ExecutableWriter::Stream() { return writer_->Stream(); }

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
    : writer_(writer), on_write_complete_(on_write_complete) {}

}  // namespace ipu
