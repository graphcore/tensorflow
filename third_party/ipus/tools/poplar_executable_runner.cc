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

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

#include "ipu/poplar_executable_data.h"

namespace ipu {

namespace {

std::string FirstLinesOf(const std::string& lines, int64_t num_lines) {
  std::stringstream ss(lines);
  std::string line;
  std::string res;
  while (num_lines-- > 0 && std::getline(ss, line, '\n')) {
    res += absl::StrCat(line, "\n");
  }
  return res;
}

enum PoplarProgramType {
  HOST_TO_DEVICE,
  MAIN_SEQUENCE,
  DEVICE_TO_HOST,
};

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

poplar::OptionFlags ParseOptionFlags(
    const std::map<std::string, std::string>& str_options) {
  poplar::OptionFlags opts;
  for (auto option : str_options) {
    opts.set(option.first, option.second);
  }
  return opts;
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

class SubStream : public std::streambuf {
 public:
  SubStream(StreamReader& reader, int64_t sizeInBytes)
      : reader_(reader),
        start_(reader.CurrentPosition()),
        sizeInBytes_(sizeInBytes),
        current_(0) {}

 protected:
  int underflow() override {
    if (current_ + std::streamsize(1) >= sizeInBytes_) {
      return traits_type::eof();
    }
    return reader_.Stream().rdbuf()->sgetc();
  }

  int uflow() override {
    if (current_ + std::streamsize(1) > sizeInBytes_) {
      return traits_type::eof();
    }
    current_ += std::streamsize(1);
    return reader_.Stream().rdbuf()->sbumpc();
  }

  std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
                         std::ios_base::openmode which =
                             std::ios_base::in | std::ios_base::out) override {
    std::streampos cursor;

    if (way == std::ios_base::beg) {
      cursor = off;
    } else if (way == std::ios_base::cur) {
      cursor = current_ + off;
    } else if (way == std::ios_base::end) {
      cursor = sizeInBytes_ - off;
    }

    if (cursor < 0 || cursor >= sizeInBytes_) {
      return std::streampos(-1);
    }
    current_ = cursor;
    if (reader_.Stream().rdbuf()->pubseekpos(start_ + current_) ==
        std::streampos(-1)) {
      return std::streampos(-1);
    }

    return current_;
  }

  std::streampos seekpos(std::streampos sp,
                         std::ios_base::openmode which =
                             std::ios_base::in | std::ios_base::out) override {
    if (sp < 0 || sp >= sizeInBytes_) {
      return std::streampos(-1);
    }
    current_ = sp;
    if (reader_.Stream().rdbuf()->pubseekpos(start_ + current_) ==
        std::streampos(-1)) {
      return std::streampos(-1);
    }
    return current_;
  }

 private:
  StreamReader& reader_;
  std::ios::streampos start_;
  std::streamsize sizeInBytes_;
  std::ios::streampos current_;
};

void WriteJsonToStream(const Json::Value& root, std::ostream* sout) {
  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";

  std::unique_ptr<Json::StreamWriter> writer(json_builder.newStreamWriter());
  writer->write(root, sout);
}

/* InsertNumberBeforeExtension("MyFile.txt", 1) -> "MyFile.1.txt"
 */
std::string InsertNumberBeforeExtension(const std::string& filename,
                                        int64_t number) {
  size_t dot_pos = filename.rfind(".");
  ERROR_ON_MSG(dot_pos == std::string::npos, "Invalid filename: no extension");
  std::string basename = filename.substr(0, dot_pos);
  std::string extension = filename.substr(dot_pos);
  return absl::StrCat(basename, ".", number, extension);
}

}  // namespace

IExecutable::IExecutable(StreamReader& stream, const Metadata& meta,
                         int64_t length) {
  try {
    SubStream sub(stream, length > 0 ? length : stream.NumBytesLeft());
    std::istream sub_stream(&sub);
    poplar::Executable poplar_executable =
        poplar::Executable::deserialize(sub_stream);
    engine_.reset(new poplar::Engine(std::move(poplar_executable),
                                     ParseOptionFlags(meta.engine_options)));
  } catch (const std::exception& e) {
    ERROR("Failed to deserialize " << stream.Filename() << " : " << e.what());
  }
}

poplar::Engine& IExecutable::Engine() { return *engine_; }

std::string IExecutable::StreamsList(bool summary) const {
  std::stringstream ss;
  if (summary) {
    std::map<std::string, std::set<int>> read_streams, write_streams;
    std::smatch m;
    const std::regex re{"(.*?)([0-9]+)\\.0([-+])"};
    std::list<std::string> other_inputs, other_outputs;
    for (auto stream : engine_->listStreams()) {
      if (std::regex_match(stream, m, re)) {
        ERROR_ON(m.size() != 4);
        const std::string prefix = m[1].str();
        const int idx = atoi(m[2].str().c_str());
        const bool write = (m[3].str() == "-");

        auto& group = write ? write_streams : read_streams;
        auto& indices = group[prefix];
        ERROR_ON(indices.find(idx) != indices.end());
        indices.emplace(idx);
      } else {
        if (stream[stream.size() - 1] == '-') {
          other_outputs.push_back(stream);
        } else {
          other_inputs.push_back(stream);
        }
      }
    }
    ss << "Input streams:\n";
    for (auto pair : read_streams) {
      ERROR_ON(*absl::c_min_element(pair.second) != 0);
      int num_indices = *absl::c_max_element(pair.second);
      ERROR_ON(num_indices + 1 != pair.second.size());
      ss << "  " << pair.first << "0.0+ --> " << pair.first << num_indices
         << ".0+\n";
    }
    for (const auto& stream : other_inputs) {
      ss << "  " << stream << std::endl;
    }
    ss << "\nOutput streams:\n";
    for (auto pair : write_streams) {
      ERROR_ON(*absl::c_min_element(pair.second) != 0);
      int num_indices = *absl::c_max_element(pair.second);
      ERROR_ON(num_indices + 1 != pair.second.size());
      ss << "  " << pair.first << "0.0- --> " << pair.first << num_indices
         << ".0-\n";
    }
    for (const auto& stream : other_outputs) {
      ss << "  " << stream << std::endl;
    }
  } else {
    int idx = 0;
    for (const auto& stream : engine_->listStreams()) {
      ss << "[" << idx++ << "] " << stream << std::endl;
    }
  }
  return ss.str();
}

Executable::Executable(StreamReader& stream, const Metadata& meta,
                       int64_t length)
    : IExecutable(stream, meta, length) {}

void Executable::Load(const poplar::Device& device) {
  LogContext ctx("Running HOST_TO_DEVICE");
  std::cout << "Loading program onto the device\n";
  engine_->load(device);
  try {
    engine_->run(PoplarProgramType::HOST_TO_DEVICE);
  } catch (const poplar::poplar_error& err) {
    PRINT_INFO(err.what());
    ERROR(FirstLinesOf(err.what(), 3));
  }
}

void Executable::Run() {
  LogContext ctx("Running MAIN_SEQUENCE");
  try {
    engine_->run(PoplarProgramType::MAIN_SEQUENCE);
  } catch (const poplar::poplar_error& err) {
    PRINT_INFO(err.what());
    ERROR(FirstLinesOf(err.what(), 3));
  }
}

void Executable::DeviceToHostCopy() {
  LogContext ctx("Running DEVICE_TO_HOST");
  try {
    engine_->run(PoplarProgramType::DEVICE_TO_HOST);
  } catch (const poplar::poplar_error& err) {
    PRINT_INFO(err.what());
    ERROR(FirstLinesOf(err.what(), 3));
  }
}

VerifiedExecutable::VerifiedExecutable(StreamReader& stream,
                                       const Metadata& meta, int64_t length,
                                       bool is_verified)
    : IExecutable(stream, meta, length), is_verified_(is_verified) {}

bool VerifiedExecutable::IsVerified() const { return is_verified_; }

void VerifiedExecutable::Prepare(poplar::Device& device) {
  engine_->prepare(device);
}

void VerifiedExecutable::Deploy() { engine_->deploy(); }

void VerifiedExecutable::Run() {
  LogContext ctx("Running FUSED_SEQUENCE");
  try {
    engine_->run(0);
  } catch (const poplar::poplar_error& err) {
    PRINT_INFO(err.what());
    ERROR(FirstLinesOf(err.what(), 3));
  }
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

poplar::Device DeviceManager::GetSpecificDevice(int64_t device_id,
                                                const Metadata& meta) {
  LogContext ctx{"DeviceManager::GetSpecificDevice"};
  poplar::Device device =
      manager_.getDevice(device_id, ParseOptionFlags(meta.device_options));

  ERROR_ON_MSG(!device.attach(),
               "Failed to attach to device "
                   << device_id << " OptionFlags="
                   << absl::StrJoin(meta.device_options, ", ",
                                    absl::PairFormatter("=")));
  unsigned mj, mn, pt;
  device.getDriverVersion(mj, mn, pt);
  const auto& ids = device.getDriverIDs();
  std::cout << "Poplar driver: " << mj << "." << mn << "." << pt << std::endl;
  std::cout << "Successfully attached to IPU" << (ids.size() > 1 ? "s" : "")
            << ": " << absl::StrJoin(ids, ",") << std::endl;
  return std::move(device);
}

poplar::Device DeviceManager::GetDevice(int64_t num_ipus,
                                        const Metadata& meta) {
  LogContext ctx{"DeviceManager::GetDevice"};
  auto device_list = manager_.getDevices(poplar::TargetType::IPU, num_ipus,
                                         ParseOptionFlags(meta.device_options));
  ERROR_ON_MSG(
      device_list.empty(),
      "Failed to find any IPU device that match the requested config: num_ipus="
          << num_ipus << " OptionFlags="
          << absl::StrJoin(meta.device_options, ", ",
                           absl::PairFormatter("=")));

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

TensorManager::TensorManager(const Metadata& meta) {
  for (auto input : meta.inputs) {
    inputs_.emplace_back(input);
  }
  for (auto output : meta.outputs) {
    outputs_.emplace_back(output);
  }
  for (auto infeed : meta.infeeds) {
    infeeds_.emplace_back(infeed);
  }
  for (auto outfeed : meta.outfeeds) {
    outfeeds_.emplace_back(outfeed);
  }
  feeds_order_ = meta.feeds_order;
  seeds_.resize(meta.replication_count);
  std::mt19937_64 seed_generator;
  for (auto& seed : seeds_) {
    seed = seed_generator();
  }
  num_ipus_ = meta.num_ipus;
  replication_count_ = meta.replication_count;
  random_number_seed_handle_ = meta.random_number_seed_handle;
}

int64_t TensorManager::NumIpus() const { return num_ipus_; }

void TensorManager::LoadCheckpointMetadataFromJson(
    const std::string& filename) {
  LogContext ctx(
      absl::StrCat("Loading checkpoint metadata from JSON file ", filename));
  Json::Value root = LoadJsonFromFile(filename);
  const Json::Value& infeed_positions = root["infeeds"];
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
  ERROR_ON_MSG(!out.is_open(), "Failed to open file '" << filename << "'");
  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
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

const std::vector<Tensor>& TensorManager::Inputs() const { return inputs_; }
const std::vector<Tensor>& TensorManager::Outputs() const { return outputs_; }
const std::vector<Infeed>& TensorManager::Infeeds() const { return infeeds_; }
std::vector<Infeed>& TensorManager::MutableInfeeds() { return infeeds_; }

void TensorManager::LoadInfeeds(const BinaryLoader& loader) {
  for (auto& infeed : infeeds_) {
    ipu::LogContext ctx{absl::StrCat("Loading infeed '", infeed.Name(), "'")};
    infeed.InitializeDataSources(loader);
  }
}

void TensorManager::AssertAllTensorsProvided(const BinaryLoader& loader) {
  std::list<std::string> missing_msg;
  std::set<std::string> tensors_provided =
      loader.GetObjectNames(ObjectType::Tensor);
  std::set<std::string> feeds_provided =
      loader.GetObjectNames(ObjectType::Feed);
  std::string errors_msg;
  for (auto& input : inputs_) {
    if (input.Info().Name() == Metadata::InputCheckpointIndexName()) {
      continue;
    }
    if (tensors_provided.find(input.Info().Name()) == tensors_provided.end()) {
      missing_msg.push_back(absl::StrCat(
          "No data provided for ", TensorTypeToString(input.Info().Type()),
          " '", input.Info().Name(), "'"));
    }
  }
  for (auto& infeed : infeeds_) {
    for (int i = 0; i < infeed.Streams().size(); i++) {
      const std::string name = infeed.Streams()[i].Info().Name();
      if (feeds_provided.find(name) == feeds_provided.end()) {
        missing_msg.push_back(
            absl::StrCat("No data provided for infeed's stream '", name, "'"));
      }
    }
  }
  ERROR_ON_MSG(!missing_msg.empty(), absl::StrJoin(missing_msg, "\n"));
}

void TensorManager::LoadInputsAndParameters(const BinaryLoader& loader) {
  for (auto& input : inputs_) {
    LogContext ctx(absl::StrCat("Loading ",
                                TensorTypeToString(input.Info().Type()), " '",
                                input.Info().Name(), "'"));
    if (loader.ContainsObject(ObjectType::Tensor, input.Info().Name())) {
      std::unique_ptr<StreamReader> in =
          loader.GetTensorStream(input.Info().Name());
      input.LoadDataFromStream(*in);
    }
  }
}

void TensorManager::SaveOutputs(TensorType type, BinaryWriter& writer,
                                bool allow_duplicates) const {
  std::map<std::string, int> duplicates;
  ERROR_ON(type != TensorType::ParameterOut && type != TensorType::OutputData);
  absl::c_for_each(outputs_, [allow_duplicates, type, &writer,
                              &duplicates](const Tensor& out) {
    if (out.Info().Type() == type) {
      // Checkpoints will be saved separately.
      if (out.Info().Name() == Metadata::CheckpointName() ||
          out.Info().Name() == Metadata::ClearCheckpointName()) {
        return;
      }
      auto& occurrences = duplicates[out.Info().Name()];
      std::string override_name;
      if (occurrences > 0) {
        ERROR_ON_MSG(!allow_duplicates, "More than one output tensor is named '"
                                            << out.Info().Name() << "'");
        override_name = absl::StrCat(out.Info().Name(), ".", occurrences);
        std::cout << "WARNING: Renaming output named '" << out.Info().Name()
                  << "' to '" << override_name << "' to avoid conflict.\n";
      }
      writer.WriteTensor(out, override_name);
      occurrences++;
    }
  });
}

void TensorManager::SaveOutputsToJson(TensorType type,
                                      const std::string& folder) const {
  std::map<std::string, int> duplicates;
  ERROR_ON(type != TensorType::ParameterOut && type != TensorType::OutputData);
  absl::c_for_each(outputs_, [folder, type, &duplicates](const Tensor& out) {
    if (out.Info().Type() == type) {
      if (out.Info().Shape().HasMetadata()) {
        PRINT_INFO("Can't save " << out.Info().Name()
                                 << " as JSON because it has metadata that"
                                    " can't be saved.");
        return;
      }
      // Checkpoints will be saved separately.
      if (out.Info().Name() == Metadata::CheckpointName() ||
          out.Info().Name() == Metadata::ClearCheckpointName()) {
        return;
      }
      auto& occurrences = duplicates[out.Info().Name()];

      std::string filename =
          absl::StrCat(SanitizeName(out.Info().Name()), ".json");
      if (occurrences > 0) {
        filename = InsertNumberBeforeExtension(filename, occurrences);
      }
      out.SaveDataToJsonFile(absl::StrCat(folder, "/", filename));
      occurrences++;
    }
  });
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

bool TensorManager::ContainsCheckpoint() const { return !infeeds_.empty(); }

void TensorManager::SaveVerifiedCheckpoint(BinaryWriter& writer) {
  ERROR_ON(!ContainsCheckpoint());
  absl::c_for_each(outputs_, [&writer](const Tensor& out) {
    if (out.Info().Name() == Metadata::CheckpointName() ||
        out.Info().Name() == Metadata::ClearCheckpointName()) {
      writer.WriteTensor(out);
    }
  });
}

void TensorManager::LoadVerifiedCheckpoint(const BinaryLoader& loader,
                                           int64_t checkpoint_index) {
  LogContext ctx("LoadVerifiedCheckpoint");
  ERROR_ON(!ContainsCheckpoint());

  std::unique_ptr<ipu::StreamReader> reader =
      loader.GetTensorStream(Metadata::ClearCheckpointName());
  ipu::Tensor feeds_positions{*reader};
  const int64_t* positions = reinterpret_cast<int64_t*>(feeds_positions.Data());

  for (auto& feed : infeeds_) {
    int feed_idx = -1;
    for (int i = 0; i < feeds_order_.size(); i++) {
      if (feed.Name() == feeds_order_[i]) {
        feed_idx = i;
        break;
      }
    }
    ERROR_ON_MSG(feed_idx < 0,
                 "Can't find index position for infeed " << feed.Name());
    for (auto& stream : feed.MutableStreams()) {
      stream.JumpToTensor(positions[feed_idx]);
    }
  }
  for (auto& input : inputs_) {
    if (input.Info().Handle() == Metadata::InputCheckpointIndexHandle()) {
      reinterpret_cast<int64_t*>(input.Data())[0] = checkpoint_index;
      return;
    }
  }
  ERROR("Couldn't find checkpointIndex in the list of inputs");
}

void TensorManager::ConnectStreams(IExecutable& executable) {
  auto& engine = executable.Engine();

  for (auto& input : inputs_) {
    PRINT_INFO("Connecting " << input.Info().Name() << " to handle "
                             << input.Info().Handle());
    engine.connectStream(input.Info().Handle(), input.Data());
  }

  for (auto& output : outputs_) {
    for (int replica_id = 0; replica_id < replication_count_; ++replica_id) {
      auto callback = [&output, replica_id](void* ptr) {
        if (replica_id == 0) {
          std::memcpy(output.Data(), ptr,
                      output.Info().Shape().DataSizeInBytes());
        }
      };
      PRINT_INFO("Connecting " << output.Info().Name() << " to handle "
                               << output.Info().Handle());
      engine.connectStreamToCallback(output.Info().Handle(), replica_id,
                                     callback);
    }
  }

  for (auto& infeed : infeeds_) {
    for (auto& infeed_stream : infeed.MutableStreams()) {
      for (int replica_id = 0; replica_id < replication_count_; ++replica_id) {
        auto callback = absl::make_unique<InfeedCallback>(infeed_stream);
        PRINT_INFO("Connecting " << infeed_stream.Info().Name() << " to handle "
                                 << infeed_stream.Info().Handle());
        engine.connectStreamToCallback(infeed_stream.Info().Handle(),
                                       replica_id, std::move(callback));
      }
    }
  }
  for (auto& outfeed : outfeeds_) {
    for (auto& outfeed_stream : outfeed.Streams()) {
      for (int replica_id = 0; replica_id < replication_count_; ++replica_id) {
        PRINT_INFO("Connecting " << outfeed_stream.Info().Name()
                                 << " to handle "
                                 << outfeed_stream.Info().Handle());
        engine.connectStreamToCallback(
            outfeed_stream.Info().Handle(), replica_id,
            [&outfeed_stream, this](void* src) {
              outfeed_stream.WriteTensor(src, this->replication_count_);
            });
      }
    }
  }
  if (!random_number_seed_handle_.empty()) {
    for (int replica_id = 0; replica_id < replication_count_; ++replica_id) {
      auto callback = [this, replica_id](void* ptr) mutable {
        reinterpret_cast<uint64_t*>(ptr)[0] = this->seeds_[replica_id];
      };

      engine.connectStreamToCallback(random_number_seed_handle_, replica_id,
                                     callback);
    }
  }
}

std::vector<InfeedStream>& Infeed::MutableStreams() { return streams_; }
const std::vector<InfeedStream>& Infeed::Streams() const { return streams_; }

const std::string& Infeed::Name() const { return name_; }

Infeed::Infeed(const FeedInfo& infeed) : name_(infeed.name) {
  for (auto stream : infeed.streams) {
    streams_.emplace_back(stream);
  }
}

void Infeed::InitializeDataSources(const BinaryLoader& loader) {
  for (int i = 0; i < streams_.size(); i++) {
    const std::string stream_name = streams_[i].Info().Name();
    LogContext ctx{stream_name};
    streams_[i].InitializeDataSource(
        loader.CreateInfeedStreamReader(stream_name));
  }
  ERROR_ON(!absl::c_all_of(streams_, [this](const InfeedStream& stream) {
    return stream.NumTensors() == this->streams_[0].NumTensors();
  }));
}

std::unique_ptr<TensorManager> BinaryLoader::CreateTensorManager(
    const std::string& metadata_name) const {
  LogContext ctx{"BinaryLoader::CreateTensorManager " + metadata_name};
  auto metadata = ReadMetadata(metadata_name);
  return absl::make_unique<TensorManager>(*metadata);
}

std::unique_ptr<Executable> BinaryLoader::CreateExecutable(
    const std::string& executable_name) const {
  LogContext ctx{"BinaryLoader::CreateExecutable " + executable_name};
  auto in = CreateExecutableReader(executable_name);
  auto metadata = ReadMetadata(executable_name);
  bool is_verified = static_cast<bool>(in->ReadInt64());
  ERROR_ON_MSG(is_verified, "Regular Executables cannot be verified");
  return absl::make_unique<Executable>(*in, *metadata, in->NumBytesLeft());
}

std::unique_ptr<VerifiedExecutable> BinaryLoader::CreateVerifiedExecutable(
    const std::string& executable_name) const {
  LogContext ctx{"BinaryLoader::CreateVerifiedExecutable " + executable_name};
  auto in = CreateExecutableReader(executable_name);
  auto metadata = ReadMetadata(executable_name);
  bool is_verified = static_cast<bool>(in->ReadInt64());
  return absl::make_unique<VerifiedExecutable>(*in, *metadata,
                                               in->NumBytesLeft(), is_verified);
}

}  // namespace ipu
