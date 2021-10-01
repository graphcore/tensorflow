/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "popef/Reader.hpp"
#include "popef/Writer.hpp"
#include "tensorflow/compiler/plugin/poplar/driver/tools/popef_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_executable_binary_file.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

namespace xla {
namespace poplarplugin {

template <class FeedInfos>
static Status AddPopEFAnchors(const FeedInfos& infos, popef::TensorType type,
                              popef::Metadata& metadata) {
  for (auto& info : infos) {
    popef::Anchor anchor;
    anchor.setName(UnmangleInputName(info.name));
    anchor.setHandle(info.handle);
    std::vector<int64_t> shape;
    ToPopEFShape(info.shape, shape);
    anchor.tensorInfo().setShape(shape);
    TF_ASSIGN_OR_RETURN(popef::DataType shape_type,
                        ToPopEFDataType(info.shape.element_type()));
    anchor.tensorInfo().setDataType(shape_type);
    anchor.setType(type);
    metadata.anchors().push_back(anchor);
  }
  return Status::OK();
}

static StatusOr<popef::Metadata> ToPopEFMetadata(
    const std::string& executable_name, const PoplarExecutableInfo& info,
    const poplar::OptionFlags& opts) {
  if (info.target_type != "IPU") {
    return tensorflow::errors::Internal("[PopEF][ToMetadata]: ",
                                        "Target type must be IPU");
  }
  if (info.target_arch.find("ipu") != 0) {
    return tensorflow::errors::Internal("[PopEF][ToMetadata]: ",
                                        "Target arch must begin with ipu");
  }
  // Extract IPU version
  auto ipu_postfix = info.target_arch.substr(3);
  int ipu_version;
  try {
    ipu_version = std::stoi(ipu_postfix);
  } catch (const std::exception& e) {
    return tensorflow::errors::Internal(
        "[PopEF][ToMetadata]: ",
        "Target arch must be of form ipux, where x is an integer");
  }
  popef::Metadata metadata;
  metadata.setReplicationFactor(info.replication_factor);
  metadata.setNumIpus(info.num_IPUs);
  metadata.setSeedHandle(GetRandomNumberSeedStream());
  metadata.setExecutable(executable_name);
  metadata.setNumProcesses(1);
  metadata.setIpuVersion(ipu_version);
  // Transfer engine options
  for (auto& opt : opts) {
    metadata.engineOptions().emplace_back(opt.first, opt.second);
  }
  auto& flow = metadata.programFlow();
  flow.load().emplace_back(0);
  flow.main().emplace_back(1);
  flow.save().emplace_back(2);

  TF_RETURN_IF_ERROR(AddPopEFAnchors(info.feed_input_infos,
                                     popef::TensorType::INPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFAnchors(info.feed_output_infos,
                                     popef::TensorType::OUTPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFAnchors(info.entry_input_infos,
                                     popef::TensorType::INPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFAnchors(info.entry_output_infos,
                                     popef::TensorType::OUTPUT, metadata));
  return metadata;
}

Status PoplarExecutableBinaryFile::Write(
    const std::string& file_name,
    const ::tensorflow::protobuf::MessageLite& proto,
    const PoplarExecutableInfo& info, const poplar::OptionFlags& opts,
    std::function<void(std::ostream&)> serialize_executable,
    const std::string& executable_hash) {
  auto file = std::ofstream(file_name, std::ios::binary);
  if (!file) {
    return InternalErrorStrCat("Failed to open file for writing: ", file_name);
  }
  std::string exec_hash = executable_hash;
  if (exec_hash.empty()) {
    exec_hash = "UNHASHED_EXECUTABLE";
  }
  std::unique_ptr<popef::Writer> popef_writer;
  std::shared_ptr<popef::BlobWriter> popef_opaque;
  std::shared_ptr<popef::BlobWriter> popef_exe;

  try {
    popef_writer = absl::make_unique<popef::Writer>(file);
    popef_opaque = popef_writer->createOpaqueBlob("tensorflow", exec_hash);
  } catch (const std::exception& e) {
    return PopEFExceptionToTensorflowStatus("[WriterSetup]", e);
  }

  std::string serialized;
  proto.AppendToString(&serialized);

  try {
    // Append the protobuf metadata.
    popef_opaque->stream << serialized;
    popef_exe = popef_writer->createExecutable(exec_hash);
  } catch (const std::exception& e) {
    return PopEFExceptionToTensorflowStatus("[WriteOpaque]", e);
  }
  // Append the Poplar executable.
  try {
    serialize_executable(popef_exe->stream);
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Serialize]", e);
  }

  TF_ASSIGN_OR_RETURN(popef::Metadata metadata,
                      ToPopEFMetadata(exec_hash, info, opts));
  try {
    popef_writer->write(metadata);
  } catch (const std::exception& e) {
    return PopEFExceptionToTensorflowStatus("[WriteMetadata]", e);
  }
  return Status::OK();
}

StatusOr<poplar::Executable> PoplarExecutableBinaryFile::Read(
    const std::string& file_name, ::tensorflow::protobuf::MessageLite* proto,
    const std::string& executable_hash) {
  std::unique_ptr<popef::Reader> popef_reader;

  try {
    popef_reader = absl::make_unique<popef::Reader>();
    popef_reader->parseFile(file_name);
  } catch (const std::exception& e) {
    return PopEFExceptionToTensorflowStatus("[ReaderSetup]", e);
  }
  const std::string error_prefix =
      absl::StrCat("[Deserialize][File: ", file_name, "]");

  bool found_opaque = false;
  for (auto opaque : popef_reader->opaqueBlobs()) {
    if (opaque.name != "tensorflow") {
      continue;
    }
    if (!executable_hash.empty() && opaque.executable != executable_hash) {
      continue;
    }
    auto& opq_stream = opaque.data;
    // Find the protobuf metadata length.
    opq_stream.seekg(0, std::ios::end);
    auto metadata_length = opq_stream.tellg();
    opq_stream.seekg(0, std::ios::beg);

    // Read the protobuf metadata.
    std::vector<char> serialized(metadata_length);
    opq_stream.read(serialized.data(), metadata_length);
    if (opq_stream.rdstate() != std::ios_base::goodbit) {
      return InternalErrorStrCat(error_prefix,
                                 " Corrupted - Cannot read the metadata.");
    }
    if (!proto->ParseFromArray(serialized.data(), metadata_length)) {
      return InternalErrorStrCat(error_prefix,
                                 " Corrupted - Cannot parse the metadata.");
    }
    serialized.clear();
    found_opaque = true;
    break;
  }
  if (!found_opaque) {
    return InternalErrorStrCat(error_prefix, "No matching opaque blob found.");
  }

  // Read the executable.
  for (auto executable : popef_reader->executables()) {
    if (!executable_hash.empty() && executable.name != executable_hash) {
      continue;
    }
    try {
      auto exe = executable.getStandaloneExecutableStream();
      return poplar::Executable::deserialize(std::move(exe));
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus(error_prefix, e);
    }
  }
  return InternalErrorStrCat(error_prefix, "No matching executable found.");
}

}  // namespace poplarplugin
}  // namespace xla
