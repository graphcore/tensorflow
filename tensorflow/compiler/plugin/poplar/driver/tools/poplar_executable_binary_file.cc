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
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popef/Reader.hpp"
#include "popef/Types.hpp"
#include "popef/Writer.hpp"
#include "tensorflow/compiler/plugin/poplar/driver/tools/popef_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_executable_binary_file.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"

namespace xla {
namespace poplarplugin {

namespace {
StatusOr<popef::Anchor> ToPopEFAnchor(
    popef::TensorType type, const std::string& name, const std::string& handle,
    const xla::Shape& shape,
    const std::vector<popef::Anchor::ProgramsType>& programs) {
  popef::Anchor anchor;
  anchor.setName(name);
  anchor.setHandle(handle);
  std::vector<int64_t> popef_shape;
  ToPopEFShape(shape, popef_shape);
  anchor.tensorInfo().setShape(popef_shape);
  TF_ASSIGN_OR_RETURN(popef::DataType shape_type,
                      ToPopEFDataType(shape.element_type()));
  anchor.tensorInfo().setDataType(shape_type);
  anchor.setType(type);
  anchor.setPrograms(programs);
  return anchor;
}

template <class FeedInfos>
Status AddPopEFFeedAnchors(const FeedInfos& infos, popef::TensorType type,
                           popef::Metadata& metadata) {
  for (auto& info : infos) {
    auto add_shape = [&](int shape_idx, const xla::Shape& shape) -> Status {
      auto handle =
          (type == popef::TensorType::INPUT)
              ? GetInfeedCopyHandle(info.config.feed_id(), shape_idx)
              : GetOutfeedCopyHandle(info.config.feed_id(), shape_idx);
      const auto& programs = metadata.programFlow().main();
      TF_ASSIGN_OR_RETURN(popef::Anchor anchor,
                          ToPopEFAnchor(type, handle, handle, shape, programs));
      metadata.anchors().push_back(anchor);
      return Status::OK();
    };
    if (info.shape.IsTuple()) {
      auto shapes = &info.shape;
      if (info.shape.tuple_shapes_size() == 2 &&
          info.shape.tuple_shapes(0).IsTuple() &&
          info.shape.tuple_shapes(1).IsToken()) {
        shapes = &(info.shape.tuple_shapes(0));
      }
      for (int shape_idx = 0; shape_idx < shapes->tuple_shapes_size();
           shape_idx++) {
        TF_RETURN_IF_ERROR(
            add_shape(shape_idx, shapes->tuple_shapes(shape_idx)));
      }
    } else {
      TF_RETURN_IF_ERROR(add_shape(0, info.shape));
    }
  }
  return Status::OK();
}

void InsertPrograms(std::vector<popef::Anchor::ProgramsType>&) {}

template <typename... ProgSets>
void InsertPrograms(
    std::vector<popef::Anchor::ProgramsType>& programs,
    const std::vector<popef::ProgramFlow::ProgramIndexType>& progs,
    ProgSets&&... prog_sets) {
  programs.insert(programs.end(), progs.cbegin(), progs.cend());
  InsertPrograms(programs, std::forward<ProgSets>(prog_sets)...);
}

StatusOr<std::vector<popef::Anchor::ProgramsType>> GetPrograms(
    const popef::ProgramFlow& program_flow,
    const InputOutputAliasingMap::InputInfo::Type& type) {
  std::vector<popef::Anchor::ProgramsType> programs;
  switch (type) {
    case InputOutputAliasingMap::InputInfo::Type::StreamedVariable: {
      InsertPrograms(programs, program_flow.main());
    } break;
    case InputOutputAliasingMap::InputInfo::Type::ResourceModified: {
      InsertPrograms(programs, program_flow.load(), program_flow.main());
    } break;
    case InputOutputAliasingMap::InputInfo::Type::ResourceNotModified: {
      InsertPrograms(programs, program_flow.load());
    } break;
    default:
      return InvalidArgumentStrCat("Input info type #", type, " is not valid.");
  }
  return programs;
}

StatusOr<std::vector<popef::Anchor::ProgramsType>> GetPrograms(
    const popef::ProgramFlow& program_flow,
    const InputOutputAliasingMap::OutputInfo::Type& type) {
  std::vector<popef::Anchor::ProgramsType> programs;
  switch (type) {
    case InputOutputAliasingMap::OutputInfo::Type::StreamedVariable: {
      InsertPrograms(programs, program_flow.main());
    } break;
    case InputOutputAliasingMap::OutputInfo::Type::ResourceModified: {
      InsertPrograms(programs, program_flow.load(), program_flow.main());
    } break;
    case InputOutputAliasingMap::OutputInfo::Type::ResourceOutputOnly: {
      InsertPrograms(programs, program_flow.load());
    } break;
    default:
      return InvalidArgumentStrCat("Output info type #", type,
                                   " is not valid.");
  }
  return programs;
}

template <class FeedInfos>
Status AddPopEFEntryAnchors(const FeedInfos& infos, popef::TensorType type,
                            popef::Metadata& metadata) {
  for (auto& info : infos) {
    TF_ASSIGN_OR_RETURN(const std::vector<popef::Anchor::ProgramsType> programs,
                        GetPrograms(metadata.programFlow(), info.GetType()));
    TF_ASSIGN_OR_RETURN(
        popef::Anchor anchor,
        ToPopEFAnchor(type, UnmangleInputName(info.Name()), info.Handles()[0],
                      info.Shape(), programs));
    metadata.anchors().push_back(anchor);
  }
  return Status::OK();
}

void DeduplicatePopEFAnchors(popef::Metadata& metadata) {
  std::map<std::string, int> counts;
  for (auto& anchor : metadata.anchors()) {
    counts[anchor.name()]++;
  }
  for (auto it = std::begin(counts); it != std::end(counts);) {
    if (it->second < 2) {
      it = counts.erase(it);
      continue;
    }
    VLOG(1) << "Warning: duplicate anchor name " << it->first
            << " will have numeric suffixes added in the exported executable.";
    it->second = 0;
    ++it;
  }
  for (auto& anchor : metadata.anchors()) {
    auto name = anchor.name();
    if (counts.find(name) == std::end(counts)) continue;
    anchor.setName(name + std::to_string(counts[name]));
    counts[name]++;
  }
}

StatusOr<popef::Metadata> ToPopEFMetadata(const std::string& executable_name,
                                          const PoplarExecutableInfo& info,
                                          const InputOutputAliasingMap& io_map,
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

  using ProgIndexType = popef::ProgramFlow::ProgramIndexType;
  static constexpr ProgIndexType load_prog_idx = 0, main_prog_idx = 1,
                                 save_prog_idx = 2;

  auto& flow = metadata.programFlow();
  flow.load().emplace_back(load_prog_idx);
  flow.main().emplace_back(main_prog_idx);
  flow.save().emplace_back(save_prog_idx);

  TF_RETURN_IF_ERROR(AddPopEFFeedAnchors(info.infeed_infos,
                                         popef::TensorType::INPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFFeedAnchors(info.outfeed_infos,
                                         popef::TensorType::OUTPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFEntryAnchors(io_map.GetEntryInputInfos(),
                                          popef::TensorType::INPUT, metadata));
  TF_RETURN_IF_ERROR(AddPopEFEntryAnchors(io_map.GetEntryOutputInfos(),
                                          popef::TensorType::OUTPUT, metadata));

  static const std::unordered_map<ProgIndexType, std::string> programs_map = {
      {load_prog_idx, "load_program"},
      {main_prog_idx, "main_program"},
      {save_prog_idx, "save_program"}};
  metadata.setProgramsMap(programs_map);

  DeduplicatePopEFAnchors(metadata);
  return metadata;
}
}  // namespace

Status PoplarExecutableBinaryFile::Write(
    const std::string& file_name,
    const ::tensorflow::protobuf::MessageLite& proto,
    const PoplarExecutableInfo& info, const InputOutputAliasingMap& io_map,
    const poplar::OptionFlags& opts,
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
    popef_exe = popef_writer->createExecutable(exec_hash, /*compress=*/false);
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
                      ToPopEFMetadata(exec_hash, info, io_map, opts));
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
