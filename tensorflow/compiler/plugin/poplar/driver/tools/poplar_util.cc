/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <regex>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

poplar::Graph& GetMasterGraph(CompilerResources& res) {
  return *res.main_graph;
}

uint64 GetShardForOutputIndex(const HloInstruction* inst,
                              int flattened_output_tuple_index) {
  if (inst->has_sharding()) {
    const auto& sharding = GetShardingDeviceIdVector(inst->sharding());
    if (flattened_output_tuple_index >= static_cast<int>(sharding.size())) {
      LOG(FATAL) << "Sharding index " << flattened_output_tuple_index
                 << " out of range on " << inst->ToString();
    }

    return sharding[flattened_output_tuple_index];
  }

  return 0;
}

poplar::Graph& GetGraphWithOutputIndex(CompilerResources& res,
                                       const HloInstruction* inst,
                                       int flattened_output_tuple_index) {
  if (inst->has_sharding()) {
    int device_id = GetShardForOutputIndex(inst, flattened_output_tuple_index);

    if (device_id >= static_cast<int>(res.shard_graphs.size())) {
      LOG(FATAL) << "Graph index " << device_id << " out of range on "
                 << inst->ToString();
    }

    return res.shard_graphs[device_id];
  }

  return GetMasterGraph(res);
}

poplar::Graph& GetGraph(CompilerResources& res, const HloInstruction* inst) {
  return GetGraphWithOutputIndex(res, inst, 0);
}

template <typename TYPE>
static void SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                           const Literal& literal) {
  const TYPE* value(static_cast<const TYPE*>(literal.untyped_data()));
  graph.setInitialValue<TYPE>(field, *value);
}

static void SetFp16VertexField(poplar::Graph& graph,
                               const poplar::FieldRef& field,
                               const Literal& literal) {
  const uint16_t* value(static_cast<const uint16_t*>(literal.untyped_data()));
  graph.setInitialValueHalf(field, *value);
}

Status SetVertexField(poplar::Graph& graph, const poplar::FieldRef& field,
                      const Literal& literal) {
  switch (literal.shape().element_type()) {
    case PRED:
      SetVertexField<bool>(graph, field, literal);
      break;
    case S32:
      SetVertexField<unsigned>(graph, field, literal);
      break;
    case U32:
      SetVertexField<int>(graph, field, literal);
      break;
    case F16:
      SetFp16VertexField(graph, field, literal);
      break;
    case F32:
      SetVertexField<float>(graph, field, literal);
      break;
    default:
      return xla::FailedPrecondition("Unrecognised type in SetVertexField: %d",
                                     literal.shape().element_type());
  }
  return Status::OK();
}

Status PoplarExceptionToTensorflowStatus(const std::string& origin,
                                         const std::exception& e) {
  const std::string prefix = "[Error]" + origin;
  /* NOTE: Reduce this list if/when Poplar errors are subclassed */
  try {
    std::rethrow_exception(std::current_exception());
  } catch (const poplar::file_load_error& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::missing_cycle_estimate& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::symbol_error& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::unknown_field& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::unknown_vertex_type& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::no_environment& e) {
    return tensorflow::errors::NotFound(prefix, e.what());
  } catch (const poplar::parse_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_option& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_machine_model& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::stream_connection_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::graph_cycle_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::invalid_tile_mapping& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::type_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::no_size_specified& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::profiling_disabled& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::control_program_error& e) {
    return tensorflow::errors::InvalidArgument(prefix, e.what());
  } catch (const poplar::runtime_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::overflow_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::tensor_io_state_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_connection_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_object_load_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_object_creation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_program_compilation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poputil::poplibs_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::link_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::stream_memory_allocation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::graph_memory_allocation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::tensor_creation_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::memory_elem_constraints_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const poplar::index_error& e) {
    return tensorflow::errors::OutOfRange(prefix, e.what());
  } catch (const poplar::poplar_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
  } catch (const std::exception& e) {
  }

  return tensorflow::errors::Unknown(prefix, e.what());
}

void SetFlagIfNotPresent(poplar::OptionFlags& opts, const std::string& key,
                         const std::string& value) {
  for (const auto& opt : opts) {
    if (opt.first == key) {
      return;
    }
  }
  opts.set(key, value);
}

poplar::OptionFlags GetReplicateAllReduceOptions() {
  poplar::OptionFlags options;
  options.set("useReplicatedImplementation", "true");
  return options;
}

void DumpIfPoplarOutOfMemoryAllocationException(
    const PoplarExecutor* poplarExecutor, const std::string& module_name,
    const poplar::graph_memory_allocation_error& p_e) {
  std::string report_directory = poplarExecutor->ReportDirectory();
  if (report_directory.empty()) {
    report_directory = ".";
  }

  std::string dump_filename = tensorflow::io::JoinPath(
      report_directory, "/GC_TensorFlow_" + module_name);

  if (p_e.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
      p_e.graphProfile.size() != 0) {
    auto opts = poplarExecutor->GetReportGraphFlags();
    SetFlagIfNotPresent(opts, "showVarStorage", "true");

    // Always produce a text report
    std::ofstream stream(dump_filename + ".txt");
    if (!stream) {
      LOG(WARNING) << "Unable to open file " << dump_filename << ".txt"
                   << ", the profiler summary will not be saved.";
    } else {
      poplar::printGraphSummary(stream, p_e.graphProfile, opts);
      LOG(INFO) << "Out of memory summary saved to " << dump_filename
                << ".txt.";
    }

    if (!poplarExecutor->CompilerReportingTextFormat()) {
      // Produce binary file
      if (poplarExecutor->CompilerReportingCborFormat()) {
        std::ofstream cbor_stream(dump_filename + ".cbor");
        if (!cbor_stream) {
          LOG(WARNING) << "Unable to open file " << dump_filename
                       << ".cbor , the profiler summary will not be saved.";
        } else {
          poplar::serializeToCBOR(cbor_stream, p_e.graphProfile);
          LOG(INFO) << "Out of memory CBOR profile saved to " << dump_filename
                    << ".cbor.";
        }
      } else {
        std::ofstream js_stream(dump_filename + ".js");
        if (!js_stream) {
          LOG(WARNING) << "Unable to open file " << dump_filename
                       << ".js , the profiler summary will not be saved.";
        } else {
          poplar::serializeToJSON(js_stream, p_e.graphProfile);
          LOG(INFO) << "Out of memory JSON profile saved to " << dump_filename
                    << ".js.";
        }
      }
    }
  }
}

StatusOr<poplar::OptionFlags> GetConvolutionOptionsForInst(
    const HloInstruction* inst, CompilerResources& res) {
  TF_ASSIGN_OR_RETURN(const MLType conv_type, GetMLType(inst));
  return GetConvolutionOptionsForInst(inst, res, conv_type);
}

StatusOr<poplar::OptionFlags> GetConvolutionOptionsForInst(
    const HloInstruction* inst, CompilerResources& res,
    const MLType conv_type) {
  poplar::OptionFlags opts = res.default_conv_options;
  // Set the pass type.
  opts.set("pass", MLType_Name(conv_type));

  // Set the options from the backend config.
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  for (const auto& opt : poplar_backend_config.convolution_options()) {
    opts.set(opt.option(), opt.value());
  }
  TF_RETURN_IF_ERROR(SetPartialsTypeIfPresent(poplar_backend_config, opts));
  return opts;
}

StatusOr<poplar::OptionFlags> GetMatMulOptionsForInst(
    const HloInstruction* inst, CompilerResources& res) {
  poplar::OptionFlags opts = res.default_matmul_options;
  if (!res.clear_matmul_pass_type) {
    // Set the pass type.
    TF_ASSIGN_OR_RETURN(const MLType ml_type, GetMLType(inst));
    opts.set("fullyConnectedPass", MLType_Name(ml_type));
  }

  // Set the options from the backend config.
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  for (const auto& opt : poplar_backend_config.matmul_options()) {
    opts.set(opt.option(), opt.value());
  }
  TF_RETURN_IF_ERROR(SetPartialsTypeIfPresent(poplar_backend_config, opts));
  return opts;
}

Status SetPartialsTypeIfPresent(
    const PoplarBackendConfig& poplar_backend_config,
    poplar::OptionFlags& option_flags) {
  if (poplar_backend_config.partials_type() != PRIMITIVE_TYPE_INVALID) {
    TF_ASSIGN_OR_RETURN(poplar::Type partials_poplar_type,
                        PoplarDataType(poplar_backend_config.partials_type()));
    option_flags.set("partialsType", partials_poplar_type.toString());
  }
  return Status::OK();
}

Status SetPartialsTypeIfPresent(const HloInstruction* inst,
                                poplar::OptionFlags& option_flags) {
  TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                      inst->backend_config<PoplarBackendConfig>());
  return SetPartialsTypeIfPresent(poplar_backend_config, option_flags);
}

poplar::program::Sequence ZeroTensors(CompilerResources& res) {
  poplar::program::Sequence zero_seq;
  for (auto t : res.zeroed_tensors) {
    popops::zero(GetMasterGraph(res), t, zero_seq, "ZeroVar");
  }
  return zero_seq;
}

bool IsRemoteParameter(int64 parameter_number,
                       const RemoteParameterInfos& remote_parameter_infos) {
  return remote_parameter_infos.find(RemoteParameterInfo{parameter_number}) !=
         remote_parameter_infos.end();
}

bool IsRemoteParameter(int64 parameter_number, const CompilerResources& res) {
  return IsRemoteParameter(parameter_number,
                           res.annotations.remote_parameter_infos);
}

bool IsRemoteParameter(HloInstruction* inst, const CompilerResources& res) {
  return IsInstructionInEntryComputation(inst) &&
         IsRemoteParameter(inst->parameter_number(), res);
}

StatusOr<std::string> GetInstructionCompilationInfo(
    const std::unique_ptr<xla::HloModule>& module, CompilerResources& res) {
  TF_ASSIGN_OR_RETURN(auto ml_type_map, GetAllNotNoneMlTypes(module.get()));
  Json::Value ml_types;

  for (auto t : ml_type_map) {
    ml_types[GetDebugName(t.first)] = Json::Value::UInt64(t.second);
  }

  Json::Value root;
  root["ml_types"] = ml_types;

  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";
  return Json::writeString(json_builder, root);
}

namespace {

std::string UnmangleInputName(std::string name) {
  const std::string long_prefix = "XLA_Args/_arg_";
  const std::string short_prefix = "XLA_Args/";

  // Try to match the long prefix
  if (name.find(long_prefix) == 0) {
    name.erase(0, long_prefix.length());

    // Try to remove the suffix _0_0/_0
    const std::regex base_regex("(.*)_\\d+_\\d+/_\\d+");
    std::smatch base_match;
    if (std::regex_match(name, base_match, base_regex)) {
      CHECK_EQ(base_match.size(), 2);
      return base_match[1].str();
    }
  }
  if (name.find(short_prefix) == 0) {
    name.erase(0, short_prefix.length());
  }
  return name;
}

Json::Value DimensionsToJson(const absl::Span<const int64> dimensions) {
  Json::Value js_dims{Json::arrayValue};
  for (auto dim : dimensions) {
    js_dims.append(Json::Value::Int64(dim));
  }
  return js_dims;
}

}  // namespace

Status SaveExecutableMetadataJson(const std::string& filename,
                                  const InputOutputAliasingMap& io_map,
                                  const InfeedInfos& infeed_infos,
                                  const OutfeedInfos& outfeed_infos,
                                  uint32 replication_count,
                                  const poplar::OptionFlags& opts,
                                  const poplar::Target& target) {
  Json::Value inputs;
  std::map<std::string, std::string> params_handle_map;
  for (auto input : io_map.GetEntryInputInfos()) {
    Json::Value stream;
    if (input.Shape().IsTuple()) {
      return tensorflow::errors::Unimplemented("Tuple inputs not supported");
    }

    stream["name"] = UnmangleInputName(input.Name());
    stream["handle"] = GetInputCopyHandle(inputs.size(), 0);
    stream["data_type"] = PrimitiveType_Name(input.Shape().element_type());
    stream["shape"] = DimensionsToJson(input.Shape().dimensions());
    if (input.IsStreaming()) {
      stream["type"] = "input_data";
    } else if (input.IsResource()) {
      stream["type"] = "parameter";
      params_handle_map[GetInputCopyHandle(inputs.size(), 0)] =
          UnmangleInputName(input.Name());
    }
    inputs.append(stream);
  }

  Json::Value outputs;
  for (auto output : io_map.GetEntryOutputInfos()) {
    if (output.Shape().IsTuple()) {
      return xla::FailedPrecondition("Nested tuples in output not supported");
    }
    Json::Value stream;
    stream["name"] = output.Name();
    stream["handle"] = GetOutputCopyHandle(outputs.size(), 0);
    stream["data_type"] = PrimitiveType_Name(output.Shape().element_type());
    stream["shape"] = DimensionsToJson(output.Shape().dimensions());
    if (output.IsStreaming()) {
      stream["type"] = "output_data";
    } else if (output.IsResource()) {
      stream["type"] = "parameter_out";
      if (output.IsResourceModified()) {
        const std::string input_handle =
            GetInputCopyHandle(output.GetInputIndex(), 0);
        stream["input_handle"] = input_handle;
        // Override the name to make sure it matches the one from the input.
        stream["name"] = params_handle_map.at(input_handle);
      }
    }
    outputs.append(stream);
  }

  Json::Value infeeds;
  for (auto infeed : infeed_infos) {
    if (!infeed.shape.IsTuple() || infeed.shape.tuple_shapes_size() != 2 ||
        !infeed.shape.tuple_shapes(0).IsTuple()) {
      return xla::FailedPrecondition(
          "Expected the shape of the infeed %s to be of the shape ((shape), "
          "token[]).",
          infeed.config.feed_id());
    }
    Json::Value feed;
    Json::Value streams;
    feed["name"] = infeed.config.feed_id();
    for (auto shape : infeed.shape.tuple_shapes(0).tuple_shapes()) {
      if (shape.IsTuple()) {
        return xla::FailedPrecondition(
            "Nested tuples in infeed not supported: shape for %s expected to "
            "be something like ((shape), token[]).",
            infeed.config.feed_id());
      }
      Json::Value stream;
      stream["name"] =
          absl::StrCat(infeed.config.feed_id(), ".", streams.size());
      stream["handle"] =
          GetInfeedCopyHandle(infeed.stream_prefix, streams.size());
      stream["shape"] = DimensionsToJson(shape.dimensions());
      stream["data_type"] = PrimitiveType_Name(shape.element_type());
      streams.append(stream);
    }
    feed["streams"] = streams;
    infeeds.append(feed);
  }

  Json::Value outfeeds;
  for (auto outfeed : outfeed_infos) {
    auto shapes = outfeed.shape.IsTuple()
                      ? outfeed.shape.tuple_shapes()
                      : std::vector<xla::Shape>({outfeed.shape});
    Json::Value streams;
    Json::Value feed;
    feed["name"] = outfeed.config.feed_id();
    for (auto shape : shapes) {
      if (shape.IsTuple()) {
        return xla::FailedPrecondition(
            "Nested tuples in outfeed not supported: shape for tuple %d in %s "
            "is a tuple %s",
            streams.size(), outfeed.config.feed_id(), shape.ToString());
      }
      Json::Value stream;
      stream["name"] =
          absl::StrCat(outfeed.config.feed_id(), ".", streams.size());
      stream["handle"] =
          GetOutfeedCopyHandle(outfeed.stream_prefix, streams.size());
      stream["data_type"] = PrimitiveType_Name(shape.element_type());
      stream["shape"] = DimensionsToJson(shape.dimensions());
      streams.append(stream);
    }
    feed["streams"] = streams;
    outfeeds.append(feed);
  }

  Json::Value config;
  Json::Value options;
  for (auto opt : opts) {
    options[opt.first] = opt.second;
  }
  if (!options.empty()) {
    config["options"] = options;
  }
  config["replication_count"] = Json::Value::Int64(replication_count);
  config["num_ipus"] = Json::Value::Int64(target.getNumIPUs());
  if (target.getTargetType() != poplar::TargetType::IPU) {
    return xla::FailedPrecondition(
        "The target's type must be poplar::TargetType::IPU");
  }

  Json::Value root;
  if (!inputs.empty()) {
    root["inputs"] = inputs;
  }
  if (!outputs.empty()) {
    root["outputs"] = outputs;
  }
  if (!infeeds.empty()) {
    root["infeeds"] = infeeds;
  }
  if (!outfeeds.empty()) {
    root["outfeeds"] = outfeeds;
  }
  root["config"] = config;

  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";

  std::string json_msg = Json::writeString(json_builder, root);
  VLOG(1) << "Module JSON Metadata: " << json_msg;
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewWritableFile(filename, &file));
  TF_RETURN_IF_ERROR(file->Append(json_msg));
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}
std::string GetTensorMappingJson(const std::string& module_name,
                                 const poplar::Graph& graph,
                                 const TensorMaps& tensor_maps) {
  Json::Value mappings;

  for (auto tm : tensor_maps) {
    mappings[tm.computation] = Json::Value(Json::arrayValue);

    for (auto tensor : tm.tensor_map) {
      const auto& pop_tensor = tensor.tensor;
      const auto& mapping = graph.getTileMapping(pop_tensor);
      Json::Value tiles = Json::Value(Json::arrayValue);

      size_t total_elements = 0;
      for (size_t tile_idx = 0; tile_idx < mapping.size(); tile_idx++) {
        const auto& tile = mapping[tile_idx];
        if (tile.size() > 0) {
          size_t element_count = 0;
          for (const auto& interval : tile) {
            element_count += interval.size();
          }
          Json::Value tile_info(Json::arrayValue);
          tile_info.append(Json::Value::UInt64(tile_idx));
          tile_info.append(Json::Value::UInt64(element_count));
          tiles.append(tile_info);

          total_elements += element_count;
        }
      }

      Json::Value tensor_shape(Json::arrayValue);
      for (auto d : pop_tensor.shape()) {
        tensor_shape.append(Json::Value::UInt64(d));
      }

      Json::Value js_tensor(Json::arrayValue);
      js_tensor.append(Json::Value(tensor.location.instruction->name()));
      js_tensor.append(
          Json::Value::UInt64(tensor.location.flattened_output_tuple_index));
      js_tensor.append(tensor_shape);
      js_tensor.append(Json::Value(pop_tensor.elementType().toString()));
      js_tensor.append(Json::Value::UInt64(pop_tensor.containsConstant()));
      js_tensor.append(Json::Value::UInt64(pop_tensor.containsAliases()));
      js_tensor.append(Json::Value::UInt64(total_elements));
      js_tensor.append(tiles);
      js_tensor.append(Json::Value(tensor.name));

      mappings[tm.computation].append(js_tensor);
    }
  }

  Json::Value root;
  root["mappings"] = mappings;

  Json::StreamWriterBuilder json_builder;
  json_builder["indentation"] = "";
  json_builder["commentStyle"] = "None";
  std::string json_msg = Json::writeString(json_builder, root);

  if (PoplarXlaFlags::Get().tensor_map_file_path.size() > 0) {
    VLOG(2) << "[Poplar] Dumping tensor mapping";
    auto path = PoplarXlaFlags::Get().tensor_map_file_path;
    auto filename =
        tensorflow::io::JoinPath(path, module_name + ".tensor_map.json");
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
    TF_CHECK_OK(file->Append(json_msg));
    TF_CHECK_OK(file->Close());
  }

  return json_msg;
}

poplar::program::Sequence TensorCopyWithAliasing(poplar::Graph& graph,
                                                 const poplar::Tensor& src,
                                                 const poplar::Tensor& dst) {
  poplar::program::Sequence seq;
  poplar::Tensor src_flat = src.flatten();
  poplar::Tensor dst_flat = dst.flatten();
  // Get the aliasing information.
  std::vector<std::vector<poplar::Interval>> flat_dealiased_intervals =
      graph.getSortedContiguousRegions(src_flat, {{0, src_flat.numElements()}},
                                       true);
  // Dealias inputs and outputs.
  src_flat = poplar::concat(src_flat.slices(flat_dealiased_intervals));
  dst_flat = poplar::concat(dst_flat.slices(flat_dealiased_intervals));
  seq.add(poplar::program::Copy(src_flat, dst_flat));
  return seq;
}

void NotifySlicePlanAllocation(CompilerResources& res,
                               const popops::SlicePlan* plan) {
  if (plan != nullptr) {
    res.used_slice_plan.insert(plan);
  }
}

bool SlicePlanHasAllocation(CompilerResources& res,
                            const popops::SlicePlan* plan) {
  return res.used_slice_plan.count(plan) == 1;
}

StatusOr<const popops::SlicePlan*> GetSlicePlan(CompilerResources& res,
                                                const HloInstruction* inst) {
  auto plan = res.slice_plan_mappings.find(inst);
  if (plan == res.slice_plan_mappings.end()) {
    return xla::FailedPrecondition("Could not find a slice plan for %s.",
                                   inst->ToString().c_str());
  }
  return plan->second;
}

DeferredArgVectors ConvertInputsToDeferredInputs(TensorVectors& inputs) {
  DeferredArgVectors deferred_inputs(inputs.size());
  for (uint64 i = 0; i != inputs.size(); ++i) {
    deferred_inputs[i] = {inputs[i].begin(), inputs[i].end()};
  }
  return deferred_inputs;
}

}  // namespace poplarplugin
}  // namespace xla
