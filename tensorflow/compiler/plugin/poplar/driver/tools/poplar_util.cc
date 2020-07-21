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
#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <regex>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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

    // If the instruction is not allowed tuple shardding, then all the outputs
    // have the same shard.
    if (!IsAllowedTupleSharding(inst)) {
      flattened_output_tuple_index = 0;
    }

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

poplar::OptionFlags GetReplicatedCollectiveOptions(
    const CompilerResources& res) {
  poplar::OptionFlags options;

  // Set the GCL options
  for (const auto& opt : res.gcl_options) {
    options.set(opt.first, opt.second);
  }

  return options;
}

poplar::OptionFlags GetReplicateAllReduceOptions(const CompilerResources& res) {
  poplar::OptionFlags options = GetReplicatedCollectiveOptions(res);
  options.set("useReplicatedImplementation", "true");
  return options;
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

void AddZeroTensorToPreamble(CompilerResources& res, const poplar::Tensor& t) {
  popops::zero(GetMasterGraph(res), t, res.preamble_sequence, "ZeroVar");
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

Status SetIpuShape(ipu::TensorInfo& info, const xla::Shape& xla_shape) {
  ipu::DataType type;
  switch (xla_shape.element_type()) {
    case xla::S32:
      type = ipu::DataType::S32;
      break;
    case xla::F32:
      type = ipu::DataType::F32;
      break;
    case xla::PrimitiveType::F16:
      type = ipu::DataType::F16;
      break;
    default:
      return xla::InvalidArgument("PrimitiveType not supported");
  }
  std::vector<int64_t> dimensions;
  for (auto dim : xla_shape.dimensions()) {
    dimensions.push_back(dim);
  }
  info.SetShape(ipu::TensorShape(dimensions, type));
  return Status::OK();
}

}  // namespace

StatusOr<ipu::Metadata> CreateExecutableMetadata(
    const InputOutputAliasingMap& io_map, const InfeedInfos& infeed_infos,
    const OutfeedInfos& outfeed_infos, uint32 replication_count,
    const poplar::OptionFlags& device_opts,
    const poplar::OptionFlags& engine_opts, const poplar::Target& target,
    const VerifiedStreamsIndices::KeyIdMappings& indices,
    const std::vector<string>& checkpoint_feeds_order) {
  try {
    ipu::MetadataBuilder builder;
    std::map<std::string, std::string> params_handle_map;
    const bool use_verified_transfers = !indices.empty();
    for (auto input : io_map.GetEntryInputInfos()) {
      VLOG(1) << "Processing input " << input.Name();
      if (input.Shape().IsTuple()) {
        return tensorflow::errors::Unimplemented("Tuple inputs not supported");
      }
      ipu::TensorInfo t;
      ipu::VerificationInfo info;
      t.SetHandle(input.Handles().at(0));
      t.SetName(UnmangleInputName(input.Name()));
      if (use_verified_transfers) {
        auto key_id = indices.at(t.Handle());
        info.SetInfo(key_id.key, key_id.id);
      }
      TF_RETURN_IF_ERROR(SetIpuShape(t, input.Shape()));
      if (input.IsStreaming()) {
        builder.AddInput(t, info);
      } else if (input.IsResource()) {
        builder.AddInputParameter(t, info);
        params_handle_map[t.Handle()] = t.Name();
      }
    }

    for (auto output : io_map.GetEntryOutputInfos()) {
      VLOG(1) << "Processing output " << output.Name();
      if (output.Shape().IsTuple()) {
        return xla::FailedPrecondition("Nested tuples in output not supported");
      }
      ipu::TensorInfo t;
      ipu::VerificationInfo info;
      t.SetName(output.Name());
      t.SetHandle(output.Handles().at(0));
      if (use_verified_transfers) {
        auto key_id = indices.at(t.Handle());
        info.SetInfo(key_id.key, key_id.id);
      }
      TF_RETURN_IF_ERROR(SetIpuShape(t, output.Shape()));
      if (output.IsStreaming()) {
        builder.AddOutput(t, info);
      } else if (output.IsResource()) {
        if (output.IsResourceModified()) {
          const std::string input_handle =
              GetInputCopyHandle(output.GetInputIndex(), 0);
          // Override the name to make sure it matches the one from the
          // input.
          t.SetName(params_handle_map.at(input_handle));
          builder.AddOutputModifiedParameter(input_handle, t, info);
        } else {
          builder.AddOutputParameter(t, info);
        }
      }
    }

    for (auto infeed : infeed_infos) {
      VLOG(1) << "Processing infeed " << infeed.config.feed_id();
      if (!infeed.shape.IsTuple() || infeed.shape.tuple_shapes_size() != 2 ||
          !infeed.shape.tuple_shapes(0).IsTuple()) {
        return xla::FailedPrecondition(
            "Expected the shape of the infeed %s to be of the shape ((shape), "
            "token[]).",
            infeed.config.feed_id());
      }
      builder.CreateInfeed(infeed.config.feed_id());
      int64_t stream_idx = 0;
      for (auto shape : infeed.shape.tuple_shapes(0).tuple_shapes()) {
        if (shape.IsTuple()) {
          return xla::FailedPrecondition(
              "Nested tuples in infeed not supported: shape for %s expected to "
              "be something like ((shape), token[]).",
              infeed.config.feed_id());
        }
        ipu::TensorInfo t;
        ipu::VerificationInfo info;
        t.SetHandle(GetInfeedCopyHandle(infeed.stream_prefix, stream_idx));
        t.SetName(absl::StrCat(infeed.config.feed_id(), ".", stream_idx));
        TF_RETURN_IF_ERROR(SetIpuShape(t, shape));
        if (use_verified_transfers) {
          auto key_id = indices.at(t.Handle());
          info.SetInfo(key_id.key, key_id.id);
        }
        builder.AddInfeedStream(infeed.config.feed_id(), t, info);
        stream_idx++;
      }
    }

    for (auto outfeed : outfeed_infos) {
      VLOG(1) << "Processing outfeed " << outfeed.config.feed_id();
      auto shapes = outfeed.shape.IsTuple()
                        ? outfeed.shape.tuple_shapes()
                        : std::vector<xla::Shape>({outfeed.shape});
      builder.CreateOutfeed(outfeed.config.feed_id());
      int64_t stream_idx = 0;
      for (auto shape : shapes) {
        if (shape.IsTuple()) {
          return xla::FailedPrecondition(
              "Nested tuples in outfeed not supported: shape for tuple %d in "
              "%s is a tuple %s",
              stream_idx, outfeed.config.feed_id(), shape.ToString());
        }
        ipu::TensorInfo t;
        ipu::VerificationInfo info;
        t.SetHandle(GetOutfeedCopyHandle(outfeed.stream_prefix, stream_idx));
        t.SetName(absl::StrCat(outfeed.config.feed_id(), ".", stream_idx));
        TF_RETURN_IF_ERROR(SetIpuShape(t, shape));
        if (use_verified_transfers) {
          auto key_id = indices.at(t.Handle());
          info.SetInfo(key_id.key, key_id.id);
        }
        builder.AddOutfeedStream(outfeed.config.feed_id(), t, info);
        stream_idx++;
      }
    }

    for (auto opt : device_opts) {
      builder.AddDeviceOption(opt.first, opt.second);
    }
    for (auto opt : engine_opts) {
      builder.AddEngineOption(opt.first, opt.second);
    }
    builder.SetConfig(replication_count, target.getNumIPUs());
    builder.SetRandomNumberSeedHandle(GetRandomNumberSeedStream());
    if (target.getTargetType() != poplar::TargetType::IPU) {
      return xla::FailedPrecondition(
          "The target's type must be poplar::TargetType::IPU");
    }

    if (!checkpoint_feeds_order.empty()) {
      VLOG(1) << "Creating checkpoint";
      if (!use_verified_transfers) {
        return xla::FailedPrecondition(
            "Can't use checkpoints without verified transfers");
      }
      ipu::VerificationInfo checkpointIn, checkpointOut;
      auto key_id = indices.at(ipu::Metadata::InputCheckpointHandle());
      checkpointIn.SetInfo(key_id.key, key_id.id);
      key_id = indices.at(ipu::Metadata::OutputCheckpointHandle());
      checkpointOut.SetInfo(key_id.key, key_id.id);
      builder.AddCheckpoint(checkpoint_feeds_order, checkpointIn,
                            checkpointOut);
    }
    return builder.BuildMetadata();
  } catch (const std::exception& e) {
    return tensorflow::errors::Internal(e.what());
  }
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

  if (src_flat.containsAliases()) {
    // Get the aliasing information.
    std::vector<std::vector<poplar::Interval>> flat_dealiased_intervals =
        graph.getSortedContiguousRegions(src_flat,
                                         {{0, src_flat.numElements()}}, true);
    // Dealias source and destination.
    src_flat = poplar::concat(src_flat.slices(flat_dealiased_intervals));
    dst_flat = poplar::concat(dst_flat.slices(flat_dealiased_intervals));
  }

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
