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
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/ml_type_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "include/json/json.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"

#include <algorithm>
#include <fstream>
#include <limits>

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
    if (flattened_output_tuple_index >= sharding.size()) {
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

    if (device_id >= res.shard_graphs.size()) {
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

Status PoplarExceptionToTensorflowStatus(const std::string& prefix,
                                         const std::exception& e) {
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
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::stream_memory_allocation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::graph_memory_allocation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::tensor_creation_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::memory_elem_constraints_error& e) {
    return tensorflow::errors::ResourceExhausted(prefix, e.what());
  } catch (const poplar::index_error& e) {
    return tensorflow::errors::OutOfRange(prefix, e.what());
  } catch (const poplar::poplar_error& e) {
    return tensorflow::errors::Internal(prefix, e.what());
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
    const PoplarExecutor* poplarExecutor) {
  auto dump_filename = PoplarXlaFlags::Get().save_oom_profiler;
  if (!dump_filename.empty()) {
    try {
      std::rethrow_exception(std::current_exception());
    } catch (const poplar::graph_memory_allocation_error& p_e) {
      if (p_e.graphProfile.type() == poplar::ProfileValue::Type::MAP &&
          p_e.graphProfile.size() != 0) {
        auto opts = poplarExecutor->GetReportFlags();
        SetFlagIfNotPresent(opts, "showVarStorage", "true");

        // Always produce a text report
        std::ofstream stream(dump_filename);
        if (!stream) {
          LOG(WARNING) << "Unable to open file " << dump_filename
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
              LOG(INFO) << "Out of memory CBOR profile saved to "
                        << dump_filename << ".cbor.";
            }
          } else {
            std::ofstream js_stream(dump_filename + ".js");
            if (!js_stream) {
              LOG(WARNING) << "Unable to open file " << dump_filename
                           << ".js , the profiler summary will not be saved.";
            } else {
              poplar::serializeToJSON(js_stream, p_e.graphProfile);
              LOG(INFO) << "Out of memory JSON profile saved to "
                        << dump_filename << ".js.";
            }
          }
        }
      }
    }
  }
}

poplar::OptionFlags GetConvolutionOptionsForType(CompilerResources& res,
                                                 const MLType conv_type) {
  poplar::OptionFlags opts = res.default_conv_options;
  opts.set("pass", MLType_Name(conv_type));
  return opts;
}

poplar::OptionFlags GetMatMulOptionsForType(CompilerResources& res,
                                            const MLType mm_type) {
  poplar::OptionFlags opts = res.default_matmul_options;
  if (!res.clear_matmul_pass_type) {
    opts.set("fullyConnectedPass", MLType_Name(mm_type));
  }
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

std::string GetTensorMappingJson(const std::string& module_name,
                                 const poplar::Graph& graph,
                                 const TensorMaps& tensor_maps) {
  Json::Value mappings;

  for (auto tm : tensor_maps) {
    mappings[tm.first] = Json::Value(Json::arrayValue);

    for (auto pair : tm.second) {
      const auto& pop_tensor = pair.second;
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

      Json::Value tensor(Json::arrayValue);
      tensor.append(Json::Value(pair.first.first));
      tensor.append(Json::Value::UInt64(pair.first.second));
      tensor.append(tensor_shape);
      tensor.append(Json::Value(pop_tensor.elementType().toString()));
      tensor.append(Json::Value::UInt64(pop_tensor.containsConstant()));
      tensor.append(Json::Value::UInt64(pop_tensor.containsAliases()));
      tensor.append(Json::Value::UInt64(total_elements));
      tensor.append(tiles);

      mappings[tm.first].append(tensor);
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

StatusOr<const popops::SlicePlan*> GetSlicePlan(CompilerResources& res,
                                                const HloInstruction* inst) {
  auto plan = res.slice_plan_mappings.find(inst);
  if (plan == res.slice_plan_mappings.end()) {
    return xla::FailedPrecondition("Could not find a slice plan for %s.",
                                   inst->ToString().c_str());
  }
  return plan->second;
}
}  // namespace poplarplugin
}  // namespace xla
