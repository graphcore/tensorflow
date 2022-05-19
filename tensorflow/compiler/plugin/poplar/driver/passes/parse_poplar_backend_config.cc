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

#include "tensorflow/compiler/plugin/poplar/driver/passes/parse_poplar_backend_config.h"

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace poplarplugin {
using pipeline_config = PoplarBackendConfig::CallConfig::PipelineConfig;
namespace {

StatusOr<std::string> GetAttribute(const FrontendAttributes& attributes,
                                   const std::string& id_string) {
  auto itr = attributes.map().find(id_string);
  if (itr == attributes.map().end()) {
    return FailedPrecondition("Expected an attribute %s.", id_string);
  }
  return itr->second;
}
StatusOr<std::string> GetAttribute(const FrontendAttributes& attributes,
                                   const FrontendAttributeId id) {
  std::string id_string = FrontendAttributeId_Name(id);
  return GetAttribute(attributes, id_string);
}

StatusOr<ThreeState> ParseThreeState(const std::string& value) {
  ThreeState parsed;
  if (!ThreeState_Parse(value, &parsed)) {
    return FailedPrecondition("Could not parse the ThreeState value.");
  }
  return parsed;
}

StatusOr<pipeline_config::RecomputationMode> ParseRecomputationMode(
    const std::string& value) {
  pipeline_config::RecomputationMode parsed;
  if (!pipeline_config::RecomputationMode_Parse(value, &parsed)) {
    return FailedPrecondition("Could not parse the RecomputationMode value.");
  }
  return parsed;
}

static Status InitialiseFunctionConfig(
    const FrontendAttributes& attributes,
    PoplarBackendConfig::CallConfig::FunctionConfig* function_config) {
  TF_ASSIGN_OR_RETURN(std::string unique_sharding_str,
                      GetAttribute(attributes, UNIQUE_SHARDING));
  bool unique_sharding = std::stoi(unique_sharding_str);
  function_config->set_unique_sharding(unique_sharding);
  TF_ASSIGN_OR_RETURN(std::string keep_input_layouts_str,
                      GetAttribute(attributes, KEEP_INPUT_LAYOUTS));
  bool keep_input_layouts = std::stoi(keep_input_layouts_str);
  function_config->set_keep_input_layouts(keep_input_layouts);
  return Status::OK();
}

static Status InitialisePipelineConfig(
    const FrontendAttributes& attributes,
    PoplarBackendConfig::CallConfig::PipelineConfig* pipeline_config) {
  // Get the batch serialization iterations.
  TF_ASSIGN_OR_RETURN(
      std::string batch_serialization_iterations_str,
      GetAttribute(attributes, PIPELINE_BATCH_SERIALIZATION_ITERATIONS));
  int64_t batch_serialization_iterations =
      std::stoll(batch_serialization_iterations_str);
  pipeline_config->set_batch_serialization_iterations(
      batch_serialization_iterations);

  // Get the repeat count.
  TF_ASSIGN_OR_RETURN(std::string repeat_count_str,
                      GetAttribute(attributes, PIPELINE_REPEAT_COUNT));
  int64_t repeat_count = std::stoll(repeat_count_str);
  pipeline_config->set_repeat_count(repeat_count);

  // Get the schedule.
  TF_ASSIGN_OR_RETURN(std::string schedule_str,
                      GetAttribute(attributes, PIPELINE_SCHEDULE));
  auto schedule =
      static_cast<PoplarBackendConfig::CallConfig::PipelineConfig::Schedule>(
          std::stoi(schedule_str));
  pipeline_config->set_schedule({schedule});

  // Set the offload activations flag.
  TF_ASSIGN_OR_RETURN(std::string offload_activations_str,
                      GetAttribute(attributes, OFFLOAD_ACTIVATIONS));
  TF_ASSIGN_OR_RETURN(auto offload_activations,
                      ParseThreeState(offload_activations_str));
  pipeline_config->set_offload_activations(offload_activations);

  // Set the partition variables flag.
  TF_ASSIGN_OR_RETURN(std::string partition_variables_str,
                      GetAttribute(attributes, PARTITION_VARIABLES));
  TF_ASSIGN_OR_RETURN(auto partition_variables,
                      ParseThreeState(partition_variables_str));
  pipeline_config->set_partition_variables(partition_variables);

  // Set the offload variables flag.
  TF_ASSIGN_OR_RETURN(std::string offload_variables_str,
                      GetAttribute(attributes, OFFLOAD_VARIABLES));
  TF_ASSIGN_OR_RETURN(auto offload_variables,
                      ParseThreeState(offload_variables_str));
  pipeline_config->set_offload_variables(offload_variables);

  // Set the offload gradient accumulation buffers flag.
  TF_ASSIGN_OR_RETURN(
      std::string offload_gradient_accumulation_buffers_str,
      GetAttribute(attributes, OFFLOAD_GRADIENT_ACCUMULATION_BUFFERS));
  TF_ASSIGN_OR_RETURN(
      auto offload_gradient_accumulation_buffers,
      ParseThreeState(offload_gradient_accumulation_buffers_str));
  pipeline_config->set_offload_gradient_accumulation_buffers(
      offload_gradient_accumulation_buffers);

  // Set the recomputation mode flag.
  TF_ASSIGN_OR_RETURN(std::string recomputation_mode_str,
                      GetAttribute(attributes, RECOMPUTATION_MODE));
  TF_ASSIGN_OR_RETURN(auto recomputation_mode,
                      ParseRecomputationMode(recomputation_mode_str));
  pipeline_config->set_recomputation_mode(recomputation_mode);

  TF_ASSIGN_OR_RETURN(
      std::string gradient_accumulation_index,
      GetAttribute(attributes, "GradientAccumulationOperandIndex"));
  pipeline_config->set_gradient_accumulation_index(
      std::stoll(gradient_accumulation_index));
  return Status::OK();
}

static Status InitialisePipelineStageConfig(
    const FrontendAttributes& attributes,
    PoplarBackendConfig::CallConfig::PipelineStageConfig*
        pipeline_stage_config) {
  // Get the stage id.
  TF_ASSIGN_OR_RETURN(std::string stage_id_str,
                      GetAttribute(attributes, PIPELINE_STAGE_ID));
  int64_t stage_id = std::stoll(stage_id_str);
  pipeline_stage_config->set_stage_id(stage_id);
  return Status::OK();
}

static Status InitialiseResourceUpdateConfig(
    const FrontendAttributes& attributes,
    PoplarBackendConfig::CallConfig::ResourceUpdateConfig*
        resource_update_config) {
  // Get the offload variables flag.
  TF_ASSIGN_OR_RETURN(
      std::string offload_variables_str,
      GetAttribute(attributes, OFFLOAD_WEIGHT_UPDATE_VARIABLES));
  TF_ASSIGN_OR_RETURN(auto offload_variables,
                      ParseThreeState(offload_variables_str));
  resource_update_config->set_offload_variables(offload_variables);

  // Get the partition offload variables flag.
  TF_ASSIGN_OR_RETURN(
      std::string partition_offload_variables_str,
      GetAttribute(attributes, PARTITION_OFFLOADED_WEIGHT_UPDATE_VARIABLES));
  TF_ASSIGN_OR_RETURN(auto partition_offload_variables,
                      ParseThreeState(partition_offload_variables_str));
  resource_update_config->set_partition_offloaded_variables(
      partition_offload_variables);

  return Status::OK();
}

}  // namespace

StatusOr<bool> ParsePoplarBackendConfig::Run(HloModule* module) {
  bool changed = false;

  for (auto* comp : module->computations()) {
    for (auto instr : comp->instructions()) {
      auto attributes = instr->frontend_attributes();
      // Check if the calls have the type field set from tf2xla.
      if (instr->opcode() == HloOpcode::kCall) {
        PoplarBackendConfig poplar_config;
        auto call_config_type_attribute =
            attributes.map().find(FrontendAttributeId_Name(CALL_CONFIG_TYPE));
        if (call_config_type_attribute != attributes.map().end()) {
          PoplarBackendConfig::CallConfig::Type type;
          bool type_parsed = PoplarBackendConfig_CallConfig_Type_Parse(
              call_config_type_attribute->second, &type);
          if (!type_parsed) {
            return FailedPrecondition("Could not parse the call type.");
          }
          auto* call_config = poplar_config.mutable_call_config();
          call_config->set_type(type);
          switch (type) {
            case PoplarBackendConfig::CallConfig::Function: {
              TF_RETURN_IF_ERROR(InitialiseFunctionConfig(
                  attributes, call_config->mutable_function_config()));
              break;
            }
            case PoplarBackendConfig::CallConfig::Pipeline: {
              TF_RETURN_IF_ERROR(InitialisePipelineConfig(
                  attributes, call_config->mutable_pipeline_config()));
              break;
            }
            case PoplarBackendConfig::CallConfig::PipelineStage:
            case PoplarBackendConfig::CallConfig::PipelineStageBackward: {
              TF_RETURN_IF_ERROR(InitialisePipelineStageConfig(
                  attributes, call_config->mutable_pipeline_stage_config()));
              break;
            }
            case PoplarBackendConfig::CallConfig::ResourceUpdate: {
              TF_RETURN_IF_ERROR(InitialiseResourceUpdateConfig(
                  attributes, call_config->mutable_resource_update_config()));
            }
            default: { break; }
          }
          changed = true;
        }
        TF_RETURN_IF_ERROR(instr->set_backend_config(poplar_config));
      }
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
