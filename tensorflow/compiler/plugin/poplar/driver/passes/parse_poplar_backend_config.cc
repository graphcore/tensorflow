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
namespace {
StatusOr<std::string> GetAttribute(const FrontendAttributes& attributes,
                                   const FrontendAttributeId id) {
  std::string id_string = FrontendAttributeId_Name(id);
  auto itr = attributes.map().find(id_string);
  if (itr == attributes.map().end()) {
    return xla::FailedPrecondition("Expected an attribute %s.", id_string);
  }
  return itr->second;
}
}  // namespace

StatusOr<bool> ParsePoplarBackendConfig::Run(HloModule* module) {
  bool changed = false;

  for (auto* comp : module->computations()) {
    for (auto instr : comp->instructions()) {
      auto attributes = instr->frontend_attributes();
      PoplarBackendConfig poplar_config;
      // Check if the calls have the type field set from tf2xla.
      if (instr->opcode() == HloOpcode::kCall) {
        auto call_config_type_attribute =
            attributes.map().find(FrontendAttributeId_Name(CALL_CONFIG_TYPE));
        if (call_config_type_attribute != attributes.map().end()) {
          PoplarBackendConfig::CallConfig::Type type;
          bool type_parsed = PoplarBackendConfig_CallConfig_Type_Parse(
              call_config_type_attribute->second, &type);
          if (!type_parsed) {
            return xla::FailedPrecondition("Could not parse the call type.");
          }
          auto* call_config = poplar_config.mutable_call_config();
          call_config->set_type(type);
          switch (type) {
            case PoplarBackendConfig::CallConfig::Pipeline: {
              auto* pipeline_config = call_config->mutable_pipeline_config();
              // Get the pipeline depth.
              TF_ASSIGN_OR_RETURN(std::string pipeline_depth_str,
                                  GetAttribute(attributes, PIPELINE_DEPTH));
              int64 pipeline_depth = std::stoll(pipeline_depth_str);
              pipeline_config->set_pipeline_depth(pipeline_depth);

              // Get the batch serialization iterations.
              TF_ASSIGN_OR_RETURN(
                  std::string batch_serialization_iterations_str,
                  GetAttribute(attributes,
                               PIPELINE_BATCH_SERIALIZATION_ITERATIONS));
              int64 batch_serialization_iterations =
                  std::stoll(batch_serialization_iterations_str);
              pipeline_config->set_batch_serialization_iterations(
                  batch_serialization_iterations);

              // Get the repeat count.
              TF_ASSIGN_OR_RETURN(
                  std::string repeat_count_str,
                  GetAttribute(attributes, PIPELINE_REPEAT_COUNT));
              int64 repeat_count = std::stoll(repeat_count_str);
              pipeline_config->set_repeat_count(repeat_count);

              // Get the schedule.
              TF_ASSIGN_OR_RETURN(std::string schedule_str,
                                  GetAttribute(attributes, PIPELINE_SCHEDULE));
              auto schedule = static_cast<
                  PoplarBackendConfig::CallConfig::PipelineConfig::Schedule>(
                  std::stoi(schedule_str));
              pipeline_config->set_schedule({schedule});
              break;
            }
            case PoplarBackendConfig::CallConfig::PipelineStage:
            case PoplarBackendConfig::CallConfig::PipelineStageBackward: {
              auto* pipeline_stage_config =
                  call_config->mutable_pipeline_stage_config();
              // Get the stage id.
              TF_ASSIGN_OR_RETURN(std::string stage_id_str,
                                  GetAttribute(attributes, PIPELINE_STAGE_ID));
              int64 stage_id = std::stoll(stage_id_str);
              pipeline_stage_config->set_stage_id(stage_id);
              break;
            }
            case PoplarBackendConfig::CallConfig::ResourceUpdate: {
              auto* resource_update_config =
                  call_config->mutable_resource_update_config();
              // Get the offload variables flag.
              TF_ASSIGN_OR_RETURN(std::string offload_variables_str,
                                  GetAttribute(attributes, OFFLOAD_VARIABLES));
              auto offload_variables = std::stoi(offload_variables_str);
              resource_update_config->set_offload_variables(offload_variables);

              // Get the partition offload variables flag.
              TF_ASSIGN_OR_RETURN(
                  std::string partition_offload_variables_str,
                  GetAttribute(attributes, PARTITION_OFFLOADED_VARIABLES));
              auto partition_offload_variables =
                  std::stoi(partition_offload_variables_str);
              resource_update_config->set_partition_offloaded_variables(
                  partition_offload_variables);

              // Get the num batches to accumulate flag.
              TF_ASSIGN_OR_RETURN(
                  std::string num_batches_to_accumulate_str,
                  GetAttribute(attributes, NUM_BATCHES_TO_ACCUMULATE));
              auto num_batches_to_accumulate =
                  std::stoi(num_batches_to_accumulate_str);
              resource_update_config->set_num_batches_to_accumulate(
                  num_batches_to_accumulate);
            }
            default: { break; }
          }
          changed = true;
        }
      }
      instr->set_backend_config(poplar_config);
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
