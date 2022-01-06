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

#include "tensorflow/compiler/plugin/poplar/driver/passes/lower_frontend_attributes.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/pipeline_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/human_readable_json.h"

namespace xla {

namespace poplarplugin {
namespace {
Status LowerPipelineFrontendAttributesIntoStage(
    HloInstruction* stage, PipelineStagePoplarConfig stage_config,
    CallGraph* call_graph) {
  // Go through all the computations this stage calls and set the backend config
  // with the specified options.
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<HloComputation*> called_in_stage,
                      GetAllComputationsCalledBy(stage, call_graph));
  for (HloComputation* comp : called_in_stage) {
    for (HloInstruction* inst : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                          inst->backend_config<PoplarBackendConfig>());
      poplar_backend_config.mutable_convolution_options()->CopyFrom(
          stage_config.convolution_options());
      poplar_backend_config.mutable_matmul_options()->CopyFrom(
          stage_config.matmul_options());
      poplar_backend_config.mutable_slice_options()->CopyFrom(
          stage_config.slice_options());
      TF_RETURN_IF_ERROR(inst->set_backend_config(poplar_backend_config));
    }
  }
  return Status::OK();
}

Status LowerPipelineFrontendAttributes(HloInstruction* pipeline_op,
                                       CallGraph* call_graph) {
  // First find the PIPELINE_POPLAR_CONFIG attribute.
  auto attributes = pipeline_op->frontend_attributes();
  auto pipeline_poplar_config_itr =
      attributes.map().find(FrontendAttributeId_Name(PIPELINE_POPLAR_CONFIG));
  if (pipeline_poplar_config_itr == attributes.map().end()) {
    return FailedPrecondition(
        "Could not find the PIPELINE_POPLAR_CONFIG in the pipeline operation "
        "'%s'.",
        pipeline_op->name().c_str());
  }
  PipelinePoplarConfig config;
  TF_RETURN_IF_ERROR(tensorflow::HumanReadableJsonToProto(
      pipeline_poplar_config_itr->second, &config));

  // Get the pipeline stages.
  HloComputation* pipeline_computation = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages,
                      GetPipelineStages(pipeline_computation));

  // Lower the configs into forward stages.
  CHECK_EQ(config.forward_stages_size(), stages.forward.size());
  for (size_t i = 0; i != stages.forward.size(); ++i) {
    TF_RETURN_IF_ERROR(LowerPipelineFrontendAttributesIntoStage(
        stages.forward[i], config.forward_stages(i), call_graph));
  }

  // Lower the configs into recomputation stages.
  // Recomputation stages use the forward stage options.
  for (auto recomp_pair : stages.recomputation) {
    int64 stage_id = recomp_pair.first;
    HloInstruction* stage = recomp_pair.second;
    TF_RETURN_IF_ERROR(LowerPipelineFrontendAttributesIntoStage(
        stage, config.forward_stages(stage_id), call_graph));
  }

  // Lower the configs into backward stages.
  CHECK_EQ(config.backward_stages_size(), stages.backward.size());
  for (size_t i = 0; i != stages.backward.size(); ++i) {
    TF_RETURN_IF_ERROR(LowerPipelineFrontendAttributesIntoStage(
        stages.backward[i], config.backward_stages(i), call_graph));
  }

  // Lower the config into resource update.
  if (stages.resource_update) {
    TF_RETURN_IF_ERROR(LowerPipelineFrontendAttributesIntoStage(
        *stages.resource_update, config.resource_update(), call_graph));
  }

  return Status::OK();
}
}  // namespace

StatusOr<bool> LowerFrontendAttributes::Run(HloModule* module) {
  bool changed = false;

  // First lower any pipeline specific attributes.
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.size()) {
    CHECK_EQ(pipeline_ops.size(), 1);
    TF_RETURN_IF_ERROR(
        LowerPipelineFrontendAttributes(pipeline_ops[0], call_graph.get()));
    changed = true;
  }

  // Note: we expect all the instructions to have a value for these attributes
  // However in some cases optimizers introduce new nodes without preserving
  // the frontend attributes of the node they replace which is why the variables
  // used to lower the frontend attributes in this method are declared outside
  // of the loop. This way we use the last successfully parsed values to
  // approximate the missing values.
  PrimitiveType partials_type = PRIMITIVE_TYPE_INVALID;

  for (auto* comp : module->computations()) {
    for (auto instr : comp->instructions()) {
      auto attributes = instr->frontend_attributes();
      TF_ASSIGN_OR_RETURN(auto poplar_backend_config,
                          instr->backend_config<PoplarBackendConfig>());
      auto partials_type_attribute =
          attributes.map().find(FrontendAttributeId_Name(PARTIALS_TYPE));
      if (partials_type_attribute != attributes.map().end()) {
        bool type_parsed = PrimitiveType_Parse(partials_type_attribute->second,
                                               &partials_type);
        if (!type_parsed) {
          return xla::FailedPrecondition("Could not parse the partials type.");
        }
        switch (partials_type) {
          case F32:
          case F16:
          case PRIMITIVE_TYPE_INVALID:  // Switch back to default
            // Allowed partials type
            break;
          default:
            return xla::FailedPrecondition("Unsupported partials type.");
        }
        changed = true;
      }
      // Change default stochastic rounding mode to be undefined, so we can
      // easily catch cases where the option hasn't been set.
      poplar_backend_config.set_stochastic_rounding(THREESTATE_UNDEFINED);
      poplar_backend_config.set_partials_type(partials_type);
      TF_RETURN_IF_ERROR(instr->set_backend_config(poplar_backend_config));
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
