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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_CREATOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_CREATOR_H_

#include <memory>
#include <string>
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

namespace xla {
namespace poplarplugin {

StatusOr<std::unique_ptr<PipelineVisitor>> GetPipelineVisitor(
    const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_CREATOR_H_
