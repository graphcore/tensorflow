/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

namespace xla {
namespace poplarplugin {

StatusOr<std::unique_ptr<HloModule>> PopItCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    const CompileOptions& options) {
  // We just do everything inside RunBackend as our lowering
  // passes are mixed with our optimisations.
  return module;
}

StatusOr<std::unique_ptr<Executable>> PopItCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  TENSORFLOW_TRACEPOINT();
  return tensorflow::errors::Unimplemented("Run backend not implemented yet");
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PopItCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    const CompileOptions& options) {
  TENSORFLOW_TRACEPOINT();
  auto hlo_modules = module_group->ConsumeModules();
  if (hlo_modules.size() != stream_exec.size()) {
    return tensorflow::errors::Unimplemented(
        "Only support cases where number of stream executors "
        "matches number of modules");
  }
  std::vector<std::unique_ptr<Executable>> result;
  result.reserve(hlo_modules.size());
  for (int64_t i = 0; i < hlo_modules.size(); ++i) {
    if (stream_exec[i].size() != 1) {
      return tensorflow::errors::Unimplemented(
          "Only support cases where number of stream executors "
          "is one");
    }
    TF_ASSIGN_OR_RETURN(auto module, RunHloPasses(std::move(hlo_modules[i]),
                                                  stream_exec[i][0], options));
    TF_ASSIGN_OR_RETURN(
        auto executable,
        RunBackend(std::move(module), stream_exec[i][0], options));
    result.emplace_back(std::move(executable));
  }
  return result;
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PopItCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup>,
                                  const AotCompilationOptions&) {
  return xla::InvalidArgument("AOT compilation not supported on PopIT");
}

se::Platform::Id PopItCompiler::PlatformId() const { return kPopItPlatformId; }

HloCostAnalysis::ShapeSizeFunction PopItCompiler::ShapeSizeBytesFunction()
    const {
  return cpu::CpuExecutable::ShapeSizeBytes;
}

static std::unique_ptr<xla::ComputationPlacer> PopItCreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool PopItRegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      xla::poplarplugin::kPopItPlatformId, &PopItCreateComputationPlacer);
  return true;
}

bool popit_placer_registration = PopItRegisterComputationPlacer();

static bool PopItInitModule() {
  xla::Compiler::RegisterCompilerFactory(
      xla::poplarplugin::kPopItPlatformId,
      []() { return absl::make_unique<xla::poplarplugin::PopItCompiler>(); });
  return true;
}
static bool module_initialized = PopItInitModule();

}  // namespace poplarplugin
}  // namespace xla
