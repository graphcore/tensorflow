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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_COMPILER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_COMPILER_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {
namespace poplarplugin {

// The compiler translates XLA HLO code into a PopIT executable
// (eg a PopIT function id).
class PopItCompiler : public Compiler {
 public:
  PopItCompiler() {}
  ~PopItCompiler() override {}

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      perftools::gputools::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      perftools::gputools::StreamExecutor* executor,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup>,
                     const AotCompilationOptions&) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  perftools::gputools::Platform::Id PlatformId() const override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopItCompiler);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_COMPILER_H_
