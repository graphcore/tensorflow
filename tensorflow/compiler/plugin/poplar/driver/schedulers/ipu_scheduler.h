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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_IPU_SCHEDULER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_IPU_SCHEDULER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"

namespace xla {
namespace poplarplugin {
struct CompilerAnnotations;

using IpuSchedulerAlgorithm = std::function<StatusOr<HloInstructionSequence>(
    HloComputation*, const HloPoplarDataflowAnalysis&,
    const absl::flat_hash_map<const HloComputation*, int64_t>&)>;

struct NamedIpuSchedulerAlgorithm {
  std::string name;
  IpuSchedulerAlgorithm function;
  NamedIpuSchedulerAlgorithm(std::string name, IpuSchedulerAlgorithm function)
      : name(std::move(name)), function(std::move(function)) {}
};

using MemoryEstimator = std::function<int64_t(
    const std::vector<HloInstruction*>&,
    const HloInstructionMap<HloPoplarBufferSet>&,
    const absl::flat_hash_map<HloPoplarBuffer::Id, int64_t>&,
    const absl::flat_hash_map<const HloComputation*, int64_t>&)>;
MemoryEstimator HeapMemoryEstimator();

/**
 *
 * An HLO module pass which applies the best scheduling algorithm from the
 * provided list, to schedule each computation in the module. Where the best
 * algorithm is determined by the provided MemoryEstimator.
 */
class ChooseBestIpuScheduler : public HloModulePass {
 public:
  ChooseBestIpuScheduler(
      const std::vector<NamedIpuSchedulerAlgorithm>& algorithms,
      const CompilerAnnotations* annotations = nullptr,
      MemoryEstimator memory_estimator = HeapMemoryEstimator());

  absl::string_view name() const override {
    return "chooose-best-ipu-scheduler";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  std::vector<NamedIpuSchedulerAlgorithm> algorithms_;
  const CompilerAnnotations* annotations_;
  MemoryEstimator memory_estimator_;
};

/**
 * An HLO module pass which applies the given scheduling algorithm to each
 * computation in the module.
 */
class IpuScheduler : public HloModulePass {
 public:
  // size_function is the function returning the number of bytes required for a
  // LogicalBuffer. algorithm is the memory scheduling algorithm to use.
  IpuScheduler(IpuSchedulerAlgorithm algorithm,
               const CompilerAnnotations* annotations = nullptr);

  absl::string_view name() const override { return "ipu-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;
  StatusOr<bool> Run(
      HloModule* module,
      std::unique_ptr<HloPoplarDataflowAnalysis> dataflow_analysis);

 private:
  IpuSchedulerAlgorithm algorithm_;
  const CompilerAnnotations* annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
