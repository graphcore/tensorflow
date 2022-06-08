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
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"

#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {
namespace {

absl::flat_hash_map<HloPoplarBuffer::Id, int64_t> GetBufferSizes(
    const HloInstructionMap<HloPoplarBufferSet>& buffer_uses) {
  absl::flat_hash_map<HloPoplarBuffer::Id, int64_t> buffer_sizes;
  for (auto& item : buffer_uses) {
    buffer_sizes.merge(DeviceMemoryBufferSizesInBytes(item.second));
  }
  return buffer_sizes;
}

StatusOr<HloSchedule> IpuScheduleModule(
    HloModule* module, const IpuSchedulerAlgorithm& algorithm,
    HloPoplarDataflowAnalysis* dataflow_analysis) {
  HloSchedule schedule(module);

  const auto buffer_uses = FindUsedBuffers(module, *dataflow_analysis);
  const auto buffer_sizes = GetBufferSizes(buffer_uses);

  auto memory_estimator = HeapMemoryEstimator();

  absl::flat_hash_map<const HloComputation*, int64_t> memory_by_computation;
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (!computation->IsFusionComputation()) {
      TF_ASSIGN_OR_RETURN(
          HloInstructionSequence computation_sequence,
          algorithm(computation, *dataflow_analysis, memory_by_computation));

      auto bytes =
          memory_estimator(computation_sequence.instructions(), buffer_uses,
                           buffer_sizes, memory_by_computation);
      memory_by_computation[computation] = bytes;
      schedule.set_sequence(computation, std::move(computation_sequence));
    }
  }
  VLOG(2) << "Module schedule:\n" << schedule;

  TF_RETURN_IF_ERROR(schedule.Verify());

  return std::move(schedule);
}

StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    const HloModule* module, const HloPoplarDataflowAnalysis& dataflow_analysis,
    std::vector<NamedIpuSchedulerAlgorithm> algorithms,
    MemoryEstimator memory_estimator) {
  if (algorithms.empty()) {
    return xla::FailedPrecondition(
        "Cannot construct BestIpuSchedule when the input is empty");
  }

  const auto buffer_uses = FindUsedBuffers(module, dataflow_analysis);
  const auto buffer_sizes = GetBufferSizes(buffer_uses);

  return IpuSchedulerAlgorithm{
      [algorithms = std::move(algorithms), memory_estimator = memory_estimator,
       buffer_uses = std::move(buffer_uses),
       buffer_sizes = std::move(buffer_sizes)](
          HloComputation* computation,
          const HloPoplarDataflowAnalysis& dataflow_analysis,
          const absl::flat_hash_map<const HloComputation*, int64_t>&
              memory_by_computation) -> StatusOr<HloInstructionSequence> {
        struct ScheduleResult {
          int64_t schedule_memory;
          std::string algorithm_name;
          HloInstructionSequence schedule;
        };
        absl::optional<ScheduleResult> minimum_memory_schedule;
        Status last_error = Status::OK();

        // TODO(T9495): Consider parallel execution.
        for (const auto& algorithm : algorithms) {
          CHECK(algorithm.function) << "Invalid function: " << algorithm.name;
          const auto schedule_or_status = algorithm.function(
              computation, dataflow_analysis, memory_by_computation);
          if (!schedule_or_status.ok()) {
            last_error = schedule_or_status.status();
            // Keep looking for an algorithm that can produce a schedule.
            continue;
          }
          const auto schedule = schedule_or_status.ValueOrDie();
          const int64_t schedule_memory =
              memory_estimator(schedule.instructions(), buffer_uses,
                               buffer_sizes, memory_by_computation);
          VLOG(2) << "Scheduler " << algorithm.name
                  << " produced a schedule for " << computation->name()
                  << " with estimated memory consumption " << schedule_memory;

          if (!minimum_memory_schedule.has_value() ||
              schedule_memory < minimum_memory_schedule->schedule_memory) {
            minimum_memory_schedule =
                ScheduleResult{schedule_memory, algorithm.name, schedule};
          }
        }

        // If no schedule was found, return the last error.
        if (!minimum_memory_schedule.has_value()) {
          CHECK(!last_error.ok());
          return last_error;
        }

        VLOG(3) << "Chosen scheduler for " << computation->name() << ": "
                << minimum_memory_schedule->algorithm_name;

        return minimum_memory_schedule->schedule;
      }};
}

}  // namespace

MemoryEstimator HeapMemoryEstimator() {
  return
      [](const std::vector<HloInstruction*>& schedule,
         const HloInstructionMap<HloPoplarBufferSet>& buffer_uses,
         const absl::flat_hash_map<HloPoplarBuffer::Id, int64_t>& buffer_sizes,
         const absl::flat_hash_map<const HloComputation*, int64_t>&
             memory_by_computation) {
        auto program_liveness = GenerateProgramLiveness(schedule, buffer_uses);
        return EstimateMinimumLiveMemory(program_liveness, buffer_sizes,
                                         memory_by_computation);
      };
}

ChooseBestIpuScheduler::ChooseBestIpuScheduler(
    const std::vector<NamedIpuSchedulerAlgorithm>& algorithms,
    const CompilerAnnotations* annotations, MemoryEstimator memory_estimator)
    : algorithms_(algorithms),
      annotations_(annotations),
      memory_estimator_(memory_estimator) {}

StatusOr<bool> ChooseBestIpuScheduler::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto analysis,
                      HloPoplarDataflowAnalysis::Run(module, annotations_));
  TF_ASSIGN_OR_RETURN(
      auto algorithm,
      BestIpuSchedule(module, *analysis, algorithms_, memory_estimator_));

  IpuScheduler scheduler(algorithm, annotations_);
  return scheduler.Run(module, std::move(analysis));
}

IpuScheduler::IpuScheduler(IpuSchedulerAlgorithm algorithm,
                           const CompilerAnnotations* annotations)
    : algorithm_(std::move(algorithm)), annotations_(annotations) {}

StatusOr<bool> IpuScheduler::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloPoplarDataflowAnalysis::Run(module, annotations_));
  return Run(module, std::move(dataflow_analysis));
}

StatusOr<bool> IpuScheduler::Run(
    HloModule* module,
    std::unique_ptr<HloPoplarDataflowAnalysis> dataflow_analysis) {
  if (!algorithm_) {
    return xla::FailedPrecondition("No IpuSchedulerAlgorithm provided");
  }

  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      IpuScheduleModule(module, algorithm_, dataflow_analysis.get()));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
