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
#include <memory>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
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

StatusOr<HloSchedule> IpuScheduleModule(
    HloModule* module, const LogicalBuffer::SizeFunction& size_function,
    const IpuSchedulerAlgorithm& algorithm,
    const CompilerAnnotations* annotations) {
  HloSchedule schedule(module);
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloPoplarDataflowAnalysis::Run(module, annotations));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  absl::flat_hash_map<const HloComputation*, int64> memory_by_computation;
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (!computation->IsFusionComputation()) {
      TF_ASSIGN_OR_RETURN(
          HloInstructionSequence computation_sequence,
          algorithm(computation, *dataflow_analysis, memory_by_computation));
      TF_ASSIGN_OR_RETURN(
          auto bytes, HeapSimulator::MinimumMemoryForComputation(
                          *computation, computation_sequence, *alias_analysis,
                          size_function, &memory_by_computation));

      memory_by_computation[computation] = bytes;
      schedule.set_sequence(computation, std::move(computation_sequence));
    }
  }
  VLOG(2) << "Module schedule:\n" << schedule;

  TF_RETURN_IF_ERROR(schedule.Verify());

  return std::move(schedule);
}

/*
 * The scheduler is run for each computation, while the alias analysis is
 * performed for the whole module. This cache allows re-using the analysis
 * across the computations to save time. It assumes that the module is not
 * modified while in use.
 */
class AliasAnalysisCache {
 public:
  using Container =
      absl::flat_hash_map<const HloModule*, std::unique_ptr<HloAliasAnalysis>>;

  const HloAliasAnalysis& FindOrRun(const HloModule* module) {
    const auto it = cache_.lazy_emplace(
        module, [module](const Container::constructor& ctor) {
          auto result = HloAliasAnalysis::Run(module);
          TF_CHECK_OK(result.status());
          ctor(module, std::move(result.ValueOrDie()));
        });
    return *it->second;
  }

 private:
  Container cache_;
};

}  // namespace

MemoryEstimator HeapMemoryEstimator() {
  using HeapSimulatorOverload = StatusOr<int64> (*)(
      const HloComputation& computation, const HloInstructionSequence& sequence,
      const HloAliasAnalysis& alias_analysis,
      const LogicalBuffer::SizeFunction& size_function,
      const absl::flat_hash_map<const HloComputation*, int64>*);

  return static_cast<HeapSimulatorOverload>(
      HeapSimulator::MinimumMemoryForComputation);
}

StatusOr<IpuSchedulerAlgorithm> BestIpuSchedule(
    const LogicalBuffer::SizeFunction& size_function,
    std::vector<NamedIpuSchedulerAlgorithm> algorithms,
    MemoryEstimator memory_estimator) {
  if (algorithms.empty()) {
    return xla::FailedPrecondition(
        "Cannot construct BestIpuSchedule when the input is empty");
  }

  return IpuSchedulerAlgorithm{
      [algorithms = std::move(algorithms),
       alias_analysis_cache = std::make_shared<AliasAnalysisCache>(),
       size_function = size_function, memory_estimator = memory_estimator](
          HloComputation* computation,
          const HloPoplarDataflowAnalysis& dataflow_analysis,
          const absl::flat_hash_map<const HloComputation*, int64>&
              memory_by_computation) -> StatusOr<HloInstructionSequence> {
        const HloAliasAnalysis& alias_analysis =
            alias_analysis_cache->FindOrRun(computation->parent());

        struct ScheduleResult {
          int64 schedule_memory;
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

          // TODO(T9494): Replace the heap simulator.
          TF_ASSIGN_OR_RETURN(
              const int64 schedule_memory,
              memory_estimator(*computation, schedule, alias_analysis,
                               size_function, &memory_by_computation));

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

IpuScheduler::IpuScheduler(const LogicalBuffer::SizeFunction& size_function,
                           IpuSchedulerAlgorithm algorithm,
                           const CompilerAnnotations* annotations)
    : size_function_(size_function),
      algorithm_(std::move(algorithm)),
      annotations_(annotations) {}

StatusOr<bool> IpuScheduler::Run(HloModule* module) {
  if (!algorithm_) {
    return xla::FailedPrecondition("No IpuSchedulerAlgorithm provided");
  }

  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      IpuScheduleModule(module, size_function_, algorithm_, annotations_));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
