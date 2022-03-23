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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SHORTEST_PATH_SCHEDULER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SHORTEST_PATH_SCHEDULER_H_

#include "tensorflow/compiler/plugin/poplar/driver/schedulers/ipu_scheduler.h"

namespace xla {
namespace poplarplugin {

struct CompilerInformation;

// Scheduler will label each node with its distance from the root.
// It will schedule the graph giving priority to the nodes nearest the root.
// For parameter nodes it will schedule them only when there are no other types
// of node to schedule, and then it will schedule the ones furthest away from
// the root.
IpuSchedulerAlgorithm CreateShortestPathScheduler(
    const CompilerInformation& information);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_SCHEDULERS_SHORTEST_PATH_SCHEDULER_H_
