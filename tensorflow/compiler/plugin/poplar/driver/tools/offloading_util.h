/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_OFFLOADING_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_OFFLOADING_UTIL_H_

#include <string>

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
class HloDataflowAnalysis;
class HloInstruction;
namespace poplarplugin {
std::string GetReplicatedParameterLoadFusionName();
std::string GetReplicatedParameterStoreFusionName();
const Shape GetReplicatedParameterLoadFusionAllGatherShape(
    const HloInstruction*);
bool IsReplicatedParameterLoadFusion(const HloInstruction*);
bool IsReplicatedParameterStoreFusion(const HloInstruction*);
bool IsReplicatedParameterLoad(const HloInstruction*);
bool IsReplicatedParameterStore(const HloInstruction*);
// Given an instruction, get its load and store users.
Status GetRemoteLoadStoreUsers(HloInstruction* inst, HloInstruction** load,
                               HloInstruction** store);

int64_t PartitionedElementCountPerReplica(int64_t element_count,
                                          int64_t partition_replication_factor);

std::size_t PartitionedByteCountPerReplica(
    std::size_t byte_count, PrimitiveType element_type,
    int64_t partition_replication_factor);

StatusOr<int64_t> GetRemoteBufferEntryParameterNumber(
    const HloInstruction* inst);
StatusOr<int64_t> GetRemoteBufferEntryParameterNumber(
    const HloDataflowAnalysis& dfa, const HloInstruction* inst);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_OFFLOADING_UTIL_H_
