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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_REPLICA_GROUPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_REPLICA_GROUPS_H_

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace poplarplugin {

class PoplarReplicaGroups {
 public:
  // Create a default instance with all the replicas in a single group.
  PoplarReplicaGroups() = default;

  /* The replicas are divided consecutively into groups of the given size.
   * If there are N replicas denoted {0, ... N-1} and group size is k then the
   * groups are:
   *   {0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}
   */
  static PoplarReplicaGroups Consecutive(uint64 group_size);

  uint64 GroupSizeOr(uint64 default_value) const;
  uint64 GroupSizeOrDie() const;

  std::string ToString() const;

  /* Convert this replica group representation to the XLA representation.
   * The main idea is that it should be a valid XLA representation and that
   * FromXlaReplicaGroups(x.ToXlaReplicaGroups()) == x.
   */
  std::vector<xla::ReplicaGroup> ToXlaReplicaGroups() const;

  static xla::StatusOr<PoplarReplicaGroups> FromXlaReplicaGroups(
      absl::Span<const xla::ReplicaGroup> groups);

  bool operator==(const PoplarReplicaGroups& other) const;
  bool operator!=(const PoplarReplicaGroups& other) const;

  size_t Hash() const;

 private:
  explicit PoplarReplicaGroups(uint64 group_size);

  absl::optional<uint64> group_size_;
};

}  // namespace poplarplugin
}  // namespace xla

namespace std {
template <>
struct hash<xla::poplarplugin::PoplarReplicaGroups> {
  size_t operator()(const xla::poplarplugin::PoplarReplicaGroups& groups) {
    return groups.Hash();
  }
};
}  // namespace std

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_POPLAR_REPLICA_GROUPS_H_
