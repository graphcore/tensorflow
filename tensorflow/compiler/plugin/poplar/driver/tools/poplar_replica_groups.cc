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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"

namespace xla {
namespace poplarplugin {

/*static*/ PoplarReplicaGroups PoplarReplicaGroups::Consecutive(
    uint64 group_size) {
  // Treat 0 is a special case meaning all replicas in the same group.
  if (group_size == 0) {
    return PoplarReplicaGroups();
  }
  return PoplarReplicaGroups(group_size);
}

uint64 PoplarReplicaGroups::GroupSizeOr(uint64 default_value) const {
  return group_size_.value_or(default_value);
}

uint64 PoplarReplicaGroups::GroupSizeOrDie() const {
  CHECK(group_size_.has_value());
  return *group_size_;
}

std::string PoplarReplicaGroups::ToString() const {
  if (!group_size_.has_value()) {
    return "single(group_size=all)";
  }
  return "consecutive(group_size=" + std::to_string(*group_size_) + ")";
}

std::vector<xla::ReplicaGroup> PoplarReplicaGroups::ToXlaReplicaGroups() const {
  if (!group_size_.has_value()) {
    return {};
  }

  // We always create a single replica group here as we do not know the total
  // number of replicas. However, we do know that there should be at least one
  // group of the given size. During lowering multiple groups will be used if
  // the total number of replicas is larger than the group size.
  xla::ReplicaGroup group;
  for (int64 i = 0; i < group_size_; ++i) {
    group.add_replica_ids(i);
  }
  return {group};
}

/*static*/ xla::StatusOr<PoplarReplicaGroups>
PoplarReplicaGroups::FromXlaReplicaGroups(
    absl::Span<const xla::ReplicaGroup> groups) {
  if (groups.empty()) {
    return PoplarReplicaGroups();
  }

  // We expect a single consecutive replica group.
  if (groups.size() != 1) {
    return xla::InvalidArgumentStrCat(
        "Expected a single replica group, but got ", groups.size());
  }

  const auto& group = groups[0];
  const int64 group_size = group.replica_ids_size();
  for (int64 i = 0; i < group_size; ++i) {
    if (group.replica_ids(i) != i) {
      return xla::InvalidArgumentStrCat(
          "Unsupported non-consecutive replica group: ", group.DebugString());
    }
  }

  return Consecutive(group_size);
}

bool PoplarReplicaGroups::operator==(const PoplarReplicaGroups& other) const {
  return group_size_ == other.group_size_;
}

bool PoplarReplicaGroups::operator!=(const PoplarReplicaGroups& other) const {
  return !(*this == other);
}

size_t PoplarReplicaGroups::Hash() const {
  return std::hash<decltype(group_size_)>()(group_size_);
}

PoplarReplicaGroups::PoplarReplicaGroups(uint64 group_size)
    : group_size_(group_size) {
  // 0 should use default constructor instead.
  CHECK_NE(group_size, 0);
}

}  // namespace poplarplugin
}  // namespace xla
