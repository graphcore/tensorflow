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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

namespace xla {
namespace poplarplugin {

/*static*/ PoplarReplicaGroups PoplarReplicaGroups::Consecutive(
    uint64 group_size) {
  // Treat 0 is a special case meaning all replicas in the same group.
  if (group_size == 0) {
    return PoplarReplicaGroups();
  }
  return PoplarReplicaGroups(group_size, Type::Consecutive);
}

/*static*/ PoplarReplicaGroups PoplarReplicaGroups::Orthogonal(
    uint64 group_size) {
  // Treat 0 is a special case meaning all replicas in the same group.
  if (group_size == 0) {
    return PoplarReplicaGroups();
  }
  return PoplarReplicaGroups(group_size, Type::Orthogonal);
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
  switch (group_type_) {
    case Type::Consecutive:
      return "consecutive(group_size=" + std::to_string(*group_size_) + ")";
    case Type::Orthogonal:
      return "orthogonal(group_size=" + std::to_string(*group_size_) + ")";
  }
  LOG(FATAL) << "Unknown group type";
}

std::vector<xla::ReplicaGroup> PoplarReplicaGroups::ToXlaReplicaGroups() const {
  if (!group_size_.has_value()) {
    return {};
  }

  // We do not know the total number of replicas at this point, so we do not
  // know how many groups to generate. Therefore we generate only enough groups
  // to satisfy the HLO verifier. During lowering the correct number of groups
  // will be used based on the total number of replicas.
  const int64 group_size = *group_size_;
  const int64 num_groups = group_type_ == Type::Consecutive ? 1 : 2;

  std::vector<xla::ReplicaGroup> result(num_groups);
  for (int64 i = 0; i < num_groups; ++i) {
    for (int64 j = 0; j < group_size; ++j) {
      result[i].add_replica_ids(j * num_groups + i);
    }
  }
  return result;
}

/*static*/ xla::StatusOr<PoplarReplicaGroups>
PoplarReplicaGroups::FromXlaReplicaGroups(
    absl::Span<const xla::ReplicaGroup> groups) {
  if (groups.empty()) {
    return PoplarReplicaGroups();
  }

  const int64 num_groups = groups.size();
  const int64 group_size = groups[0].replica_ids_size();
  if (group_size == 0) {
    return xla::InvalidArgument("Unsupported empty replica group");
  }

  for (int64 i = 0; i < num_groups; ++i) {
    const xla::ReplicaGroup& group = groups[i];
    if (group.replica_ids_size() != group_size) {
      return xla::InvalidArgumentStrCat(
          "Irregular replica group size: Expected ", group_size, ", actual ",
          group.replica_ids_size());
    }

    for (int64 j = 0; j < group_size; ++j) {
      const int64 expected = j * num_groups + i;
      const int64 actual = group.replica_ids(j);
      if (expected != actual) {
        return xla::InvalidArgumentStrCat(
            "Unsupported replica group: Expected ", expected, " at index ", j,
            ", actual ", actual, ": ", group.DebugString());
      }
    }
  }

  switch (num_groups) {
    case 1:
      return Consecutive(group_size);
    case 2:
      return Orthogonal(group_size);
  }
  return xla::InvalidArgumentStrCat("Unsupported number of replica groups: ",
                                    num_groups);
}

bool PoplarReplicaGroups::operator==(const PoplarReplicaGroups& other) const {
  return group_size_ == other.group_size_ && group_type_ == other.group_type_;
}

bool PoplarReplicaGroups::operator!=(const PoplarReplicaGroups& other) const {
  return !(*this == other);
}

size_t PoplarReplicaGroups::Hash() const {
  return hash_util::hash(group_size_, group_type_);
}

PoplarReplicaGroups::Type PoplarReplicaGroups::GroupType() const {
  return group_type_;
}

PoplarReplicaGroups::PoplarReplicaGroups(uint64 group_size, Type group_type)
    : group_size_(group_size), group_type_(group_type) {
  // 0 should use default constructor instead.
  CHECK_NE(group_size, 0);
}

std::ostream& operator<<(std::ostream& oss, const PoplarReplicaGroups& groups) {
  return oss << groups.ToString();
}

}  // namespace poplarplugin
}  // namespace xla
