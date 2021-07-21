/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/executable_build_options.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

ExecutableBuildOptions& ExecutableBuildOptions::set_device_allocator(
    se::DeviceMemoryAllocator* allocator) {
  device_allocator_ = allocator;
  return *this;
}

se::DeviceMemoryAllocator* ExecutableBuildOptions::device_allocator() const {
  return device_allocator_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_ordinal(
    int device_ordinal) {
  CHECK_GE(device_ordinal, 0);
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableBuildOptions::device_ordinal() const { return device_ordinal_; }

DebugOptions* ExecutableBuildOptions::mutable_debug_options() {
  if (!has_debug_options()) {
    debug_options_ = GetDebugOptionsFromFlags();
  }
  return &debug_options_.value();
}

ExecutableBuildOptions& ExecutableBuildOptions::set_result_layout(
    const Shape& shape_with_layout) {
  result_layout_set_ = true;
  result_layout_ = shape_with_layout;
  return *this;
}

const Shape* ExecutableBuildOptions::result_layout() const {
  return result_layout_set_ ? &result_layout_ : nullptr;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_replicas(
    int num_replicas) {
  num_replicas_ = num_replicas;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_partitions(
    int num_partitions) {
  num_partitions_ = num_partitions;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_use_spmd_partitioning(
    bool use_spmd_partitioning) {
  use_spmd_partitioning_ = use_spmd_partitioning;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_deduplicate_hlo(
    bool deduplicate_hlo) {
  deduplicate_hlo_ = deduplicate_hlo;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_assignment(
    const DeviceAssignment& device_assignment) {
  device_assignment_ = device_assignment;
  return *this;
}

string ExecutableBuildOptions::ToString() const {
  string result_layout = "nullopt";
  if (result_layout_set_) {
    result_layout = ShapeUtil::HumanStringWithLayout(result_layout_);
  }
  return absl::StrFormat(
      "ExecutableBuildOptions{device_ordinal=%d, result_layout=%s, "
      "num_replicas=%d}",
      device_ordinal_, result_layout, num_replicas_);
}

ExecutableBuildOptions& ExecutableBuildOptions::set_argument_input_indices(
    const std::vector<int>& argument_input_indices) {
  argument_input_indices_ = argument_input_indices;
  return *this;
}

std::vector<int> ExecutableBuildOptions::argument_input_indices() const {
  return argument_input_indices_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_resource_input_indices(
    const std::vector<int>& resource_input_indices) {
  resource_input_indices_ = resource_input_indices;
  return *this;
}

std::vector<int> ExecutableBuildOptions::resource_input_indices() const {
  return resource_input_indices_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_resource_input_initialized(
    const std::vector<bool>& resource_input_initialized) {
  resource_input_initialized_ = resource_input_initialized;
  return *this;
}

std::vector<bool> ExecutableBuildOptions::resource_input_initialized() const {
  return resource_input_initialized_;
}

ExecutableBuildOptions&
ExecutableBuildOptions::set_resource_update_to_input_index(
    const std::vector<int>& resource_update_to_input_index) {
  std::copy(resource_update_to_input_index.begin(),
            resource_update_to_input_index.end(),
            std::back_inserter(resource_update_to_input_index_));
  return *this;
}

std::vector<int> ExecutableBuildOptions::resource_update_to_input_index()
    const {
  return resource_update_to_input_index_;
}

}  // namespace xla
