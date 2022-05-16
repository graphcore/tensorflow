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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SLICE_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SLICE_UTIL_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "absl/types/span.h"

namespace xla {

class HloInstruction;
class HloDynamicIndexInstruction;
class HloDynamicSliceInstruction;
class HloDynamicUpdateSliceInstruction;

namespace poplarplugin {
// Utility type for grouping an appropriately setup
// dynamic_update/add/dynamic_slice together as a DynamicUpdateAdd;
struct DynamicUpdateAdd {
  static bool IsDynamicUpdateAdd(
      const HloDynamicUpdateSliceInstruction* dynamic_update);

  explicit DynamicUpdateAdd(HloDynamicUpdateSliceInstruction* dynamic_update);

  HloDynamicUpdateSliceInstruction* update;
  HloInstruction* add;
  HloDynamicSliceInstruction* slice;
};

// Try and replace dynamic-slice with multi-slice instructions, since
// multi-slice can be planned and dynamic-slice can't be. Returns multi-slice
// instruction or nullptr if no replacement occurred.
StatusOr<HloInstruction*> TryReplaceDynamicSliceWithMultiSlice(
    HloDynamicSliceInstruction* dynamic_slice);
StatusOr<HloInstruction*> TryReplaceDynamicUpdateWithMultiUpdate(
    HloDynamicUpdateSliceInstruction* dynamic_update);
StatusOr<HloInstruction*> TryReplaceDynamicUpdateAddWithMultiUpdateAdd(
    DynamicUpdateAdd dynamic_update_add);

// Helper for Dynamic(Update)Slice where we recognize dynamic and constant slice
// dimensions.
struct DynamicSliceHelper {
  explicit DynamicSliceHelper(const HloDynamicIndexInstruction* inst);
  DynamicSliceHelper() = delete;

  SliceInfo dynamic_slice_info;
  SliceInfo constant_slice_info;

  bool has_constant_slice = false;
  bool has_dynamic_slice = false;
};

bool Is1DSliceInFirstDimension(const HloInstruction* slice);

// Reduce multidimensional indices into one-dimensional indices for a
// flattened version of the shape being indexed.
//   indices: The index tensor beign flattened.
//   dim: The dimension of indices to reduce.
//   sizes: The sizes of the dimensions being indexed into.
StatusOr<HloInstruction*> ReduceIndices(HloInstruction* indices, int64 dim,
                                        absl::Span<const int64> sizes);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SLICE_UTIL_H_
