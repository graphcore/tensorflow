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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_INTER_TILESET_COPY_INSERTER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_INTER_TILESET_COPY_INSERTER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * This pass inserts copy instructions between instructions that are placed on
 * different tilesets (i.e. to/from IO tiles). This is necessary since each
 * tileset is a disjoint Poplar virtual graph, and a tensor mapped to a
 * different virtual graph (using non-overlapping tiles) cannot be accessed
 * directly, and must instead be copied first.
 */
class InterTilesetCopyInserter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "inter-tileset-copy-inserter";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_INTER_TILESET_COPY_INSERTER_H_
