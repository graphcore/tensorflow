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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_IO_TILES_PLACER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_IO_TILES_PLACER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
class HloModule;
class CallGraph;

namespace poplarplugin {

/**
 * If enabled, this pass will place instructions that perform host IO
 * (host exchanges) on the IO tiles. This is done by setting the tileset
 * backend config on the instruction to TILESET_IO_TILES. A later pass
 * will then insert the necessary inter-tileset copies to/from IO tiles
 * between instructions on different tilesets.
 */
class IoTilesPlacer : public HloModulePass {
 public:
  explicit IoTilesPlacer(bool enabled) : enabled_(enabled) {}

  absl::string_view name() const override { return "io-tiles-placer"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* comp,
                                  const CallGraph& call_graph);

  bool enabled_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_IO_TILES_PLACER_H_
