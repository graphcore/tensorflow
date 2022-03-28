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
#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_transfer_manager.h"

namespace xla {
namespace poplarplugin {

PopItTransferManager::PopItTransferManager()
    : GenericTransferManager(kPopItPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreatePopItTransferManager() {
  return absl::make_unique<xla::poplarplugin::PopItTransferManager>();
}

static bool InitPopItModule() {
  xla::TransferManager::RegisterTransferManager(
      xla::poplarplugin::kPopItPlatformId, &CreatePopItTransferManager);
  return true;
}

static bool popit_module_initialized = InitPopItModule();
