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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable_cache.h"

#include <memory>
#include <mutex>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_hash.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace poplarplugin {

std::mutex& PoplarExecutableCache::Entry::GetCompilationMutex() {
  return compilation_mu_;
}

Status PoplarExecutableCache::Entry::SetPoplarExecutableCore(
    std::shared_ptr<PoplarExecutableCore>& executable_core) {
  CHECK(executable_core_.expired());
  executable_core_ = executable_core;
  return Status::OK();
}

std::shared_ptr<PoplarExecutableCore>
PoplarExecutableCache::Entry::TryGetPoplarExecutableCore() const {
  return executable_core_.lock();
}

PoplarExecutableCache& PoplarExecutableCache::GetInstance() {
  static PoplarExecutableCache cache;
  return cache;
}

StatusOr<std::unique_ptr<PoplarExecutable>>
PoplarExecutableCache::GetOrCompileExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    PoplarExecutor* executor, CompileFunction compile_fn) {
  const uint64 executable_hash = tensorflow::Hash64Combine(
      tensorflow::Hash64Combine(HloHash(hlo_module.get()).GetHash(),
                                executor->GetPoplarDeviceHash()),
      executor->device_ordinal());

  // Get the correct entry in the cache.
  PoplarExecutableCache::Entry* entry;
  {
    std::lock_guard<std::mutex> cache_guard(cache_mu_);
    auto& e = cache_[executable_hash];
    if (!e) {
      VLOG(2) << "Creating a new cache entry.";
      e.reset(new PoplarExecutableCache::Entry());
    }
    entry = e.get();
  }

  std::shared_ptr<PoplarExecutableCore> executable_core =
      entry->TryGetPoplarExecutableCore();

  if (!executable_core) {
    // The core was not compiled/it was deallocated - the core needs to be
    // recompiled.
    std::lock_guard<std::mutex> guard(entry->GetCompilationMutex());
    // Lock the compilation mutex and try to get the core again. If it is not
    // populated then the thread with the lock will compile it and all other
    // threads waiting on the lock will be able to get a shared reference after.
    executable_core = entry->TryGetPoplarExecutableCore();
    if (!executable_core) {
      VLOG(2) << "Module " << hlo_module->name()
              << " has not been compiled yet (Hash: 0x" << std::hex
              << executable_hash << ").";
      TF_ASSIGN_OR_RETURN(auto compiled, compile_fn(hlo_module.get(), executor,
                                                    executable_hash));
      executable_core = std::move(compiled);
      TF_RETURN_IF_ERROR(entry->SetPoplarExecutableCore(executable_core));
    }
  }
  CHECK(executable_core);

  return absl::make_unique<PoplarExecutable>(
      std::move(hlo_module), std::move(hlo_profile_printer),
      std::move(hlo_profile_index_map), std::move(executable_core));
}

}  // namespace poplarplugin
}  // namespace xla
