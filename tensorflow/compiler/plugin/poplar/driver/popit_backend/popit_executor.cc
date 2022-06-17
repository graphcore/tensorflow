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
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_executor.h"

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_stream.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

#define NOT_IMPLEMENTED LOG(FATAL) << "Popit Not implemented " << __LINE__;

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

bool HasIpuHardware() {
  auto device_manager = poplar::DeviceManager::createDeviceManager();
  auto device_list = device_manager.getDevices();
  for (const auto& d : device_list) {
    if (d.getTarget().getTargetType() == poplar::TargetType::IPU) {
      return true;
    }
  }
  return false;
}

Status ValidTargetOptions(const IpuOptions& options, int ordinal) {
  if (options.device_config_size() > 0 &&
      ordinal >= options.device_config_size()) {
    return InternalError("Device ordinal %d not in device configuration list.",
                         ordinal);
  }
  return Status::OK();
}

poplar::Target CreateIPUModelTarget(const IpuOptions& options, int num_ipus) {
  const auto& model_config = options.ipu_model_config();
  const auto version_string = model_config.ipu_model_version();
  poplar::IPUModel model(version_string.c_str());

  model.numIPUs = num_ipus;
  model.compileIPUCode = model_config.compile_ipu_code();
  if (model_config.tiles_per_ipu() > 0) {
    model.tilesPerIPU = model_config.tiles_per_ipu();
  } else if (PoplarXlaFlags::Get().ipu_model_tiles > 0) {
    model.tilesPerIPU = PoplarXlaFlags::Get().ipu_model_tiles;
  }
  auto device = model.createDevice();
  return device.getTarget();
}

poplar::Target CreateTargetWithNIPUs(int num_ipus) {
  // TODO(samuelh) want to extend this to support all systems
  // think easy way to get this string might just be
  // to get any device from the device manager and then just
  // return the system string from that device
  return poplar::Target::createIPUTarget(num_ipus, "IPU-POD16");
}

StatusOr<poplar::Target> CreatePoplarTarget(
    const absl::optional<IpuOptions>& ipu_options, int ordinal) {
  if (!ipu_options) {
    // By default we will make eager mode deal with single ipu
    // devices
    return CreateTargetWithNIPUs(1);
  }
  const auto& options = *ipu_options;
  TF_RETURN_IF_ERROR(ValidTargetOptions(options, ordinal));
  if (PoplarXlaFlags::Get().use_ipu_model) {
    // only create single ipu model targets
    return CreateIPUModelTarget(options, 1);
  }
  const bool has_config = options.device_config_size() > 0;
  if (!has_config) {
    return CreateTargetWithNIPUs(1);
  }
  auto device_config = options.device_config(ordinal);
  if (device_config.selection_case() ==
      IpuOptions::DeviceConfig::SelectionCase::kAutoCount) {
    return CreateTargetWithNIPUs(device_config.auto_count());
  }
  // create a device to match the hardware for this ordinal
  CHECK(HasIpuHardware());
  auto device_manager = poplar::DeviceManager::createDeviceManager();
  auto device_list = device_manager.getDevices();
  auto& device = device_list.at(device_config.cfg_index());

  return device.getTarget();
}

unsigned GetSessionReplicationFactor(
    const absl::optional<IpuOptions>& ipu_options) {
  if (!ipu_options) {
    return 1;
  }
  // TODO(samuelh) this isn't right but I think it is really ridiculous
  // to have the session need a replication factor;
  return ipu_options->multi_replica_process_count();
}

absl::InlinedVector<popitMemSpaceDesc, 2> CreateMemSpaces(
    const absl::optional<IpuOptions>& ipu_options,
    const poplar::Target& target) {
  if (!ipu_options) {
    return {
        {popitMemSpaceType_t::POPIT_MEMSPACE_GENERAL, target.getNumTiles()}};
  }
  absl::InlinedVector<popitMemSpaceDesc, 2> result;
  for (int64_t ipu = 0; ipu < target.getNumIPUs(); ++ipu) {
    int64_t num_io_tiles = ipu_options->num_io_tiles();
    result.emplace_back(
        popitMemSpaceDesc{popitMemSpaceType_t::POPIT_MEMSPACE_GENERAL,
                          target.getTilesPerIPU() - num_io_tiles});
    if (num_io_tiles) {
      result.emplace_back(popitMemSpaceDesc{
          popitMemSpaceType_t::POPIT_MEMSPACE_IO, num_io_tiles});
    }
  }
  return result;
}

Status PopItExecutor::Init(int device_ordinal,
                           se::DeviceOptions device_options) {
  TF_ASSIGN_OR_RETURN(const auto target,
                      CreatePoplarTarget(ipu_options_, device_ordinal));
  auto mem_spaces = CreateMemSpaces(ipu_options_, target);
  auto session_ptr = popitCreateSession(
      reinterpret_cast<const poplarTarget_t*>(&target),
      GetSessionReplicationFactor(ipu_options_), mem_spaces.data(),
      static_cast<unsigned>(mem_spaces.size()));
  session_ = SessionType(session_ptr);
  return Status::OK();
}

se::DeviceMemoryBase PopItExecutor::Allocate(uint64 size,
                                             int64_t memory_space) {
  NOT_IMPLEMENTED;
}
void* PopItExecutor::GetSubBuffer(se::DeviceMemoryBase* parent, uint64 offset,
                                  uint64 size) {
  NOT_IMPLEMENTED;
}
void PopItExecutor::Deallocate(se::DeviceMemoryBase* mem) { NOT_IMPLEMENTED; }
void* PopItExecutor::HostMemoryAllocate(uint64 size) { NOT_IMPLEMENTED; }
void PopItExecutor::HostMemoryDeallocate(void* mem) { NOT_IMPLEMENTED; }
bool PopItExecutor::HostMemoryRegister(void* mem, uint64 size) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::HostMemoryUnregister(void* mem) { NOT_IMPLEMENTED; }
bool PopItExecutor::SynchronizeAllActivity() { NOT_IMPLEMENTED; }
Status PopItExecutor::SynchronousMemZero(se::DeviceMemoryBase* location,
                                         uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::SynchronousMemSet(se::DeviceMemoryBase* location,
                                        int value, uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::SynchronousMemcpy(se::DeviceMemoryBase* gpu_dst,
                                        const void* host_src, uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::SynchronousMemcpy(void* host_dst,
                                        const se::DeviceMemoryBase& gpu_src,
                                        uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::SynchronousMemcpyDeviceToDevice(
    se::DeviceMemoryBase* gpu_dst, const se::DeviceMemoryBase& gpu_src,
    uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::MemZero(se::Stream* stream,
                              se::DeviceMemoryBase* location, uint64 size) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::Memset32(se::Stream* stream,
                               se::DeviceMemoryBase* location, uint32 pattern,
                               uint64 size) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::Memcpy(se::Stream* stream, void* host_dst,
                           const se::DeviceMemoryBase& gpu_src, uint64 size) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* gpu_dst,
                           const void* host_src, uint64 size) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::MemcpyDeviceToDevice(se::Stream* stream,
                                         se::DeviceMemoryBase* gpu_dst,
                                         const se::DeviceMemoryBase& gpu_src,
                                         uint64 size) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::HostCallback(se::Stream* stream,
                                 std::function<void()> callback) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::HostCallback(se::Stream* stream,
                                 std::function<Status()> callback) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::AllocateEvent(se::Event* event) { NOT_IMPLEMENTED; }
Status PopItExecutor::DeallocateEvent(se::Event* event) { NOT_IMPLEMENTED; }
Status PopItExecutor::RecordEvent(se::Stream* stream, se::Event* event) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::WaitForEvent(se::Stream* stream, se::Event* event) {
  NOT_IMPLEMENTED;
}
se::Event::Status PopItExecutor::PollForEventStatus(se::Event* event) {
  NOT_IMPLEMENTED;
}
// If we decide to put the popitSession inside the stream then this
// will create the session but I think the session belongs in the executor
bool PopItExecutor::AllocateStream(se::Stream* stream) { return true; }
void PopItExecutor::DeallocateStream(se::Stream* stream) {}
bool PopItExecutor::CreateStreamDependency(se::Stream* dependent,
                                           se::Stream* other) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::AllocateTimer(se::Timer* timer) { NOT_IMPLEMENTED; }
void PopItExecutor::DeallocateTimer(se::Timer* timer) { NOT_IMPLEMENTED; }
bool PopItExecutor::StartTimer(se::Stream* stream, se::Timer* timer) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::StopTimer(se::Stream* stream, se::Timer* timer) {
  NOT_IMPLEMENTED;
}
Status PopItExecutor::BlockHostUntilDone(se::Stream* stream) {
  NOT_IMPLEMENTED;
}
int PopItExecutor::PlatformDeviceCount() { NOT_IMPLEMENTED; }
Status PopItExecutor::EnablePeerAccessTo(
    se::internal::StreamExecutorInterface* other) {
  NOT_IMPLEMENTED;
}
bool PopItExecutor::CanEnablePeerAccessTo(
    se::internal::StreamExecutorInterface* other) {
  NOT_IMPLEMENTED;
}
StatusOr<std::unique_ptr<se::DeviceDescription>>
PopItExecutor::CreateDeviceDescription() const {
  auto platform = se::MultiPlatformManager::PlatformWithName(
      tensorflow::POPIT_PLATFORM_NAME);
  if (platform.ok()) {
    auto* p = static_cast<PopItPlatform*>(platform.ValueOrDie());
    return p->DescriptionForDevice(0);
  }
  return InternalError("Failed to create device description.");
}
std::unique_ptr<se::internal::EventInterface>
PopItExecutor::CreateEventImplementation() {
  NOT_IMPLEMENTED;
}
std::unique_ptr<se::internal::KernelInterface>
PopItExecutor::CreateKernelImplementation() {
  NOT_IMPLEMENTED;
}
std::unique_ptr<se::internal::StreamInterface>
PopItExecutor::GetStreamImplementation() {
  return absl::make_unique<PopItStream>();
}
std::unique_ptr<se::internal::TimerInterface>
PopItExecutor::GetTimerImplementation() {
  NOT_IMPLEMENTED;
}

}  // namespace poplarplugin
}  // namespace xla

#undef NOT_IMPLEMENTED
