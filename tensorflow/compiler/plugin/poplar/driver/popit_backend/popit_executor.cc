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

#include "tensorflow/compiler/plugin/poplar/driver/popit_backend/popit_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

#define NOT_IMPLEMENTED LOG(FATAL) << "Popit Not implemented " << __LINE__;

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

Status PopItExecutor::Init(int device_ordinal,
                           se::DeviceOptions device_options) {
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
bool PopItExecutor::AllocateStream(se::Stream* stream) { NOT_IMPLEMENTED; }
void PopItExecutor::DeallocateStream(se::Stream* stream) { NOT_IMPLEMENTED; }
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
  NOT_IMPLEMENTED;
}
std::unique_ptr<se::internal::TimerInterface>
PopItExecutor::GetTimerImplementation() {
  NOT_IMPLEMENTED;
}

}  // namespace poplarplugin
}  // namespace xla

#undef NOT_IMPLEMENTED
