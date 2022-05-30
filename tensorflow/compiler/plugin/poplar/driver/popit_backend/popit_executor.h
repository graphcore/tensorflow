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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_EXECUTOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_EXECUTOR_H_

#include <memory>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = stream_executor;

namespace xla {
namespace poplarplugin {

class PopItExecutor : public se::internal::StreamExecutorInterface {
 public:
  Status Init(int device_ordinal, se::DeviceOptions device_options) override;
  se::DeviceMemoryBase Allocate(uint64 size, int64_t memory_space) override;
  void* GetSubBuffer(se::DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override;
  void Deallocate(se::DeviceMemoryBase* mem) override;
  void* HostMemoryAllocate(uint64 size) override;
  void HostMemoryDeallocate(void* mem) override;
  bool HostMemoryRegister(void* mem, uint64 size) override;
  bool HostMemoryUnregister(void* mem) override;
  bool SynchronizeAllActivity() override;
  Status SynchronousMemZero(se::DeviceMemoryBase* location,
                            uint64 size) override;
  Status SynchronousMemSet(se::DeviceMemoryBase* location, int value,
                           uint64 size) override;
  Status SynchronousMemcpy(se::DeviceMemoryBase* gpu_dst, const void* host_src,
                           uint64 size) override;
  Status SynchronousMemcpy(void* host_dst, const se::DeviceMemoryBase& gpu_src,
                           uint64 size) override;
  Status SynchronousMemcpyDeviceToDevice(se::DeviceMemoryBase* gpu_dst,
                                         const se::DeviceMemoryBase& gpu_src,
                                         uint64 size) override;
  Status MemZero(se::Stream* stream, se::DeviceMemoryBase* location,
                 uint64 size) override;
  Status Memset32(se::Stream* stream, se::DeviceMemoryBase* location,
                  uint32 pattern, uint64 size) override;
  bool Memcpy(se::Stream* stream, void* host_dst,
              const se::DeviceMemoryBase& gpu_src, uint64 size) override;
  bool Memcpy(se::Stream* stream, se::DeviceMemoryBase* gpu_dst,
              const void* host_src, uint64 size) override;
  bool MemcpyDeviceToDevice(se::Stream* stream, se::DeviceMemoryBase* gpu_dst,
                            const se::DeviceMemoryBase& gpu_src,
                            uint64 size) override;
  bool HostCallback(se::Stream* stream, std::function<void()> callback);
  bool HostCallback(se::Stream* stream,
                    std::function<Status()> callback) override;
  Status AllocateEvent(se::Event* event) override;
  Status DeallocateEvent(se::Event* event) override;
  Status RecordEvent(se::Stream* stream, se::Event* event) override;
  Status WaitForEvent(se::Stream* stream, se::Event* event) override;
  se::Event::Status PollForEventStatus(se::Event* event) override;
  bool AllocateStream(se::Stream* stream) override;
  void DeallocateStream(se::Stream* stream) override;
  bool CreateStreamDependency(se::Stream* dependent,
                              se::Stream* other) override;
  bool AllocateTimer(se::Timer* timer) override;
  void DeallocateTimer(se::Timer* timer) override;
  bool StartTimer(se::Stream* stream, se::Timer* timer) override;
  bool StopTimer(se::Stream* stream, se::Timer* timer) override;
  Status BlockHostUntilDone(se::Stream* stream) override;
  int PlatformDeviceCount() override;
  Status EnablePeerAccessTo(
      se::internal::StreamExecutorInterface* other) override;
  bool CanEnablePeerAccessTo(
      se::internal::StreamExecutorInterface* other) override;
  StatusOr<std::unique_ptr<se::DeviceDescription>> CreateDeviceDescription()
      const override;
  std::unique_ptr<se::internal::EventInterface> CreateEventImplementation()
      override;
  std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation()
      override;
  std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation()
      override;
  std::unique_ptr<se::internal::TimerInterface> GetTimerImplementation()
      override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_POPIT_BACKEND_POPIT_EXECUTOR_H_
