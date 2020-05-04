/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"

#include <string.h>

#include <deque>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/StreamCallback.hpp>
#include <poplar/Tensor.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "google/protobuf/util/message_differencer.h"
#include "include/json/json.h"

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/send_recv_runtime_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"

/*
 * TensorControl is a structure that maintains state about the location
 * of a tensor - either on the device or cached on the host.
 *
 * Tensorflow/XLA assumes that a tensor is on the device when the device
 * allocator is called (PoplarExecutor::Allocate).  However, Poplar cannot
 * allocate tensors independently of the compiled Engine.  The TensorControl
 * structure tracks where the tensors are.
 *
 * TensorControl has three pieces of interacting state:
 *   on_device: This says whether the data is on the device (in one of the
 *              tensors belonging to the currently loaded engine).  When this
 *              is false, it means the data is being held in the host side
 *              buffer.
 *
 *   input_handle: If the tensor is on_device, and this is not -1, then it
 *                 indicates which of the input tensors of the current engine
 *                 contains the data.
 *
 *   output_handle: If the tensor is on_device, and this is not empty, then it
 *                  indicates which of the output tensors of the current
 *                  engine contains the data.
 *
 *   The states are:
 *     on_device=false :
 *       The data is in the host buffer.  If this buffer is passed as an
 *       argument when an engine is executed then it must be copied to the
 *       device.
 *
 *     on_device=true, input_handle not empty, output_handle is empty :
 *       During the previous engine execution, the data was copied to the
 *       device as one of the arguments.  On the next execution, if the engine
 *       does not change, and the argument index is the same, then the data
 *       does not need to be recopied to the device.  I suspect that this case
 *       is rare.
 *
 *     on_device=true, input_handle is empty, output_handle not empty :
 *       During the last execution, the buffer was allocated to represent one
 *       of the outputs of the engine.  If the host wants to read the data back
 *       then it will have to be retrieved from the device.  If the next
 *       execution changes the engine, then the data will have to be read back.
 *
 *     on_device=true, input_handle not empty, output_handle not empty :
 *       During the last execution, the buffer was an argument to the execution
 *       and was also one of the output parameters.  This typically indicates
 *       that it is a variable (weights/biases) that has been updated in place.
 *       If the next execution doesn't change the engine, and the data is not
 *       read back to the host in between executions, and the data remains as
 *       an argument to the same input number, then the data does not need to be
 *       copied back to the host.  This is the ideal situation when executing an
 *       engine repeatedly with the same set of weights/biases.
 *
 */
namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

std::string GetRandomNumberSeedStream() { return "__seed_stream"; }

std::string GetInfeedCopyHandle(const std::string& name, int64 shape_index) {
  return tensorflow::strings::Printf("infeed_%s.%lld", name.c_str(),
                                     shape_index);
}

std::string GetOutfeedCopyHandle(const std::string& name, int64 shape_index) {
  return tensorflow::strings::Printf("outfeed_%s.%lld", name.c_str(),
                                     shape_index);
}

se::host::HostStream* PoplarExecutor::AsPoplarStream(se::Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<se::host::HostStream*>(stream->implementation());
}

PoplarXfeedManager* GetXfeedManager(int device_ordinal) {
  static auto* managers = new absl::flat_hash_map<int, PoplarXfeedManager*>();
  static absl::Mutex* mutex = new absl::Mutex();

  absl::MutexLock lock(mutex);
  auto it = managers->find(device_ordinal);
  if (it == managers->end()) {
    it = managers->emplace(device_ordinal, new PoplarXfeedManager()).first;
  }
  return it->second;
}

void ResetXfeedManager(int device_ordinal) {
  auto* xfeed_manager = GetXfeedManager(device_ordinal);
  xfeed_manager->Reset();
}

namespace {
Status CreateDirIfMissing(const std::string& path) {
  CHECK(!path.empty());
  auto* env = tensorflow::Env::Default();

  // Two threads could race to observe the absence of the directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(path).ok()) {
    const auto status = env->RecursivelyCreateDir(path);
    if (!status.ok() && !env->IsDirectory(path).ok()) {
      return status;
    }
  }

  return Status::OK();
}

Shape GetOutfeedShape(const Shape& output_shape,
                      const uint32 replication_factor) {
  if (replication_factor > 1) {
    // When the graph is replicated, we expect an extra dimension at the front
    // of the output.
    std::vector<int64> dimensions = {replication_factor};
    absl::c_copy(output_shape.dimensions(), std::back_inserter(dimensions));
    return ShapeUtil::MakeShape(output_shape.element_type(), dimensions);
  } else {
    return output_shape;
  }
}

std::vector<Shape> GetOutfeedShapes(const std::vector<Shape>& output_shapes,
                                    const uint32 replication_factor) {
  std::vector<Shape> result(output_shapes.size());
  absl::c_transform(output_shapes, result.begin(), [&](const Shape& shape) {
    return GetOutfeedShape(shape, replication_factor);
  });
  return result;
}

int64 GetConfigHash(const IpuOptions& to_hash) {
  IpuOptions hashable_config = to_hash;

  // Remove elements which do not contribute to a difference in the
  // compiled executable.  We hash the device characteristics independently
  // so there is no need to do any device selection state.
  hashable_config.mutable_profiling()->set_enable_poplar_reports_text(false);
  hashable_config.mutable_profiling()->set_report_every_nth_execution(0);
  hashable_config.mutable_profiling()->set_enable_ipu_trace_events(false);
  hashable_config.mutable_profiling()->set_enable_poplar_reports_cbor(false);
  hashable_config.mutable_profiling()->set_report_directory(std::string());
  hashable_config.mutable_profiling()->set_max_report_size(0);
  hashable_config.mutable_device_config()->Clear();

  std::string config_proto_str;
  tensorflow::SerializeToStringDeterministic(hashable_config,
                                             &config_proto_str);
  return std::hash<string>()(config_proto_str);
}

std::vector<int64> GetGclHashes() {
  std::vector<int64> hashes;

  // Add a hash for each environment variable known to impact the GCL
  // compilation result.
  // TODO(T20018) - get this from GCL.
  for (const char* name :
       {"GCL_NUM_IO_TILES", "GCL_REAL_COLLECTIVES", "GCL_LIBRARY_PATH",
        "GCL_MAX_BYTES_PER_TILE", "GCL_GP_PATH"}) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      hashes.push_back(std::hash<string>()(value));
    }
  }

  return hashes;
}

int64 CombinedHash(const std::vector<int64>& components) {
  int64 hash = 42;
  for (int64 h : components) {
    hash = tensorflow::Hash64Combine(hash, h);
  }
  return hash;
}

bool HasIpuHardware() {
  auto device_list = PoplarExecutor::GetDeviceManager().getDevices();
  for (const auto& d : device_list) {
    if (d.getTarget().getTargetType() == poplar::TargetType::IPU) {
      return true;
    }
  }
  return false;
}

poplar::Target CreateIpuTarget(uint num_ipus, int64 ipu_version) {
  return poplar::Target::createIPUTarget(num_ipus,
                                         "ipu" + std::to_string(ipu_version));
}

}  // namespace

PoplarExecutor::TensorControl::TensorControl(size_t size_) {
  size = size_;
  ref_count = 1;
  on_device = false;
  input_handle.clear();
  output_handle.clear();
  output_convertor = nullptr;
  converted_data.clear();
  data = static_cast<char*>(tensorflow::port::AlignedMalloc(size_, 64));
}

PoplarExecutor::TensorControl::~TensorControl() {
  tensorflow::port::AlignedFree(data);
}

PoplarExecutor::OutfeedContext::OutfeedContext(const FeedInfo& outfeed_info)
    : config(outfeed_info.config),
      shapes(GetOutfeedShapes(FlattenedXlaShape(outfeed_info.shape),
                              outfeed_info.config.replication_factor())),
      tf_data_types(outfeed_info.config.tf_data_types().size()),
      tf_shapes(shapes.size()),
      callback_to_io_thread_queues(shapes.size()) {
  CHECK_EQ(shapes.size(), tf_data_types.size());
  int64 replication_factor = config.replication_factor();
  for (uint64 i = 0; i < shapes.size(); i++) {
    tf_data_types[i] = static_cast<tensorflow::DataType>(
        outfeed_info.config.tf_data_types()[i]);
    tensorflow::XLAShapeToTensorShape(shapes[i], &tf_shapes[i]);

    // Set up the queue per tensor per replica.
    int64 num_bytes_per_replica =
        ShapeUtil::ByteSizeOf(shapes[i]) / replication_factor;
    num_bytes_per_replica *= outfeed_info.config.io_batch_size();
    for (int64 replica_id = 0; replica_id < replication_factor; replica_id++) {
      void* ptr = tensorflow::port::AlignedMalloc(sizeof(OutfeedQueueType), 64);
      callback_to_io_thread_queues[i].emplace_back(
          new (ptr) OutfeedQueueType(num_bytes_per_replica));
    }
  }
}

bool PoplarExecutor::OutfeedContext::Matches(const FeedInfo& outfeed_info) {
  auto s = GetOutfeedShapes(FlattenedXlaShape(outfeed_info.shape),
                            outfeed_info.config.replication_factor());
  if (s.size() != shapes.size()) {
    return false;
  }
  for (auto i = 0; i < s.size(); i++) {
    if (s[i] != shapes[i]) {
      return false;
    }
  }
  return google::protobuf::util::MessageDifferencer::Equivalent(
      config, outfeed_info.config);
}

PoplarExecutor::PoplarExecutor()
    : ordinal_(0),
      current_engine_(nullptr),
      device_attached_(false),
      poplar_device_hash_(0),
      has_cycle_counter_(false),
      rendezvous_(tensorflow::NewLocalRendezvous()) {
  // TODO should this use the time/ms?
  static std::random_device rd;
  seed_generator_.Seed(rd());
}

PoplarExecutor::~PoplarExecutor() {}

void* PoplarExecutor::Allocate(uint64 size) {
  TensorControl* allocated = new TensorControl(size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    allocations_.push_back(allocated);
  }
  return allocated;
}

void* PoplarExecutor::GetSubBuffer(se::DeviceMemoryBase* parent,
                                   uint64 offset_bytes, uint64 size_bytes) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(parent->opaque());
  return tc->data + offset_bytes;
}

void PoplarExecutor::Deallocate(se::DeviceMemoryBase* mem) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(mem->opaque());
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (tc->ref_count > 0) {
      tc->ref_count--;
    }
  }
}

Status PoplarExecutor::ConnectSendCallbacksToRendezvous(
    const SendRecvInfos& send_infos) {
  if (send_infos.empty()) {
    return Status::OK();
  }

  const int64 num_replicas = current_replication_factor_;

  const bool can_buffers_overlap =
      CanPoplarSendBuffersOverlap(option_flags_, current_config_);

  // The IPU model uses temporary buffers, so they must be copied.
  const bool can_avoid_buffer_copy =
      !can_buffers_overlap && !PoplarXlaFlags::Get().use_ipu_model;

  if (can_avoid_buffer_copy) {
    VLOG(1)
        << "Assuming that Poplar send buffer pointers can be used without copy";
  }

  for (const SendRecvInfo& send : send_infos) {
    VLOG(1) << "Connecting Poplar IPU->host stream to rendezvous key '"
            << send.rendezvous_key << "' with shape " << send.shape
            << " and replication handling "
            << (send.concat_replicas ? "'Concat'" : "'First'");

    tensorflow::TensorShape shape;
    TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(send.shape, &shape));

    TF_ASSIGN_OR_RETURN(
        const tensorflow::DataType type,
        tensorflow::EncodePrimitiveTypeAsDataType(send.shape.element_type()));

    tensorflow::Rendezvous::ParsedKey key;
    TF_RETURN_IF_ERROR(
        tensorflow::Rendezvous::ParseKey(send.rendezvous_key, &key));

    // We allow capturing a raw pointer to the rendezvous in the lambda as
    // `this` which holds a refcount of it should outlive the engine.
    auto* rendezvous = GetRendezvous();

    auto callback_creator =
        send.concat_replicas ? SendConcatenatedCallbackCreator(
                                   shape, type, key, rendezvous, num_replicas)
                             : SendFromFirstReplicaCallbackCreator(
                                   shape, type, key, rendezvous, num_replicas,
                                   can_avoid_buffer_copy);

    for (int64 replica_id = 0; replica_id < num_replicas; ++replica_id) {
      current_engine_->connectStreamToCallback(send.stream_handle, replica_id,
                                               callback_creator(replica_id));
    }
  }

  return Status::OK();
}

Status PoplarExecutor::ConnectRecvCallbacksToRendezvous(
    const SendRecvInfos& recv_infos) {
  for (const SendRecvInfo& recv : recv_infos) {
    VLOG(1) << "Connecting Poplar host->IPU stream to rendezvous key '"
            << recv.rendezvous_key << "' with shape " << recv.shape;

    // We allow capturing a raw pointer to the rendezvous in the lambda as
    // `this` which holds a refcount of it should outlive the engine.
    auto* rendezvous = GetRendezvous();

    tensorflow::Rendezvous::ParsedKey key;
    TF_RETURN_IF_ERROR(
        tensorflow::Rendezvous::ParseKey(recv.rendezvous_key, &key));

    // This stream has ReplicatedStreamMode::BROADCAST, so every replica
    // will receive the same data sent here.
    current_engine_->connectStreamToCallback(
        recv.stream_handle, [rendezvous, key](void* dst) {
          tensorflow::Tensor tensor;
          bool is_dead = false;
          rendezvous->Recv(key, tensorflow::Rendezvous::Args{}, &tensor,
                           &is_dead);
          CHECK(!is_dead);
          auto* src = tensorflow::DMAHelper::buffer(&tensor);
          std::memcpy(dst, src->data(), src->size());
        });
  }

  return Status::OK();
}

namespace {
uint64 DeviceIncarnation(int device_ordinal, int replica) {
  return (device_ordinal << 5) | replica;
}
}  // namespace

Status PoplarExecutor::ConnectHostEmbeddingLookupToRendezvous(
    const HostEmbeddingInfo& lookup_info) {
  if (UseSyntheticData()) {
    return Status::OK();
  }

  // Extract the shapes and types.
  tensorflow::TensorShape indices_shape;
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(
      lookup_info.indices_shape, &indices_shape));

  for (int replica = 0;
       replica < std::max<int64>(1, current_replication_factor_); ++replica) {
    auto& embedding_interface = host_embeddings_[lookup_info.embedding_id];

    {
      std::unique_lock<std::mutex> lk(host_embeddings_mutex_);
      // Wait up to 5 seconds for the embedding interface to be initialized.
      if (!host_embeddings_cv.wait_until(
              lk, std::chrono::system_clock::now() + std::chrono::seconds(5),
              [&] { return static_cast<bool>(embedding_interface); })) {
        return xla::FailedPrecondition(
            "Host embedding interface with id='%s' not registered. Did you run "
            "the associated host_embedding op in the session?",
            lookup_info.embedding_id);
      }
    }

    // Connect the indices callback.
    current_engine_->connectStreamToCallback(
        lookup_info.stream_handle + lookup_info.embedding_id + "_indices",
        replica, [replica, indices_shape, &embedding_interface](void* ptr) {
          embedding_interface->EnqueueLookupIndices(
              replica, static_cast<int*>(ptr), indices_shape.num_elements());
        });

    // Connect the grads callback.
    current_engine_->connectStreamToCallback(
        lookup_info.stream_handle + lookup_info.embedding_id + "_activations",
        replica, [replica, &embedding_interface](void* ptr) {
          embedding_interface->DequeueLookupActivations(replica, ptr);

          if (embedding_interface->Done()) {
            embedding_interface.reset();
          }
        });
  }

  return Status::OK();
}

Status PoplarExecutor::ConnectHostEmbeddingUpdateToRendezvous(
    const HostEmbeddingInfo& update_info) {
  if (UseSyntheticData()) {
    return Status::OK();
  }

  // Extract the shapes and types.
  tensorflow::TensorShape indices_shape;
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(
      update_info.indices_shape, &indices_shape));

  for (int replica = 0;
       replica < std::max<int64>(1, current_replication_factor_); ++replica) {
    auto& embedding_interface = host_embeddings_[update_info.embedding_id];

    {
      std::unique_lock<std::mutex> lk(host_embeddings_mutex_);
      // Wait up to 5 seconds for the embedding interface to be initialized.
      if (!host_embeddings_cv.wait_until(
              lk, std::chrono::system_clock::now() + std::chrono::seconds(5),
              [&] { return static_cast<bool>(embedding_interface); })) {
        return xla::FailedPrecondition(
            "Host embedding interface with id='%s' not registered. Did you run "
            "the associated host_embedding op in the session?",
            update_info.embedding_id);
      }
    }

    // Connect the indices callback.
    current_engine_->connectStreamToCallback(
        update_info.stream_handle + update_info.embedding_id + "_indices",
        replica, [replica, indices_shape, &embedding_interface](void* ptr) {
          embedding_interface->EnqueueUpdateIndices(
              replica, static_cast<int*>(ptr), indices_shape.num_elements());
        });

    // Connect the grads callback.
    current_engine_->connectStreamToCallback(
        update_info.stream_handle + update_info.embedding_id + "_grads",
        replica, [replica, &embedding_interface](void* ptr) {
          embedding_interface->EnqueueUpdateGrads(replica, ptr);

          if (embedding_interface->Done()) {
            embedding_interface.reset();
          }
        });
  }

  return Status::OK();
}

namespace {
class InfeedPrefetchCallback : public poplar::StreamCallback {
 public:
  InfeedPrefetchCallback(InfeedQueue* queue, uint64 num_bytes)
      : queue_(queue), num_bytes_(num_bytes) {}

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    tensorflow::TensorBuffer* buffer;
    // Try to get a value from the queue.
    if (queue_->TryPop(buffer)) {
      std::memcpy(dest, buffer->data(), num_bytes_);
      return poplar::StreamCallback::Result::Success;
    } else {
      return poplar::StreamCallback::Result::NotAvailable;
    }
  }

  void fetch(void* dest) noexcept override {
    tensorflow::TensorBuffer* buffer;
    if (!queue_->BlockPop(buffer)) {
      LOG(FATAL) << "Infeed dataset iterator out of range. Are you trying to "
                 << "dequeue more elements than are in the dataset?";
    }

    std::memcpy(dest, buffer->data(), num_bytes_);
  }

  void complete() noexcept override { queue_->AdvanceReadPosition(); }

 private:
  InfeedQueue* queue_;
  const uint64 num_bytes_;
};

class NullPrefetchCallback : public poplar::StreamCallback {
 public:
  explicit NullPrefetchCallback(InfeedAllocator* allocator, uint64 num_bytes)
      : num_bytes_(num_bytes), allocator_(allocator) {
    for (auto& buffer : buffers_) {
      buffer = static_cast<uint8*>(allocator_->AllocateRaw(64, num_bytes));
      std::memset(buffer, 0x2, num_bytes);
    }
  }

  ~NullPrefetchCallback() {
    for (auto& buffer : buffers_) {
      allocator_->DeallocateRaw(buffer);
    }
  }

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    std::memcpy(dest, buffers_[index_], num_bytes_);
    return poplar::StreamCallback::Result::Success;
  }

  void fetch(void* dest) noexcept override {
    // This case shouldn't be hit, if poplar prefetches the data.
    std::memcpy(dest, buffers_[index_], num_bytes_);
  }

  void complete() noexcept override { index_ = (index_ + 1) % 16; }

 private:
  int index_ = 0;
  uint8* buffers_[16];
  const uint64 num_bytes_;
  InfeedAllocator* allocator_;
};
}  // namespace

void PoplarExecutor::ConnectInfeedsToStreamCallback(
    const InfeedInfos& infeed_infos) {
  // Don't connect any streams if using synthetic data
  if (UseSyntheticData()) {
    return;
  }

  for (const auto& infeed_info : infeed_infos) {
    auto itr = infeed_iterators_.find(infeed_info.config.feed_id());
    if (itr == infeed_iterators_.end()) {
      LOG(FATAL) << "Trying to access an infeed dataset iterator which has not "
                    "been created."
                 << " Did you initialize the infeed_queue?";
    }
    auto* infeed_dataset_iterator = itr->second.get();
    auto& shapes = infeed_dataset_iterator->GetShapes();
    auto& queues = infeed_dataset_iterator->GetInfeedQueues();

    for (auto replica_id = 0; replica_id < current_replication_factor_;
         ++replica_id) {
      auto& replica_queues = queues[replica_id];
      for (size_t j = 0; j < shapes.size(); ++j) {
        const auto length = ShapeUtil::ByteSizeOf(shapes[j]);
        const auto bytes_per_replica = length / current_replication_factor_;
        std::unique_ptr<poplar::StreamCallback> infeed_callback;
        if (PoplarXlaFlags::Get().null_data_feed) {
          infeed_callback = absl::make_unique<NullPrefetchCallback>(
              GetInfeedAllocator(), bytes_per_replica);
        } else {
          infeed_callback = absl::make_unique<InfeedPrefetchCallback>(
              replica_queues[j], bytes_per_replica);
        }
        current_engine_->connectStreamToCallback(
            GetInfeedCopyHandle(infeed_info.stream_prefix, j), replica_id,
            std::move(infeed_callback));
      }
    }
  }
}

void PoplarExecutor::ConnectOutfeedToStreamCallback(
    const OutfeedInfos& outfeed_infos) {
  // Don't connect any streams if using synthetic data
  if (UseSyntheticData()) {
    return;
  }

  for (const auto& outfeed_info : outfeed_infos) {
    const auto& outfeed_id = outfeed_info.config.feed_id();
    auto itr = outfeed_contexts_.find(outfeed_id);
    if (itr == outfeed_contexts_.end()) {
      LOG(FATAL) << "Outfeed with id='" << outfeed_id
                 << "' is not registered, but is required by the engine.";
    }

    auto* outfeed_context = itr->second.get();
    auto tensor_count = outfeed_context->shapes.size();
    for (unsigned j = 0; j < tensor_count; ++j) {
      size_t length = ShapeUtil::ByteSizeOf(outfeed_context->shapes[j]);
      auto bytes_per_replica = length / current_replication_factor_;
      bytes_per_replica *= outfeed_info.config.io_batch_size();
      for (auto replica_id = 0; replica_id < current_replication_factor_;
           ++replica_id) {
        auto& queue =
            outfeed_context->callback_to_io_thread_queues[j][replica_id];
        current_engine_->connectStreamToCallback(
            GetOutfeedCopyHandle(outfeed_info.stream_prefix, j), replica_id,
            [&queue, bytes_per_replica](void* src) {
              // The outfeed callback gets the buffer at the back of the queue,
              // writes to it, and then moves the write position of the queue.
              void* dest = queue->BlockBack();
              std::memcpy(dest, src, bytes_per_replica);
              queue->FinishedBack();
            });
      }
    }
  }
}

IOFunction PoplarExecutor::CreateInfeedIOThreadFunction(
    const FeedInfo& infeed_info) {
  // Find the iterator.
  auto itr = infeed_iterators_.find(infeed_info.config.feed_id());
  if (itr == infeed_iterators_.end()) {
    LOG(FATAL)
        << "Trying to access an infeed context which has not been created."
        << " Did you initialize the infeed_queue?";
  }
  InfeedIterator* infeed_dataset_iterator = itr->second.get();

  return [this, infeed_dataset_iterator](std::atomic<bool>& cancelled) {
    auto& infeed_queues = infeed_dataset_iterator->GetInfeedQueues();
    while (!cancelled) {
      // We do not call GetNext if queues are full.
      // We make an assumption that all tensors from each queue for each
      // replica for an infeed are dequeued every iteration - we therefore
      // only need to check if the first queue is full to know whether all the
      // queues are full.
      if (infeed_queues[0][0]->IsFull()) {
        VLOG(2) << "Infeed queue is full.";
        continue;
      }

      if (infeed_queues[0][0]->IsEmpty()) {
        VLOG(2) << "Infeed queue is empty.";
      }

      std::vector<tensorflow::Tensor> outputs;
      bool end_of_sequence = false;
      TF_RETURN_IF_ERROR(
          infeed_dataset_iterator->GetNext(&outputs, &end_of_sequence));

      if (end_of_sequence) {
        VLOG(1) << "The dataset iterator has reached the end of the dataset.";

        for (auto& queues : infeed_queues) {
          for (auto& queue : queues) {
            queue->SignalEndOfQueue();
          }
        }

        // This is not considered an error. However, we will report an
        // error if the consumer tries to pop past the end of the queue.
        return Status::OK();
      }

      for (size_t j = 0; j < outputs.size(); ++j) {
        auto& tensor = outputs[j];
        std::vector<tensorflow::Tensor> tensor_slices;
        if (current_replication_factor_ > 1) {
          // For replicated graphs, slice the input tensor and enqueue
          // it separately for each replica.
          CHECK_EQ(tensor.dim_size(0), current_replication_factor_);
          tensor_slices.reserve(current_replication_factor_);
          for (auto replica_id = 0; replica_id < current_replication_factor_;
               ++replica_id) {
            // Note that the tensor_slice shares the data buffer with the
            // tensor which works with ref counting.
            tensor_slices.push_back(tensor.SubSlice(replica_id));
          }
        } else {
          tensor_slices = {tensor};
        }

        // Enqueue tensors to each replica.
        for (size_t replica_id = 0; replica_id < tensor_slices.size();
             replica_id++) {
          auto& queue = infeed_queues[replica_id][j];
          auto* tb = tensorflow::DMAHelper::buffer(&tensor_slices[replica_id]);
          tb->Ref();
          queue->BlockPush(tb);
          queue->AdvanceWritePosition();
        }
      }
    }
    return Status::OK();
  };
}

namespace {
inline void AllocateTensors(std::deque<std::vector<tensorflow::Tensor>>& queue,
                            const std::vector<tensorflow::DataType>& types,
                            const std::vector<tensorflow::TensorShape>& shapes,
                            int count) {
  for (int c = 0; c < count; c++) {
    queue.emplace_front(types.size());
    auto& tensors = queue.front();
    for (size_t i = 0; i != types.size(); ++i) {
      tensors[i] = tensorflow::Tensor(types[i], shapes[i]);
    }
  }
}
}  // namespace

IOFunction PoplarExecutor::CreateOutfeedIOThreadFunction(
    const FeedInfo& outfeed_info) {
  auto itr = outfeed_contexts_.find(outfeed_info.config.feed_id());
  if (itr == outfeed_contexts_.end()) {
    LOG(FATAL)
        << "Trying to access an outfeed context which has not been created.";
  }
  OutfeedContext* outfeed_context = itr->second.get();

  return [this, outfeed_context](std::atomic<bool>& cancelled) {
    int replicas = current_replication_factor_;
    replicas = std::max(replicas, 1);

    // Lock the outfeed queues if it is of the GetLast type so that the CPU
    // OP does not try to dequeue the outfeed during the execution.
    if (outfeed_context->config.mode() == PoplarFeedConfig::GetLast) {
      outfeed_context->mutex.lock();
    }

    // Continue while the thread has not been cancelled, and if it has been
    // cancelled allow for up to two extra runs.
    uint32 all_queues_empty_for = 0;
    while (!cancelled || all_queues_empty_for != 2) {
      bool all_queues_empty = true;
      int io_batch_size = outfeed_context->config.io_batch_size();
      for (auto& tensor_queues :
           outfeed_context->callback_to_io_thread_queues) {
        for (auto& replica_queue : tensor_queues) {
          all_queues_empty &= !replica_queue->HasItemsWaiting();
        }
      }

      // Track empty queues when we are trying to exit
      if (all_queues_empty && cancelled) {
        all_queues_empty_for++;
      }

      // Continue if all the outfeed queues are empty.
      if (all_queues_empty) {
        continue;
      }

      // Lock the outfeed queue so that the CPU OP does not try to dequeue
      // whilst moving data off the device.
      {
        std::lock_guard<std::recursive_mutex> guard(outfeed_context->mutex);
        // Allocate the tensors before dequeuing.
        bool allocate_tensors = true;
        if (outfeed_context->config.mode() == PoplarFeedConfig::GetLast) {
          // For the get last we only allocate tensors once.
          allocate_tensors = outfeed_context->io_thread_output_queues.empty();
        }

        if (allocate_tensors) {
          AllocateTensors(outfeed_context->io_thread_output_queues,
                          outfeed_context->tf_data_types,
                          outfeed_context->tf_shapes, io_batch_size);
        }

        // We need to copy along 3 axis.  There are multiple queues from
        // the IPU, one  per tuple and per replica.  In each queue there
        // is a block of data containing one or more tensors.  There is a
        // single queue out of the executor, consisting of a vector of
        // Tensors, one per tuple entry.  If there are multiple replicas
        // then the outer dimension of the Tensors has the same value as the
        // replica count, and the output from each replica is concatenated
        // into that Tensor.
        //
        // We loop over each queue (by tuple  and replica), and dequeue the
        // block of data. This is then inserted  into the output queue as
        // appropriate.
        for (size_t tuple_idx = 0; tuple_idx < outfeed_context->shapes.size();
             ++tuple_idx) {
          // Dequeue tensors from each replica.
          for (int64 replica_id = 0; replica_id < replicas; replica_id++) {
            auto& queue =
                outfeed_context
                    ->callback_to_io_thread_queues[tuple_idx][replica_id];

            // Dequeue the data and insert into the correct output queue.
            uint8_t* src = reinterpret_cast<uint8_t*>(queue->BlockFront());
            for (int b = 0; b < io_batch_size; b++) {
              std::vector<tensorflow::Tensor>& tensors_to_write_to =
                  outfeed_context->io_thread_output_queues.at(io_batch_size -
                                                              b - 1);

              auto& tensor = tensors_to_write_to[tuple_idx];

              // When there are mutiple replicas, insert the data into a slice
              // out of dinension 0.  Otherwise just use the whole tensor.
              auto output_tensor =
                  (replicas == 1 ? tensor : tensor.SubSlice(replica_id));
              auto* tb = tensorflow::DMAHelper::buffer(&output_tensor);

              std::memcpy(tb->data(), src, output_tensor.AllocatedBytes());
              src += output_tensor.AllocatedBytes();
            }
            queue->FinishedFront();
          }
        }
      }
    }

    // Unlock all the outfeed if it is of the GetLast type.
    if (outfeed_context->config.mode() == PoplarFeedConfig::GetLast) {
      outfeed_context->mutex.unlock();
    }
    return Status::OK();
  };
}

void PoplarExecutor::LaunchIOThreads(const InfeedInfos& infeed_infos,
                                     const OutfeedInfos& outfeed_infos) {
  CHECK_EQ(io_threads_.size(), 0);
  // Start all the infeeds.
  for (const FeedInfo& info : infeed_infos) {
    IOFunction fn = CreateInfeedIOThreadFunction(info);
    io_threads_.emplace_back(
        absl::make_unique<IOThread>(info.config.feed_id(), std::move(fn)));
  }

  // Start all the outfeeds.
  for (const FeedInfo& info : outfeed_infos) {
    IOFunction fn = CreateOutfeedIOThreadFunction(info);
    io_threads_.emplace_back(
        absl::make_unique<IOThread>(info.config.feed_id(), std::move(fn)));
  }
}

void PoplarExecutor::StopIOThreads() {
  // Blocks the thread until all the threads have stopped and joined back.
  io_threads_.clear();
}

void PoplarExecutor::DeferredDeallocation() {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());

  const auto new_end =
      std::partition(allocations_.begin(), allocations_.end(),
                     [](TensorControl* tc) { return tc->ref_count > 0; });

  std::for_each(new_end, allocations_.end(),
                [](TensorControl* tc) { delete tc; });

  allocations_.erase(new_end, allocations_.end());
}

bool PoplarExecutor::Memcpy(se::Stream* stream, void* host_dst,
                            const se::DeviceMemoryBase& pop_src, uint64 size) {
  AsPoplarStream(stream)->EnqueueTask([this, host_dst, pop_src, size]() {
    Status ok = SynchronousMemcpy(host_dst, pop_src, size);
  });
  return true;
}

bool PoplarExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const void* host_src, uint64 size) {
  se::DeviceMemoryBase dst = *pop_dst;
  AsPoplarStream(stream)->EnqueueTask([this, dst, host_src, size]() mutable {
    Status ok = SynchronousMemcpy(&dst, host_src, size);
  });
  return true;
}

Status PoplarExecutor::SynchronousMemcpy(se::DeviceMemoryBase* pop_dst,
                                         const void* host_src, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  memcpy(tc->data, host_src, size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    tc->on_device = false;
    tc->input_handle.clear();
  }
  return Status::OK();
}

Status PoplarExecutor::SynchronousMemcpy(void* host_dst,
                                         const se::DeviceMemoryBase& pop_src,
                                         uint64 size) {
  const TensorControl* tc =
      reinterpret_cast<const TensorControl*>(pop_src.opaque());
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (tc->on_device == true && !tc->output_handle.empty()) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost());
    }
  }
  memcpy(host_dst, tc->data, size);
  return Status::OK();
}

Status PoplarExecutor::SynchronousMemcpyDeviceToDevice(
    se::DeviceMemoryBase* dst, const se::DeviceMemoryBase& src, uint64 size) {
  TensorControl* dst_tc = reinterpret_cast<TensorControl*>(dst->opaque());
  const TensorControl* src_tc =
      reinterpret_cast<const TensorControl*>(src.opaque());
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (src_tc->on_device == true && !src_tc->output_handle.empty()) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost());
    }
  }
  memcpy(dst_tc->data, src_tc->data, size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    dst_tc->on_device = false;
    dst_tc->input_handle.clear();
  }
  return Status::OK();
}

bool PoplarExecutor::MemcpyDeviceToDevice(se::Stream* stream,
                                          se::DeviceMemoryBase* pop_dst,
                                          const se::DeviceMemoryBase& pop_src,
                                          uint64 size) {
  se::DeviceMemoryBase dst = *pop_dst;
  AsPoplarStream(stream)->EnqueueTask([this, dst, pop_src, size]() mutable {
    SynchronousMemcpyDeviceToDevice(&dst, pop_src, size);
  });
  return true;
}

bool PoplarExecutor::HostCallback(se::Stream* stream,
                                  std::function<void()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::HostCallback(se::Stream* stream,
                                  std::function<Status()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::CreateStreamDependency(se::Stream* dependent,
                                            se::Stream* other) {
  AsPoplarStream(dependent)->EnqueueTask(
      [other]() { auto ok = other->BlockHostUntilDone(); });
  AsPoplarStream(dependent)->BlockUntilDone();
  return true;
}

bool PoplarExecutor::StartTimer(se::Stream* stream, se::Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool PoplarExecutor::StopTimer(se::Stream* stream, se::Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

Status PoplarExecutor::BlockHostUntilDone(se::Stream* stream) {
  AsPoplarStream(stream)->BlockUntilDone();
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  return Status::OK();
}

bool PoplarExecutor::SynchronizeAllActivity() {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  return true;
}

StatusOr<std::unique_ptr<se::DeviceDescription>>
PoplarExecutor::CreateDeviceDescription() const {
  auto platform =
      se::MultiPlatformManager::PlatformWithName(tensorflow::PLATFORM_NAME);
  if (platform.ok()) {
    auto* p = static_cast<PoplarPlatform*>(platform.ValueOrDie());
    return p->DescriptionForDevice(0);
  }
  return InternalError("Failed to create device description.");
}

bool PoplarExecutor::IPUConfig::DeviceConfigured() const {
  return device_.has_value();
}

bool PoplarExecutor::IPUConfig::TargetConfigured() const {
  return target_.has_value();
}

void PoplarExecutor::IPUConfig::ClearDevice() { device_.reset(); }

std::recursive_mutex& PoplarExecutor::IPUConfig::Mutex() { return mutex_; }

const poplar::Target& PoplarExecutor::IPUConfig::Target() {
  if (!target_ && PoplarXlaFlags::Get().use_ipu_model) {
    // If the device has not been configured via configure_ipu_system, but we
    // have requested an IPU model, then we create a CPU device.
    std::lock_guard<std::recursive_mutex> g(mutex_);
    // Poplar CPU device
    device_ = poplar::Device::createCPUDevice();
    target_ = device_->getTarget();
  }
  return TargetOrDie();
}

const poplar::Target& PoplarExecutor::IPUConfig::TargetOrDie() const {
  CHECK(target_);
  return *target_;
}

const poplar::Device& PoplarExecutor::IPUConfig::Device() const {
  CHECK(device_);
  return *device_;
}

void PoplarExecutor::IPUConfig::SetDevice(poplar::Device&& device) {
  device_ = std::move(device);
}

void PoplarExecutor::IPUConfig::SetDeviceAndTarget(poplar::Device&& device) {
  device_ = std::move(device);
  target_ = device_->getTarget();
}

void PoplarExecutor::IPUConfig::SetTarget(const poplar::Target& target) {
  target_ = target;
}

std::string PoplarExecutor::GetDeviceTargetName() const {
  return poplar::toString(ipu_.TargetOrDie().getTargetType());
}

static bool DeviceConfigurationsEqual(const IpuOptions& a,
                                      const IpuOptions& b) {
  return google::protobuf::util::MessageDifferencer::Equivalent(a, b);
}

const poplar::Target& PoplarExecutor::GetOrCreatePoplarTarget() {
  if (!ipu_.TargetConfigured() && PoplarXlaFlags::Get().use_ipu_model) {
    // If the device has not been configured via configure_ipu_system, but we
    // have requested an IPU model, then we create a CPU device.
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    // Poplar CPU device
    ipu_.SetDeviceAndTarget(poplar::Device::createCPUDevice());
  }

  return ipu_.Target();
}

bool PoplarExecutor::HasPoplarTarget() const {
  // Target is configured, or the model will be used.
  // (In which case it will be lazy-initialized in GetOrCreatePoplarTarget().
  return PoplarXlaFlags::Get().use_ipu_model || ipu_.TargetConfigured();
}

bool PoplarExecutor::PoplarDeviceIsAttached() const { return device_attached_; }

Status PoplarExecutor::AttachToPoplarDevice() {
  const bool use_ipu_model = PoplarXlaFlags::Get().use_ipu_model;
  if (device_attached_) {
    return InternalError("Already attached to device");
  }

  try {
    if (!ipu_.TargetConfigured()) {
      if (!use_ipu_model) {
        return InvalidArgument("Device not configured and IPU model disabled.");
      }
      GetOrCreatePoplarTarget();
    }
    if (ipu_.DeviceConfigured()) {
      // Device was selected when the target was created: attach or fail.
      if (!ipu_.Device().attach()) {
        if (use_ipu_model) {
          return xla::InternalError(
              "Unable to acquire poplar device model for ordinal %d", ordinal_);
        } else {
          const int32 cfg_index =
              current_config_.device_config(ordinal_).cfg_index();
          return xla::InternalError(
              "Could not attach to requested device configuration index %d",
              cfg_index);
        }
      }
    } else {
      // Poplar device would already be set if we were using the model.
      CHECK(HasIpuHardware());
      const poplar::Target& target = GetOrCreatePoplarTarget();
      // Hardware devices
      auto device_list = GetDeviceManager().getDevices(target.getTargetType(),
                                                       target.getNumIPUs());
      for (auto& d : device_list) {
        // Try to attach to that device.
        if (d.attach()) {
          ipu_.SetDevice(std::move(d));
          break;
        }
      }
      if (!ipu_.DeviceConfigured()) {
        if (device_list.size()) {
          return xla::InternalError(
              "Failed to attach to any of the device(s) with matching configs "
              "for ordinal %d",
              ordinal_);
        } else {
          return InvalidArgument(
              "No device matches the requested configuration");
        }
      }
    }

    // If real HW only
    if (!use_ipu_model) {
      unsigned mj, mn, pt;
      ipu_.Device().getDriverVersion(mj, mn, pt);
      VLOG(1) << "Poplar driver: " << mj << "." << mn << "." << pt;

      const auto& ids = ipu_.Device().getDriverIDs();
      LOG(INFO) << "Device /device:IPU:" << ordinal_ << " attached to IPU"
                << (ids.size() > 1 ? "s" : "") << ": "
                << absl::StrJoin(ids, ",");
    }

    VLOG(1) << "Opened Poplar device type " << GetDeviceTargetName();
    device_attached_ = true;
  } catch (poplar::poplar_error e) {
    return xla::InternalError("Unable to open poplar device for ordinal %d: %s",
                              ordinal_, e.what());
  }

  return Status::OK();
}

Status PoplarExecutor::CreatePoplarTarget() {
  bool has_user_config = (current_config_.device_config_size() > 0);

  if (!PoplarXlaFlags::Get().use_ipu_model) {
    if (current_config_.device_connection_type() !=
            IpuDeviceConnectionType::NEVER &&
        !HasIpuHardware()) {
      return InvalidArgument(
          "Target configuration failed: model disabled and no hardware IPU "
          "found. (Are you sure you enabled the Poplar driver ?)");
    }
    // Try to extract info from the user configuration if it was provided.
    absl::optional<int64> device_index;
    absl::optional<int64> num_devices;
    if (has_user_config) {
      // User has specified a configuration
      if (ordinal_ >= current_config_.device_config_size()) {
        return InternalError(
            "Device ordinal %d not in device configuration list.", ordinal_);
      }

      auto device_config = current_config_.device_config(ordinal_);
      if (device_config.selection_case() ==
          IpuOptions::DeviceConfig::SelectionCase::kCfgIndex) {
        // The config specifies the index of the device to use.
        device_index = device_config.cfg_index();
      } else {
        num_devices = device_config.auto_count();
      }
    } else {
      // User didn't specify a configuration - default to a single IPU.
      LOG(INFO) << "No IPU device was configured for /device:IPU:" << ordinal_
                << ", creating a device with a single IPU.";
      num_devices = 1;
    }

    if (device_index) {
      CHECK(HasIpuHardware());
      // A specific device was chosen.
      auto device_list = GetDeviceManager().getDevices();
      ipu_.SetDeviceAndTarget(std::move(device_list.at(*device_index)));
    } else {
      CHECK(num_devices);
      // If there is an IPU version configured then use that.
      if (current_config_.has_ipu_version()) {
        ipu_.SetTarget(
            CreateIpuTarget(*num_devices, current_config_.ipu_version()));
      } else {
        // Deduce the IPU target given the configuration.
        switch (current_config_.device_connection_type()) {
          case IpuDeviceConnectionType::ALWAYS:
          case IpuDeviceConnectionType::ON_DEMAND: {
            CHECK(HasIpuHardware());
            // Get target from the available devices.
            auto device_list = GetDeviceManager().getDevices(
                poplar::TargetType::IPU, *num_devices);
            if (device_list.empty()) {
              return FailedPrecondition(
                  "Could not find any IPU devices with %d IPUs for "
                  "/device:IPU:%d",
                  *num_devices, ordinal_);
            }
            ipu_.SetTarget(device_list.front().getTarget());
            break;
          }
          case IpuDeviceConnectionType::NEVER: {
            return FailedPrecondition(
                "Expected the `ipu_version` to be set when the "
                "`device_connection_type` is set to "
                "`IpuDeviceConnectionType.NEVER`");
          }
          default: {
            return FailedPrecondition("Unrecognised connection type.");
          }
        }
      }
    }
  } else {
    int num_ipus = 1;
    if (has_user_config) {
      auto device_config = current_config_.device_config(ordinal_);

      if (device_config.selection_case() ==
          IpuOptions::DeviceConfig::SelectionCase::kCfgIndex) {
        return InvalidArgument(
            "Must specify the number of IPUs using auto_count, not an "
            "index.");
      }

      num_ipus = device_config.auto_count();
    }
    poplar::IPUModel model;
    model.numIPUs = num_ipus;

    model.compileIPUCode =
        current_config_.ipu_model_config().compile_ipu_code();
    ipu_.SetDeviceAndTarget(model.createDevice());
  }
  return Status::OK();
}

Status PoplarExecutor::ConfigurePoplarDevice(const IpuOptions& cfg) {
  bool has_user_config = (current_config_.device_config_size() > 0);
  if (!DeviceConfigurationsEqual(cfg, current_config_) && has_user_config) {
    XLA_VLOG_LINES(1, "Current config: " + current_config_.DebugString() +
                          "\nNew config: " + cfg.DebugString());
    return InternalError("IPU system configuration can only be set once.");
  }
  if (device_attached_) {
    if (DeviceConfigurationsEqual(current_config_, IpuOptions())) {
      // If there is no config associated to the open device then it is a CPU
      // device: dettach from it and initialize a Poplar device instead.
      VLOG(1) << "Detaching from " << GetDeviceTargetName() << " ordinal "
              << ordinal_;
      ipu_.Device().detach();
      ipu_.ClearDevice();
      device_attached_ = false;
    } else {
      VLOG(1) << "Poplar device: type " << GetDeviceTargetName() << " ordinal "
              << ordinal_ << " is already configured: staying attached to it.";
    }
  }
  current_config_ = cfg;

  if (!device_attached_) {
    TF_RETURN_IF_ERROR(CreatePoplarTarget());
    if (cfg.device_connection_type() == IpuDeviceConnectionType::ALWAYS) {
      TF_RETURN_IF_ERROR(AttachToPoplarDevice());
    }
  }

  option_flags_ = poplar::OptionFlags();

  // Set appropriate options for trace levels.
  switch (current_config_.profiling().execution_trace_type()) {
    case IpuExecutionProfileType::NO_PROFILE:
      break;
    case IpuExecutionProfileType::DEVICE_PROFILE:
      option_flags_.set("debug.instrument", "true");
      option_flags_.set("debug.computeInstrumentationLevel", "device");
      break;
    case IpuExecutionProfileType::IPU_PROFILE:
      option_flags_.set("debug.instrument", "true");
      option_flags_.set("debug.computeInstrumentationLevel", "ipu");
      break;
    case IpuExecutionProfileType::TILE_PROFILE:
      option_flags_.set("debug.instrument", "true");
      option_flags_.set("debug.computeInstrumentationLevel", "tile");
      break;
  }

  if (UseVerifiedTransfers()) {
    option_flags_.set("opt.useAutoloader", "false");
    option_flags_.set("target.useBufferedCompletions", "false");
  }

  // By setting stream options before user options we make sure the user can
  // override this default behaviour.
  if (current_config_.prefetch_data_streams()) {
    // By default we only rearrange copies on the host for resource variable
    // inputs which do not need to be prefetched, however if we rearrange
    // everything on the host, we do not overlap any stream buffers.
    option_flags_.set(
        "exchange.streamBufferOverlap",
        AlwaysRearrangeCopiesOnTheHost() ? "none" : "hostRearrangeOnly");
    option_flags_.set("exchange.enablePrefetch", "true");
  }

  for (const auto& opt : current_config_.compilation_options()) {
    option_flags_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.convolution_options()) {
    conv_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.matmul_options()) {
    matmul_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.pooling_options()) {
    pooling_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.profiling().graph_options()) {
    graph_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.profiling().execution_options()) {
    execution_options_.set(opt.option(), opt.value());
  }

  const auto max_compilation_threads =
      PoplarXlaFlags::Get().max_compilation_threads;
  if (max_compilation_threads > 0) {
    option_flags_.set("opt.maxCompilationThreads",
                      std::to_string(max_compilation_threads));
  }

  if (CompilerReportingEnabled()) {
    option_flags_.set("debug.allowOutOfMemory", "true");
  }

  for (auto opt : option_flags_) {
    VLOG(1) << "Engine option: " << opt.first << " = " << opt.second;
  }

  for (auto opt : conv_options_) {
    VLOG(1) << "Convolution option: " << opt.first << " = " << opt.second;
  }

  for (auto opt : matmul_options_) {
    VLOG(1) << "MatMul option: " << opt.first << " = " << opt.second;
  }

  for (auto opt : pooling_options_) {
    VLOG(1) << "Pooling option: " << opt.first << " = " << opt.second;
  }

  VLOG(1) << "Use verified transfers: "
          << (UseVerifiedTransfers() ? "Yes" : "No");

  for (auto opt : graph_options_) {
    VLOG(1) << "Graph report option: " << opt.first << " = " << opt.second;
  }

  for (auto opt : execution_options_) {
    VLOG(1) << "Execution report option: " << opt.first << " = " << opt.second;
  }

  // Generate Target hash
  std::vector<int64> target_hash;
  target_hash.push_back(ipu_.Target().getNumTiles());
  target_hash.push_back(ipu_.Target().getDataPathWidth());
  target_hash.push_back(ipu_.Target().getBytesPerTile());
  target_hash.push_back(ipu_.Target().getNumWorkerContexts());
  target_hash.push_back(ipu_.Target().getTilesPerIPU());
  target_hash.push_back(ipu_.Target().getNumIPUs());
  target_hash.push_back((unsigned)ipu_.Target().getTargetType());

  // Generate Options hash
  target_hash.push_back(GetConfigHash(current_config_));

  // Generate compiler hashes
  target_hash.push_back(std::hash<string>()(tf_git_version()));
  target_hash.push_back(std::hash<string>()(poplar::packageHash()));

  // Get environment PoplarXlaFlags hash
  target_hash.push_back(PoplarXlaFlags::Get().hlo_hash);

  // TODO(T12447) - use a hash returned by Poplar.
  char* env_engine_options = getenv("POPLAR_ENGINE_OPTIONS");
  if (env_engine_options) {
    target_hash.push_back(std::hash<string>()(std::string(env_engine_options)));
  }

  // Get remote memory support.
  target_hash.push_back(SupportsRemoteBuffers());

  // Get hashes for GCL compilation parameters.
  absl::c_copy(GetGclHashes(), std::back_inserter(target_hash));

  poplar_device_hash_ = CombinedHash(target_hash);

  return Status::OK();
}

bool PoplarExecutor::HaveExecutableCache() const {
  return !PoplarXlaFlags::Get().executable_cache_path.empty();
}

Status PoplarExecutor::CreateExecutableCacheDirIfMissing() const {
  return CreateDirIfMissing(PoplarXlaFlags::Get().executable_cache_path);
}

std::string ModuleFilenames::SerializedExecutableFilename() const {
  return tensorflow::io::JoinPath(serialization_folder_,
                                  basename_ + ".ipu_bin");
}

std::string ModuleFilenames::Name() const { return basename_; }

std::string ModuleFilenames::SerializedMetadataFilename() const {
  return tensorflow::io::JoinPath(serialization_folder_, basename_ + ".json");
}

Status PoplarExecutor::CreateSerializedExecutableDirIfMissing() const {
  return CreateDirIfMissing(SerializationFolder());
}

ModuleFilenames::ModuleFilenames(const HloModule& module, int64 device_hash,
                                 const std::string& serialization_folder)
    : basename_(tensorflow::strings::Printf(
          "%0llx",
          tensorflow::Hash64Combine(HloHash(&module).GetHash(), device_hash))),
      serialization_folder_(serialization_folder) {}

std::string ModuleFilenames::CachedExecutableFilename() const {
  return absl::StrCat(CachedEngineFilename(), ".poplar_exec");
}

std::string ModuleFilenames::CachedEngineFilename() const {
  return tensorflow::io::JoinPath(PoplarXlaFlags::Get().executable_cache_path,
                                  basename_ + ".xla_engine");
}

ModuleFilenames PoplarExecutor::GetModuleFilenames(
    const HloModule& module) const {
  return ModuleFilenames(module, poplar_device_hash_, SerializationFolder());
}

bool PoplarExecutor::HaveCachedExecutable(
    const ModuleFilenames& filenames) const {
  return tensorflow::Env::Default()
      ->FileExists(filenames.CachedEngineFilename())
      .ok();
}

bool PoplarExecutor::SupportsRemoteBuffers() const {
  if (!PoplarDeviceIsAttached()) {
    return false;
  }
  if (ipu_.TargetOrDie().getTargetType() != poplar::TargetType::IPU) {
    return false;
  }

  return ipu_.Device().supportsGraphStreaming();
}

tensorflow::IpuTraceEvent PoplarExecutor::NewTraceEvent() {
  uint64 now = tensorflow::Env::Default()->NowMicros();
  tensorflow::IpuTraceEvent evt;
  evt.set_timestamp(static_cast<double>(now) / 1000000.0);
  evt.set_ordinal(ordinal_);
  return evt;
}

void PoplarExecutor::AddCompileBeginEventRecord(
    const std::string& module_name) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::COMPILE_BEGIN);
  evt.mutable_compile_begin()->set_module_name(std::move(module_name));

  reports_.push_back(evt);
}

std::string PoplarExecutor::ReportFileExtension() const {
  std::string report_file_extension = "";
  if (CompilerReportingTextFormat()) {
    report_file_extension = "txt";
  } else if (CompilerReportingCborFormat()) {
    report_file_extension = "cbor";
  } else {
    report_file_extension = "json";
  }

  return report_file_extension;
}

void PoplarExecutor::AddCompileEndEventRecord(
    const std::string& module_name, const std::string& report,
    const std::string& poplar_graph, const std::string& tensor_map,
    const std::string& instruction_info, int64 duration) {
  std::string rep = std::move(report);
  std::string map = std::move(tensor_map);
  std::string gph = std::move(poplar_graph);

  if (ReportDirectory().size() > 0) {
    std::unique_ptr<tensorflow::WritableFile> file;

    std::string report_file_extension = ReportFileExtension();

    std::string report_dir =
        tensorflow::io::JoinPath(ReportDirectory(), module_name);
    CreateDirIfMissing(report_dir);

    if (rep.size() > 0) {
      std::string filename = tensorflow::io::JoinPath(
          report_dir, "graph." + report_file_extension);
      TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
      TF_CHECK_OK(file->Append(rep));
      TF_CHECK_OK(file->Close());
      rep = filename;
    }

    if (map.size() > 0) {
      std::string filename =
          tensorflow::io::JoinPath(report_dir, "tensor_map.json");
      TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
      TF_CHECK_OK(file->Append(map));
      TF_CHECK_OK(file->Close());
      map = filename;
    }

    if (gph.size() > 0) {
      std::string filename =
          tensorflow::io::JoinPath(report_dir, "serialized_graph.capnp");
      TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
      TF_CHECK_OK(file->Append(gph));
      TF_CHECK_OK(file->Close());
      gph = filename;
    }
  }

  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::COMPILE_END);

  auto* compile_end = evt.mutable_compile_end();
  compile_end->set_module_name(std::move(module_name));
  compile_end->set_compilation_report(std::move(rep));
  compile_end->set_poplar_graph(std::move(gph));
  compile_end->set_duration(duration);
  compile_end->set_tensor_map(std::move(map));
  compile_end->set_instruction_info(std::move(instruction_info));

  reports_.push_back(evt);
}

void PoplarExecutor::AddHostToDeviceEventRecord(const std::string& json) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::HOST_TO_DEVICE_TRANSFER);
  evt.mutable_data_transfer()->set_data_transfer(std::move(json));

  reports_.push_back(evt);
}

void PoplarExecutor::AddDeviceToHostEventRecord(const std::string& json) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::DEVICE_TO_HOST_TRANSFER);
  evt.mutable_data_transfer()->set_data_transfer(std::move(json));

  reports_.push_back(evt);
}

void PoplarExecutor::AddLoadEngineEventRecord(const std::string& module_name) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::LOAD_ENGINE);
  evt.mutable_load_engine()->set_module_name(std::move(module_name));

  reports_.push_back(evt);
}

void PoplarExecutor::AddExecuteEventRecord(const std::string& module_name,
                                           const std::string& report) {
  std::string rep = std::move(report);
  if (ReportDirectory().size() > 0 && rep.size()) {
    std::unique_ptr<tensorflow::WritableFile> file;

    std::string report_file_extension = ReportFileExtension();

    std::string report_dir =
        tensorflow::io::JoinPath(ReportDirectory(), module_name);
    CreateDirIfMissing(report_dir);

    std::string filename = tensorflow::io::JoinPath(
        report_dir, "execution." + report_file_extension);
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
    TF_CHECK_OK(file->Append(rep));
    TF_CHECK_OK(file->Close());
    rep = filename;
  }

  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::EXECUTE);
  evt.mutable_execute()->set_module_name(std::move(module_name));
  evt.mutable_execute()->set_execution_report(std::move(rep));

  reports_.push_back(evt);
}

Status PoplarExecutor::GetCompilerEvents(
    std::list<tensorflow::IpuTraceEvent>& out) {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  out.splice(out.end(), std::move(reports_));
  reports_.clear();
  return Status::OK();
}

void PoplarExecutor::FlattenedDeviceMemoryList(
    InputPairList& list, const xla::Shape& shape, void* base,
    const InputOutputAliasingMap::InputInfo& input_info,
    bool is_remote_parameter) {
  TensorControl* tc = static_cast<TensorControl*>(base);
  if (shape.IsTuple()) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t = 0; t < xla::ShapeUtil::TupleElementCount(shape);
         t++) {
      void* ptr = ptrs[t];
      FlattenedDeviceMemoryList(list,
                                xla::ShapeUtil::GetTupleElementShape(shape, t),
                                ptr, input_info, is_remote_parameter);
    }
  } else {
    list.push_back(InputDef(tc, GetInputConversionFunction(shape),
                            input_info.IsStreaming(), is_remote_parameter));
  }
}

void PoplarExecutor::UpdateArgsHandleMap(
    const Args& args, se::DeviceMemoryAllocator* allocator,
    const xla::poplarplugin::PoplarExecutable& executable) {
  args_map_.clear();

  const auto* comp = executable.module().entry_computation();
  std::vector<xla::Shape> shapes(comp->num_parameters());
  for (const auto& inst : comp->parameter_instructions()) {
    shapes[inst->parameter_number()] = inst->shape();
  }

  const auto& inputs_info =
      executable.GetInputOutputAliasingMap().GetEntryInputInfos();
  CHECK_EQ(inputs_info.size(), args.size());
  CHECK_EQ(shapes.size(), args.size());

  // We require all the resource arguments which are modified to be not-aliasing
  // with each other.
  absl::flat_hash_set<const TensorControl*> modified_resources;

  for (unsigned int a = 0; a < inputs_info.size(); a++) {
    const auto& input_info = inputs_info[a];
    InputPairList bufs;
    const bool is_remote_parameter =
        IsRemoteParameter(a, executable.GeRemoteParameterInfos());
    FlattenedDeviceMemoryList(bufs, shapes[a],
                              const_cast<void*>(args[a].opaque()), input_info,
                              is_remote_parameter);
    for (unsigned i = 0; i < bufs.size(); i++) {
      InputDef input = bufs[i];
      auto input_handle = input_info.Handles().at(i);
      if (input_info.IsResource() && !input_info.IsResourceNotModified()) {
        if (modified_resources.contains(input.tc)) {
          // We found an alias - we add a copy.
          VLOG(1) << "Found an alias for input handle " << input_handle
                  << ", duplicating the buffer.";
          se::DeviceMemoryBase allocated =
              allocator->Allocate(ordinal_, input.tc->size, false)
                  .ConsumeValueOrDie()
                  .Release();
          TensorControl* tc =
              reinterpret_cast<TensorControl*>(allocated.opaque());
          std::memcpy(tc->data, input.tc->data, input.tc->size);
          input =
              InputDef(tc, input.fn, input.streamed, input.remote_parameter);
        }
        modified_resources.insert(input.tc);
      }

      args_map_[input_handle] = input;
    }
  }
}

void PoplarExecutor::FlattenedOutputDeviceMemoryList(
    OutputPairList& list, const xla::Shape& shape, void* base,
    const InputOutputAliasingMap::OutputInfo& output_info) {
  TensorControl* tc = static_cast<TensorControl*>(base);
  if (shape.IsTuple()) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t = 0; t < xla::ShapeUtil::TupleElementCount(shape);
         t++) {
      void* ptr = ptrs[t];
      FlattenedOutputDeviceMemoryList(
          list, xla::ShapeUtil::GetTupleElementShape(shape, t), ptr,
          output_info);
    }
  } else {
    list.push_back(OutputDef(tc, output_info.IsStreaming()));
  }
}

void PoplarExecutor::UpdateOutputsHandleMap(
    const xla::poplarplugin::PoplarExecutable& executable,
    const xla::Shape& shape, se::DeviceMemoryBase retbuf) {
  outputs_map_.clear();

  // Get all output pointers and their shapes
  std::vector<void*> outputs;
  std::vector<xla::Shape> shapes;

  if (shape.IsTuple()) {
    TensorControl* tc = static_cast<TensorControl*>(retbuf.opaque());
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      shapes.push_back(ShapeUtil::GetTupleElementShape(shape, i));
      outputs.push_back(ptrs[i]);
    }
  } else {
    shapes.push_back(shape);
    outputs.push_back(retbuf.opaque());
  }

  // For all outputs
  const auto& outputs_info =
      executable.GetInputOutputAliasingMap().GetEntryOutputInfos();
  CHECK_EQ(outputs_info.size(), shapes.size());
  CHECK_EQ(outputs.size(), shapes.size());
  for (unsigned int a = 0; a < outputs_info.size(); a++) {
    const auto& output_info = outputs_info[a];
    OutputPairList bufs;
    FlattenedOutputDeviceMemoryList(bufs, shapes[a], outputs[a], output_info);
    for (unsigned i = 0; i < bufs.size(); i++) {
      outputs_map_[bufs[i].tc->output_handle] = bufs[i];
    }
  }
}

se::DeviceMemoryBase PoplarExecutor::ConstantOutputAllocation::GetAllocation(
    se::DeviceMemoryAllocator* allocator, const xla::Shape& shape,
    const int64 output_index, int64& flat_tensor_index, const Args&,
    const InputOutputAliasingMap::OutputInfo&, const ArgsHandleMap&,
    const int ordinal) const {
  const auto& constant = constants_[output_index][flat_tensor_index];
  const int64 size(xla::ShapeUtil::ByteSizeOf(shape));
  se::DeviceMemoryBase allocated =
      allocator->Allocate(ordinal, size, false).ConsumeValueOrDie().Release();
  TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
  tc->size = size;
  tc->on_device = false;
  tc->output_handle = std::string();
  tc->output_convertor = nullptr;

  void* buf(static_cast<void*>(tc->data));
  memcpy(buf, constant.untyped_data(), constant.size_bytes());
  return allocated;
}

se::DeviceMemoryBase PoplarExecutor::RemapOutputAllocation::GetAllocation(
    se::DeviceMemoryAllocator* allocator, const xla::Shape&,
    const int64 output_index, int64& flat_tensor_index, const Args& args,
    const InputOutputAliasingMap::OutputInfo&, const ArgsHandleMap& args_map,
    const int ordinal) const {
  const auto& remap_idx = remap_map_[output_index];
  auto it = args_map.find(GetInputCopyHandle(remap_idx, flat_tensor_index));
  if (it == args_map.end()) {
    LOG(FATAL) << "Could not remap an output to input tensor.";
  }

  bool make_a_copy = false;

  auto input_infos = input_output_aliasing_map_.GetEntryInputInfos();
  auto output_infos = input_output_aliasing_map_.GetEntryOutputInfos();
  if (input_infos.size() > 0 && output_infos.size() > 0) {
    int input_index = output_infos[output_index].GetInputIndex();
    bool is_input_resource = input_infos[input_index].IsResource();
    bool is_output_resource = output_infos[output_index].IsResource();
    make_a_copy = is_input_resource != is_output_resource;
  }

  if (make_a_copy) {
    TensorControl* orig = it->second.tc;
    se::DeviceMemoryBase allocated =
        allocator->Allocate(ordinal, orig->size, false)
            .ConsumeValueOrDie()
            .Release();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
    if (orig->on_device) {
      auto status = executor_->MoveDeviceToHost();
      if (!status.ok()) {
        LOG(FATAL) << status.ToString();
      }
    }

    memcpy(tc->data, orig->data, orig->size);

    return se::DeviceMemoryBase(tc, tc->size);
  } else {
    // Return a reference
    TensorControl* tc = it->second.tc;
    tc->ref_count++;
    return se::DeviceMemoryBase(tc, tc->size);
  }
}

se::DeviceMemoryBase PoplarExecutor::BufferOutputAllocation::GetAllocation(
    se::DeviceMemoryAllocator* allocator, const xla::Shape& shape,
    const int64 output_index, int64& flat_tensor_index, const Args& args,
    const InputOutputAliasingMap::OutputInfo& output_info,
    const ArgsHandleMap& args_map, const int ordinal) const {
  int64 size(xla::ShapeUtil::ByteSizeOf(shape));
  if (output_info.IsResourceModified()) {
    // The output is an in-place update of one of the inputs
    // TODO: is this a multi-threading bug?
    auto it = args_map.find(
        GetInputCopyHandle(output_info.GetInputIndex(), flat_tensor_index));
    if (it == args_map.end()) {
      LOG(FATAL) << "Could not find matching input resource tensor.";
    }
    TensorControl* tc = it->second.tc;
    tc->size = size;
    tc->on_device = output_info.IsStreaming() ? false : true;
    tc->ref_count++;
    tc->output_handle = output_info.Handles().at(flat_tensor_index);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return se::DeviceMemoryBase(tc);
  } else {
    // The output is not one of the inputs
    se::DeviceMemoryBase allocated =
        allocator->Allocate(ordinal, size, false).ConsumeValueOrDie().Release();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
    tc->size = size;
    tc->on_device = output_info.IsStreaming() ? false : true;
    tc->output_handle = output_info.Handles().at(flat_tensor_index);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return allocated;
  }
}

se::DeviceMemoryBase PoplarExecutor::HandleOutputBuffer(
    se::DeviceMemoryAllocator* allocator,
    const PoplarExecutor::OutputAllocation& allocation_info,
    const xla::Shape& shape, const int64 output_index, int64& flat_tensor_index,
    const Args& args, const InputOutputAliasingMap::OutputInfo& output_info) {
  if (!shape.IsTuple()) {
    se::DeviceMemoryBase buf = allocation_info.GetAllocation(
        allocator, shape, output_index, flat_tensor_index, args, output_info,
        args_map_, ordinal_);
    flat_tensor_index++;
    return buf;
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    se::DeviceMemoryBase allocated = allocator->Allocate(ordinal_, size, false)
                                         .ConsumeValueOrDie()
                                         .Release();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());

    void** buf = reinterpret_cast<void**>(tc->data);
    for (int64 i = 0; i < xla::ShapeUtil::TupleElementCount(shape); i++) {
      se::DeviceMemoryBase out = HandleOutputBuffer(
          allocator, allocation_info, shape.tuple_shapes(i), output_index,
          flat_tensor_index, args, output_info);
      *buf++ = out.opaque();
    }
    return se::DeviceMemoryBase(tc, size);
  }
}

se::DeviceMemoryBase PoplarExecutor::GetOutputBuffer(
    const xla::poplarplugin::PoplarExecutable& executable,
    se::DeviceMemoryAllocator* allocator,
    const PoplarExecutor::OutputAllocation& allocation_info,
    const xla::Shape& shape, const Args& args,
    const InputOutputAliasingMap& input_output_aliasing_map) {
  // Get all output shapes
  std::vector<xla::Shape> shapes;
  const int64 size = shape.IsTuple()
                         ? xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*))
                         : xla::ShapeUtil::ByteSizeOf(shape);

  if (shape.IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); i++) {
      shapes.push_back(ShapeUtil::GetTupleElementShape(shape, i));
    }
  } else {
    shapes.push_back(shape);
  }

  std::vector<void*> ptrs;
  // For all outputs
  // Call a recursive function HandleOutputBuffer for each output instruction
  const auto& outputs_info =
      executable.GetInputOutputAliasingMap().GetEntryOutputInfos();
  CHECK_EQ(outputs_info.size(), shapes.size());
  for (unsigned int idx = 0; idx < shapes.size(); idx++) {
    const auto& output_info = outputs_info[idx];
    int64 start_flat_tensor_index = 0;
    se::DeviceMemoryBase out =
        HandleOutputBuffer(allocator, allocation_info, shapes[idx], idx,
                           start_flat_tensor_index, args, output_info);
    ptrs.push_back(out.opaque());
  }
  if (shape.IsTuple()) {
    se::DeviceMemoryBase allocated = allocator->Allocate(ordinal_, size, false)
                                         .ConsumeValueOrDie()
                                         .Release();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
    void** buf = reinterpret_cast<void**>(tc->data);
    for (void* ptr : ptrs) {
      *buf++ = ptr;
    }
    return se::DeviceMemoryBase(tc, size);
  } else {
    CHECK_EQ(ptrs.size(), 1);
    return se::DeviceMemoryBase(ptrs[0]);
  }
}

// Takes a tensor and returns a pointer to a buffer with the data in the right
// format
void* PoplarExecutor::PreProcessBuffer(InputDef& id) {
  TensorControl* tc = id.tc;
  void* buf(static_cast<void*>(tc->data));
  if (id.fn != nullptr) {
    tc->converted_data = id.fn(buf, tc->size, 0);
    buf = tc->converted_data.data();
  }
  return buf;
}

// Convers the data into the right host format
void PoplarExecutor::PostProcessBuffer(TensorControl* tc) {
  if (tc->output_convertor) {
    void* buf(static_cast<void*>(tc->data));
    std::vector<char> converted = tc->output_convertor(buf, 0, tc->size);
    memcpy(buf, converted.data(), converted.size());
  }
}

StatusOr<bool> PoplarExecutor::CheckMoveDeviceToHostRequired(
    const bool engine_changed) {
  // Pull previous execution outputs back from device if:
  // a) one is on the device _and_
  // b)   the engine is changing _or_
  // c)   output buffer isn't an input to the current engine _or_
  // d)   output buffer isn't currently in the right place for the new input
  bool do_device_to_host = false;
  for (const auto& tc : allocations_) {
    if (tc->on_device == true && !tc->output_handle.empty()) {
      if (engine_changed || args_map_.count(tc->input_handle) == 0 ||
          tc != args_map_.at(tc->input_handle).tc) {
        do_device_to_host = true;
      }
    }
  }
  return do_device_to_host;
}

// Check if there is tensor/arg of current executable on device.
StatusOr<bool> PoplarExecutor::CheckAnyArgOnDevice(const Args& args) {
  for (auto& device_buffer : args) {
    const TensorControl* tc =
        reinterpret_cast<const TensorControl*>(device_buffer.opaque());

    if (tc->on_device && !tc->output_handle.empty()) {
      return true;
    }
  }
  return false;
}

StatusOr<bool> PoplarExecutor::CheckMoveHostToDeviceRequired(
    const bool engine_changed) {
  // Put resources on the device if:
  // a) the engine has changed
  // b) resource is not on the device
  // c) resource is on the device, but in the wrong place
  bool do_host_to_device = false;

  for (const auto& arg : args_map_) {
    if (!arg.second.streamed) {
      auto it =
          std::find(allocations_.begin(), allocations_.end(), arg.second.tc);
      if (it == allocations_.end()) {
        return tensorflow::errors::InvalidArgument(
            "Argument isn't allocated on device: ", (void*)arg.second.tc);
      }
      if (engine_changed || arg.second.tc->on_device == false ||
          arg.second.tc->input_handle != arg.first) {
        do_host_to_device = true;
      }
    }
  }
  return do_host_to_device;
}

void PoplarExecutor::ConnectReplicatedDeviceToHost(
    const std::string& stream_name, TensorControl* tc) {
  void* dest = static_cast<void*>(tc->data);
  const std::size_t size = tc->size;
  for (int64 replica_id = 0; replica_id < current_replication_factor_;
       ++replica_id) {
    auto callback = [dest, size, replica_id](void* ptr) {
      if (replica_id == 0) {
        std::memcpy(dest, ptr, size);
      }
    };

    current_engine_->connectStreamToCallback(stream_name, replica_id, callback);
  }
}

Status PoplarExecutor::MoveDeviceToHost() {
  if (UseSyntheticData()) {
    return Status::OK();
  }

  Json::Value root;
  root["tensors"] = Json::Value(Json::arrayValue);
  uint64 total_size = 0;
  uint64 total_count = 0;
  try {
    for (const auto& tc : allocations_) {
      // Set up streams
      if (tc->on_device == true && !tc->output_handle.empty()) {
        if (tc->in_remote_memory) {
          // We currently only get one copy of the buffer.
          // Note that only resource variables are on device, hence they must
          // have the input handle set too.
          CHECK(tc->input_handle.size());
          const unsigned replica_id = 0;
          current_engine_->copyFromRemoteBuffer(tc->input_handle, tc->data, 0,
                                                replica_id);
        } else {
          ConnectReplicatedDeviceToHost(tc->output_handle, tc);
        }

        Json::Value tensor;
        tensor["name"] = Json::Value(tc->output_handle);
        tensor["size"] = Json::Value::UInt64(tc->size);
        root["tensors"].append(tensor);
        total_size += tc->size;
        total_count++;
      }
    }
    root["total_size"] = Json::Value::UInt64(total_size);
    Json::StreamWriterBuilder json_builder;
    std::string json_msg = Json::writeString(json_builder, root);

    // perform device -> host read
    if (total_count > 0) {
      current_engine_->disableExecutionProfiling();
      current_engine_->run(PoplarProgramType::DEVICE_TO_HOST);
    }

    if (current_config_.profiling().enable_ipu_trace_events() &&
        current_config_.profiling().enable_io_trace()) {
      AddDeviceToHostEventRecord(json_msg);
    }

    // Post process upload
    for (const auto& tc : allocations_) {
      if (tc->on_device == true && !tc->output_handle.empty()) {
        PostProcessBuffer(tc);
      }

      tc->in_remote_memory = false;
      tc->on_device = false;
      tc->output_handle.clear();
      tc->input_handle.clear();
    }
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Device to host] ", e);
  }
  return Status::OK();
}

Status PoplarExecutor::MoveHostToDevice() {
  if (UseSyntheticData()) {
    return Status::OK();
  }
  try {
    Json::Value root;
    root["tensors"] = Json::Value(Json::arrayValue);
    uint64 total_size = 0;

    for (auto arg : args_map_) {
      TensorControl* tc = arg.second.tc;
      std::vector<std::pair<std::string, int64>> stream_list;
      void* buf(static_cast<void*>(tc->data));
      if (!arg.second.streamed) {
        buf = PreProcessBuffer(arg.second);

        if (arg.second.remote_parameter) {
          // This is a remote parameter - copy it to the remote buffer for each
          // replica.
          tc->in_remote_memory = true;
          for (int replica_id = 0; replica_id < current_replication_factor_;
               ++replica_id) {
            current_engine_->copyToRemoteBuffer(buf, arg.first, 0, replica_id);
          }
        } else {
          tc->in_remote_memory = false;
          current_engine_->connectStream(arg.first, buf);
        }

        tc->on_device = true;
        tc->input_handle = arg.first;

        Json::Value tensor;
        tensor["name"] = Json::Value(arg.first);
        tensor["size"] = Json::Value::UInt64(tc->size);
        root["tensors"].append(tensor);
        total_size += tc->size;

        stream_list.push_back(std::make_pair(arg.first, 0));
      }
    }
    root["total_size"] = Json::Value::UInt64(total_size);
    Json::StreamWriterBuilder json_builder;
    std::string json_msg = Json::writeString(json_builder, root);

    current_engine_->disableExecutionProfiling();
    current_engine_->run(PoplarProgramType::HOST_TO_DEVICE);

    if (current_config_.profiling().enable_ipu_trace_events() &&
        current_config_.profiling().enable_io_trace()) {
      AddHostToDeviceEventRecord(json_msg);
    }

    for (auto arg : args_map_) {
      TensorControl* tc = arg.second.tc;
      tc->converted_data.clear();
    }
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Host to device] ", e);
  }

  return Status::OK();
}

StatusOr<se::DeviceMemoryBase> PoplarExecutor::GetTupleBufferByIndex(
    const se::DeviceMemoryBase& base, int64 value) {
  const TensorControl* tc =
      reinterpret_cast<const TensorControl*>(base.opaque());
  void** bufs = (void**)tc->data;
  int64 size = reinterpret_cast<const TensorControl*>(bufs[value])->size;

  return se::DeviceMemoryBase(bufs[value], size);
}

void PoplarExecutor::ConnectStreamedVariablesHostToDevice() {
  // Don't connect any streams if using synthetic data
  if (UseSyntheticData()) {
    return;
  }

  for (auto arg : args_map_) {
    if (arg.second.streamed) {
      void* buf = PreProcessBuffer(arg.second);
      current_engine_->connectStream(arg.first, buf);
    }
  }
}

void PoplarExecutor::ConnectStreamedVariablesDeviceToHost() {
  // Don't connect any streams if using synthetic data
  if (UseSyntheticData()) {
    return;
  }

  for (auto output : outputs_map_) {
    if (output.second.streamed) {
      TensorControl* tc = output.second.tc;
      ConnectReplicatedDeviceToHost(output.first, tc);
    }
  }
}

void PoplarExecutor::PostProcessStreamedVariablesDeviceToHost() {
  for (auto output : outputs_map_) {
    if (output.second.streamed) {
      PostProcessBuffer(output.second.tc);
    }
  }
}

void PoplarExecutor::AboutToFreeEngine(poplar::Engine* engine) {
  if (current_engine_ != nullptr) {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (engine == current_engine_) {
      auto status = MoveDeviceToHost();
      if (!status.ok()) {
        LOG(FATAL) << status.ToString();
      }
      DeferredDeallocation();
      current_engine_ = NULL;
    }
  }
}

const int PoplarExecutor::device_ordinal() const { return ordinal_; }

poplar::DeviceManager& PoplarExecutor::GetDeviceManager() {
  static poplar::DeviceManager device_mgr =
      poplar::DeviceManager::createDeviceManager();
  return device_mgr;
}

void PoplarExecutor::CreateInfeedIterator(
    const PoplarFeedConfig& config, const std::vector<xla::Shape>& shapes,
    const tensorflow::data::IteratorContext::Params& params,
    tensorflow::FunctionLibraryRuntime* flr,
    tensorflow::data::DatasetBase* dataset) {
  auto& feed_id = config.feed_id();
  if (infeed_iterators_.contains(feed_id)) {
    LOG(FATAL) << "Infeed with id='" << feed_id
               << "' already exists. Consider changing the `feed_name` in "
                  "IPUInfeedQueue. The Poplar backend requires all infeeds in "
                  "the same TensorFlow device to have unique names.";
  } else {
    infeed_iterators_[feed_id] = absl::make_unique<InfeedIterator>(
        flr, params, dataset, cancellation_manager(), GetInfeedAllocator(),
        config.replication_factor(), shapes, feed_id);
  }
}

Status PoplarExecutor::DeleteInfeedIterator(const std::string& feed_id) {
  std::lock_guard<std::recursive_mutex> l(ipu_.Mutex());

  if (io_threads_.size()) {
    return xla::FailedPrecondition(
        "Cannot delete infeed with id='%s' while in use", feed_id.c_str());
  }

  const auto num_erased = infeed_iterators_.erase(feed_id);
  if (num_erased == 0) {
    return xla::NotFound(
        "Infeed with id='%s'. Make sure that you have run the initializer "
        "for this infeed before attempting to delete it.",
        feed_id.c_str());
  }

  return Status::OK();
}

InfeedAllocator* PoplarExecutor::GetInfeedAllocator() {
  return &infeed_allocator;
}

std::vector<std::vector<tensorflow::Tensor>>
PoplarExecutor::GetTensorsFromOutfeed(const std::string& feed_id,
                                      const PoplarFeedConfig_Mode& mode) {
  auto itr = outfeed_contexts_.find(feed_id);
  if (itr == outfeed_contexts_.end()) {
    LOG(INFO)
        << "Trying to dequeue elements from the outfeed queue with id="
        << feed_id
        << " which has not executed yet. Make sure to execute the "
           "program with the outfeed before trying to dequeue an outfeed.";
    return {};
  }
  auto& outfeed_context = itr->second;
  // Lock whilst we dequeue all the tensors.
  std::lock_guard<std::recursive_mutex> guard(outfeed_context->mutex);

  if (mode == xla::poplarplugin::PoplarFeedConfig::GetAll) {
    std::vector<std::vector<tensorflow::Tensor>> output(
        outfeed_context->io_thread_output_queues.size());
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = outfeed_context->io_thread_output_queues.back();
      outfeed_context->io_thread_output_queues.pop_back();
    }
    return output;
  } else {
    std::vector<std::vector<tensorflow::Tensor>> output(1);
    output[0] = outfeed_context->io_thread_output_queues.front();
    outfeed_context->io_thread_output_queues.clear();
    return output;
  }
}

Status PoplarExecutor::RegisterOutfeeds(const OutfeedInfos& outfeed_infos) {
  for (auto& outfeed_info : outfeed_infos) {
    auto outfeed_id = outfeed_info.config.feed_id();
    const auto existing_feed = outfeed_contexts_.find(outfeed_id);
    if (existing_feed != outfeed_contexts_.end()) {
      if (!existing_feed->second->Matches(outfeed_info)) {
        return xla::FailedPrecondition(
            "Outfeed with id='%s' already exists but with a different tensor "
            "shape. Consider changing the `feed_name` in IPUOutfeedQueue. "
            "The Poplar backend requires all outfeeds in the same TensorFlow "
            "device to have unique names.",
            outfeed_id.c_str());
      }
    } else {
      outfeed_contexts_[outfeed_id] =
          absl::make_unique<OutfeedContext>(outfeed_info);
    }
  }
  return Status::OK();
}

Status PoplarExecutor::DeleteOutfeed(const std::string& feed_id) {
  std::lock_guard<std::recursive_mutex> l(ipu_.Mutex());

  if (io_threads_.size()) {
    return xla::FailedPrecondition(
        "Cannot delete outfeed with id='%s' while in use", feed_id.c_str());
  }

  const auto num_erased = outfeed_contexts_.erase(feed_id);
  if (num_erased == 0) {
    return xla::NotFound(
        "Outfeed with id='%s'. Make sure that you have executed the program "
        "with this outfeed before attempting to delete it.",
        feed_id.c_str());
  }

  return Status::OK();
}

Status PoplarExecutor::RegisterHostEmbedding(
    const std::string& embedding_id,
    std::unique_ptr<HostEmbeddingInterface_> embedding) {
  {
    std::unique_lock<std::mutex> lk(host_embeddings_mutex_);
    host_embeddings_[embedding_id] = std::move(embedding);
  }

  host_embeddings_cv.notify_all();

  return Status::OK();
}

tensorflow::Rendezvous* PoplarExecutor::GetRendezvous() {
  return rendezvous_.get();
}

void PoplarExecutor::ConnectSeedCallback() {
  // Don't connect any streams if using synthetic data
  if (UseSyntheticData()) {
    return;
  }

  auto& generator = seed_generator_;
  for (int replica_id = 0; replica_id < current_replication_factor_;
       ++replica_id) {
    auto callback = [&generator, replica_id](void* ptr) mutable {
      reinterpret_cast<uint64_t*>(ptr)[0] = generator.Get(replica_id);
    };

    current_engine_->connectStreamToCallback(GetRandomNumberSeedStream(),
                                             replica_id, callback);
  }
}

void PoplarExecutor::ResetSeed(int seed) { seed_generator_.Seed(seed); }

std::string PoplarExecutor::GetCycleCounterStream() {
  return "__cycle_count_stream";
}

void PoplarExecutor::ConnectCycleCounterCallback() {
  if (has_cycle_counter_) {
    for (int i = 0; i < current_replication_factor_; i++) {
      current_engine_->connectStreamToCallback(
          PoplarExecutor::GetCycleCounterStream(), i, [=](void* p) {
            // Just log cyclecount for replica 0
            if (i == 0) {
              uint64_t count;
              std::memcpy(&count, p, sizeof(count));
              LOG(INFO) << "Cycle count: " << count;
            }
          });
    }
  }
}

namespace {

Status TransformEvaluatorSubOutput(ShapeIndex& shape_index, const Shape& layout,
                                   Literal& evaluator_output,
                                   std::vector<Literal>& sub_result) {
  if (layout.IsTuple()) {
    // Continue a depth-first traversal from each subshape.
    for (int64 i = 0; i < layout.tuple_shapes_size(); i++) {
      // We need to traverse the shape tree down.
      shape_index.push_back(i);
      auto& sub_shape = layout.tuple_shapes(i);
      TF_RETURN_IF_ERROR(TransformEvaluatorSubOutput(
          shape_index, sub_shape, evaluator_output, sub_result));
      shape_index.pop_back();
    }
  } else {
    // A single element.
    Literal literal(layout);
    TF_RETURN_IF_ERROR(literal.CopyFrom(evaluator_output, {}, shape_index));
    sub_result.emplace_back(std::move(literal));
  }

  return Status::OK();
}

// Transforms literal evaluate to ConstantOutputAllocation format
// when dealing with ScalarElementwiseGraph. Deals with nested tuples.
Status TransformEvaluatorOutput(const Shape& layout, Literal& evaluator_output,
                                std::vector<std::vector<Literal>>& result) {
  Shape shape = evaluator_output.shape();
  if (shape.IsTuple()) {
    ShapeIndex shape_index;
    // Start a depth-first traversal from each subshape.
    for (int64 i = 0; i < shape.tuple_shapes_size(); i++) {
      // We need to traverse the shape tree down.
      shape_index.push_back(i);
      auto& sub_shape = layout.tuple_shapes(i);
      std::vector<Literal> sub_result;
      TF_RETURN_IF_ERROR(TransformEvaluatorSubOutput(
          shape_index, sub_shape, evaluator_output, sub_result));
      result.emplace_back(std::move(sub_result));
      shape_index.pop_back();
    }
  } else {
    // A single element.
    Literal literal = evaluator_output.Clone().Relayout(layout);
    std::vector<Literal> sub_result;
    sub_result.emplace_back(std::move(literal));
    result.emplace_back(std::move(sub_result));
  }
  return Status::OK();
}
}  // namespace

// Computes vector(s) literal input for ConstantOutputAllocation when
// dealing with ScalarElementwiseGraph.
Status PoplarExecutor::LiteralEvaluateForScalarElementwiseGraph(
    xla::poplarplugin::PoplarExecutable& executable, const Args& args,
    std::vector<std::vector<Literal>>& literal_evaluate_break_down) {
  literal_evaluate_break_down.clear();
  std::vector<Literal> arg_literals;
  const auto* comp = executable.module().entry_computation();

  for (const auto& inst : comp->parameter_instructions()) {
    Literal literal(inst->shape(), true);
    const TensorControl* src_tc = reinterpret_cast<const TensorControl*>(
        args[inst->parameter_number()].opaque());
    memcpy(literal.untyped_data(), src_tc->data, src_tc->size);
    arg_literals.push_back(std::move(literal));
  }

  HloEvaluator hlo_evaluator(1);
  TF_ASSIGN_OR_RETURN(Literal literal_evaluate,
                      hlo_evaluator.Evaluate(*comp, arg_literals));

  const auto& output_shape = executable.result_shape();
  TF_RETURN_IF_ERROR(TransformEvaluatorOutput(output_shape, literal_evaluate,
                                              literal_evaluate_break_down));

  return Status::OK();
}

StatusOr<se::DeviceMemoryBase> PoplarExecutor::ExecuteEngine(
    perftools::gputools::StreamExecutor* executor,
    xla::poplarplugin::PoplarExecutable& executable,
    se::DeviceMemoryAllocator* allocator, const Args& args) {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  const auto& input_output_aliasing_map =
      executable.GetInputOutputAliasingMap();
  const auto& output_shape = executable.result_shape();
  poplar::Engine* engine = executable.Engine();

  perftools::gputools::DeviceMemoryBase retbuf;

  bool engine_changed(current_engine_ != engine);

  UpdateArgsHandleMap(args, allocator, executable);

  if (engine == NULL) {
    // An empty engine is either a graph that just passes its inputs through
    // to its outputs, or a graph which returns a constant.
    if (executable.IsConstantGraph()) {
      retbuf =
          GetOutputBuffer(executable, allocator,
                          ConstantOutputAllocation(executable.LiteralValue()),
                          output_shape, args, input_output_aliasing_map);
    } else if (executable.IsRemapGraph()) {
      RemapOutputAllocation remap(this, executable.RemapMap(),
                                  input_output_aliasing_map);
      retbuf = GetOutputBuffer(executable, allocator, remap, output_shape, args,
                               input_output_aliasing_map);
    } else if (executable.IsScalarElementwiseGraph()) {
      // If some arg are on device, move them to host.
      TF_ASSIGN_OR_RETURN(bool any_arg_on_device, CheckAnyArgOnDevice(args));
      if (any_arg_on_device) {
        TF_RETURN_IF_ERROR(MoveDeviceToHost());
      }

      std::vector<std::vector<Literal>> literal_evaluate_break_down;
      LiteralEvaluateForScalarElementwiseGraph(executable, args,
                                               literal_evaluate_break_down);

      retbuf =
          GetOutputBuffer(executable, allocator,
                          ConstantOutputAllocation(literal_evaluate_break_down),
                          output_shape, args, input_output_aliasing_map);

    } else {
      LOG(FATAL) << "Cannot construct a NULL graph.";
    }
  } else {
    if (!executable.has_module()) {
      return tensorflow::errors::InvalidArgument(
          "Executable must have an HloModule");
    }

    TF_ASSIGN_OR_RETURN(const bool move_device_to_host,
                        CheckMoveDeviceToHostRequired(engine_changed));

    if (move_device_to_host) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost());
    }

    if (engine_changed) {
      try {
        engine->load(ipu_.Device());

        current_engine_ = engine;
        current_replication_factor_ = executable.GetReplicationFactor();

        ConnectSeedCallback();
        ConnectCycleCounterCallback();

        if (current_config_.profiling().enable_ipu_trace_events() &&
            current_config_.profiling().enable_io_trace()) {
          AddLoadEngineEventRecord(executable.module().name());
        }

        executable.OnEngineLoaded();

      } catch (const std::exception& e) {
        return PoplarExceptionToTensorflowStatus("[Load engine] ", e);
      }
    }

    // Deallocate all the marked buffers.
    DeferredDeallocation();

    TF_ASSIGN_OR_RETURN(const bool move_host_to_device,
                        CheckMoveHostToDeviceRequired(engine_changed));
    if (move_host_to_device) {
      TF_RETURN_IF_ERROR(MoveHostToDevice());
    }

    // Outfeeds add empty tuples as output shape, no need to get an output
    // buffer in this case
    if (ShapeUtil::IsEmptyTuple(output_shape)) {
      outputs_map_.clear();
    } else {
      retbuf = GetOutputBuffer(executable, allocator, BufferOutputAllocation(),
                               output_shape, args, input_output_aliasing_map);

      UpdateOutputsHandleMap(executable, output_shape, retbuf);
    }

    VLOG(1) << "Executing on poplar stream ordinal " << ordinal_ << " of type "
            << GetDeviceTargetName();

    // Create any outfeed queues which do not already exist
    TF_RETURN_IF_ERROR(RegisterOutfeeds(executable.GetOutfeedInfos()));

    // Create our own free list which we use to allocate all the memory used by
    // all the tensors.
    std::list<std::unique_ptr<char[]>> memory_buffer;

    // Allocate the parameters for each of the functors, sorted by the user
    // instruction which they are created for.
    std::unordered_map<const HloInstruction*, std::vector<void*>> in_buffers;
    std::unordered_map<const HloInstruction*, std::vector<std::uint32_t>>
        in_sizes;
    std::unordered_map<const HloInstruction*, std::vector<void*>> out_buffer;

    try {
      // Connect the streams to and from the device
      ConnectStreamedVariablesHostToDevice();
      ConnectStreamedVariablesDeviceToHost();
      const StreamInfos& stream_infos = executable.GetStreamInfos();

      // If this is a user op copy the buffers.
      // We add one call back to the stream which allocates the buffers and once
      // all buffers have been allocated finally calls down to the user
      // operation.
      for (auto& pair : executable.GetStreamMetaInfos()) {
        StreamCopyMetaInfo infos = pair.second;

        const HloInstruction* instruction = infos.parent_instruction;

        out_buffer[instruction].resize(infos.output_stream_info.size());

        // Resize the input vectors to be the number of inputs in advance.
        in_buffers[instruction].resize(infos.num_inputs);
        in_sizes[instruction].resize(infos.num_inputs);

        // For each of the output stream copies allocate a buffer.
        for (StreamCopyInfo* stream_copy : infos.output_stream_info) {
          assert(stream_copy->operand_number <
                     infos.output_stream_info.size() &&
                 "Operand ID is greater than the number of output streams "
                 "StreamCopyMetaInfo can see.");

          const std::uint32_t totalSize =
              stream_copy->size_of_element * stream_copy->number_of_elements;
          memory_buffer.push_back(std::unique_ptr<char[]>(new char[totalSize]));

          out_buffer[instruction][stream_copy->operand_number] =
              (void*)memory_buffer.back().get();
        }
      }

      // The send/recv callbacks only need to be re-connected when the engine
      // has changed as they do not depend on any external state and are
      // designed to be re-used.
      if (engine_changed) {
        TF_RETURN_IF_ERROR(
            ConnectSendCallbacksToRendezvous(executable.GetSendInfos()));
        TF_RETURN_IF_ERROR(
            ConnectRecvCallbacksToRendezvous(executable.GetRecvInfos()));
      }

      const auto& infeed_infos = executable.GetInfeedInfos();
      if (!infeed_infos.empty()) {
        ConnectInfeedsToStreamCallback(infeed_infos);
      }

      for (auto& host_embedding_lookup_info :
           executable.GetHostEmbeddingLookupInfos()) {
        TF_RETURN_IF_ERROR(
            ConnectHostEmbeddingLookupToRendezvous(host_embedding_lookup_info));
      }

      for (auto& host_embedding_update_info :
           executable.GetHostEmbeddingUpdateInfos()) {
        ConnectHostEmbeddingUpdateToRendezvous(host_embedding_update_info);
      }

      const auto& outfeed_infos = executable.GetOutfeedInfos();
      if (!outfeed_infos.empty()) {
        ConnectOutfeedToStreamCallback(outfeed_infos);
      }

      for (auto& pair : stream_infos) {
        const std::string name = pair.first;
        const std::list<StreamCopyInfo>& list = pair.second;

        // Track how many inputs have been initalized so far.
        std::uint32_t number_of_inputs_initalized = 0;

        // For all of the stream copies, both inputs and outputs.
        for (const StreamCopyInfo& info : list) {
          StreamCopyInfo::FunctionTy functor = info.callback_to_register;

          // If there is a functor then this is an input tensor, we will attach
          // the callbacks to the stream otherwise just copy into the previously
          // allocated pegged memory.
          if (functor != nullptr) {
            // Create a custom callback which we use to copy the inputs as
            // these callbacks are called in a random order we have to work
            // out which tensor we are writing into and we have to check how
            // many inputs we have already initialized so we know to call the
            // user provided operation once they have all been set up.
            auto callback = [&, functor](void* buffer) {
              std::vector<void*>& in_buffer =
                  in_buffers[info.parent_instruction];
              std::vector<std::uint32_t>& in_size =
                  in_sizes[info.parent_instruction];

              // Allocate space for the input tensor and then memcopy into it.
              // The 'buffer' pointer is only garunteed to be alive for the
              // duration of this callback.
              std::uint32_t totalSize =
                  info.size_of_element * info.number_of_elements;
              memory_buffer.push_back(
                  std::unique_ptr<char[]>(new char[totalSize]));
              in_buffer[info.operand_number] =
                  (void*)memory_buffer.back().get();

              // Copy into the newly allocated memory.
              std::memcpy((char*)in_buffer[info.operand_number], (char*)buffer,
                          totalSize);
              number_of_inputs_initalized++;

              // Store the size of each input.
              in_size[info.operand_number] = info.number_of_elements;

              // These callbacks are called in a random order by poplar so we
              // need to only call the user provided callback once, after all of
              // the data has been initialized.
              if (number_of_inputs_initalized == in_buffer.size()) {
                functor(in_buffer, in_size,
                        out_buffer[info.parent_instruction]);
              }
            };

            current_engine_->connectStreamToCallback(info.stream_handle,
                                                     callback);
          } else {
            // Connect the output stream to the correct pre-allocated buffer.
            current_engine_->connectStream(
                info.stream_handle,
                out_buffer[info.parent_instruction][info.operand_number]);
          }
        }
      }
      // Launch the IO threads when we are not using synthetic data.
      if (!UseSyntheticData()) {
        LaunchIOThreads(infeed_infos, outfeed_infos);
      }

      // Before executing the main program, prepare the random seeds for each
      // replica.
      seed_generator_.PrepareSeedsForReplicas(current_replication_factor_);

      // Run the main engine
      current_engine_->enableExecutionProfiling();
      current_engine_->run(PoplarProgramType::MAIN_SEQUENCE);

      // Stop the IO threads when we are not using synthetic data.
      if (!UseSyntheticData()) {
        StopIOThreads();
      }

      // We need to call post process to make sure all the data is in the
      // right format on the host
      PostProcessStreamedVariablesDeviceToHost();
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Execute engine] ", e);
    }

    try {
      if (!PoplarXlaFlags::Get().save_interval_report.empty() &&
          executable.ExecutionCount() == 0) {
        auto filename =
            tensorflow::io::JoinPath(PoplarXlaFlags::Get().save_interval_report,
                                     executable.module().name() + ".csv");
        VLOG(1) << "Dumping interval report " << filename;
        std::ofstream stream(filename);
        current_engine_->reportIntervals(stream);
      }

      if (current_config_.profiling().enable_ipu_trace_events()) {
        std::string report;
        if (current_config_.profiling().execution_trace_type() !=
            IpuExecutionProfileType::NO_PROFILE) {
          if (executable.ExecutionCount() == 0 &&
              !executable.IsLoadedFromCache()) {
            std::stringstream report_stream;
            auto graph_profile = current_engine_->getGraphProfile();
            auto exec_profile = current_engine_->getExecutionProfile();

            if (PoplarXlaFlags::Get().dump_text_reports_to_stdio) {
              auto opts = GetReportExecutionFlags();
              SetFlagIfNotPresent(opts, "showExecutionSteps", "true");
              poplar::printExecutionSummary(std::cout, graph_profile,
                                            exec_profile, opts);
            }

            if (CompilerReportingTextFormat()) {
              auto opts = GetReportExecutionFlags();
              SetFlagIfNotPresent(opts, "showExecutionSteps", "true");

              poplar::printExecutionSummary(report_stream, graph_profile,
                                            exec_profile, opts);
            } else if (CompilerReportingCborFormat()) {
              poplar::serializeToCBOR(report_stream, exec_profile);
            } else {
              poplar::serializeToJSON(report_stream, exec_profile);
            }

            current_engine_->resetExecutionProfile();

            if (report_stream.tellp() > MaxReportSize()) {
              LOG(INFO)
                  << "Dropping a Poplar compilation report of size "
                  << report_stream.tellp()
                  << " which is larger than the configured maximum report size "
                  << std::to_string(MaxReportSize())
                  << ". To change the maximum report size use the "
                     "max_report_size"
                  << " argument in ipu.utils.create_ipu_config.\n"
                  << "Example:\n"
                  << "cfg = "
                     "ipu.utils.create_ipu_config(max_report_size=0x100000000) "
                  << "Note that the max report size is in bytes.";
              report_stream.str(std::string());
            }
            report = report_stream.str();
          }
        }

        AddExecuteEventRecord(executable.module().name(), report);
      }
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Execute engine] ", e);
    }
  }

  return retbuf;
}

}  // namespace poplarplugin
}  // namespace xla
