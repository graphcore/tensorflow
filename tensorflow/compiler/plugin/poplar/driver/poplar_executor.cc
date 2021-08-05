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

#include <algorithm>
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
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
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
  hashable_config.mutable_profiling()->clear_enable_poplar_reports_text();
  hashable_config.mutable_profiling()->clear_report_every_nth_execution();
  hashable_config.mutable_profiling()->clear_enable_ipu_trace_events();
  hashable_config.mutable_profiling()->clear_enable_poplar_reports_cbor();
  hashable_config.mutable_profiling()->clear_report_directory();
  hashable_config.mutable_profiling()->clear_max_report_size();
  hashable_config.clear_device_connection_type();
  hashable_config.mutable_device_config()->Clear();
  hashable_config.clear_multi_replica_process_count();
  hashable_config.clear_multi_replica_process_index();

  // Clear the target options set by `IPUConfig.device_connection` already
  // included in the target hash. This allows for doing offline compilation
  // and then loading the executable onto a hardware device, given that it has
  // the same target configuration as the offline compilation target.
  hashable_config.clear_ipu_version();
  hashable_config.clear_enable_remote_buffers_without_device();

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
  for (const char* name : {"GCL_LIBRARY_PATH", "GCL_GP_PATH"}) {
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

// Create a new target based on `target`, but with a total number of IPUs
// that is multiplied by the `process_count`. This is used for multi-replica
// distribution, where we compile against a `poplar::Target` with the total
// global number of IPUs, but execute on the subset of those IPUs that belong
// to the current process using the Poplar "runtime replica subset" feature.
poplar::Target CreateMultiReplicaDistributionTarget(
    const poplar::Target& target, int64 process_count) {
  const int64 num_ipus = target.getNumIPUs();
  CHECK_GT(process_count, 0);
  const int64 global_num_ipus = process_count * num_ipus;
  return poplar::Target::createIPUTarget(
      global_num_ipus, target.getTargetArchString(), target.getTargetOptions());
}

struct UserOpsExecutionState {
  explicit UserOpsExecutionState(const StreamMetaInfos& stream_meta_infos) {
    // We add one call back to the stream which allocates the buffers and
    // once all buffers have been allocated finally calls down to the user
    // operation.
    for (auto& pair : stream_meta_infos) {
      StreamCopyMetaInfo infos = pair.second;

      const HloInstruction* instruction = infos.parent_instruction;

      out_buffer[instruction].resize(infos.output_stream_info.size());

      // Resize the input vectors to be the number of inputs in advance.
      in_buffers[instruction].resize(infos.num_inputs);
      in_sizes[instruction].resize(infos.num_inputs);

      // For each of the output stream copies allocate a buffer.
      for (StreamCopyInfo* stream_copy : infos.output_stream_info) {
        CHECK_LT(stream_copy->operand_number, infos.output_stream_info.size())
            << "Operand ID is greater than the number of output streams "
               "StreamCopyMetaInfo can see.";

        const std::uint32_t total_size =
            stream_copy->size_of_element * stream_copy->number_of_elements;
        memory_buffer.push_back(std::unique_ptr<char[]>(new char[total_size]));

        out_buffer[instruction][stream_copy->operand_number] =
            static_cast<void*>(memory_buffer.back().get());
      }
    }
  }

  void ConnectStream(const StreamCopyInfo& info, poplar::Engine& engine) {
    // If there is a functor then this is an input tensor, we will
    // attach the callbacks to the stream otherwise just copy into the
    // previously allocated pegged memory.
    if (info.callback_to_register != nullptr) {
      engine.connectStreamToCallback(info.stream_handle, CreateCallback(info));
    } else {
      // Connect the output stream to the correct pre-allocated buffer.
      engine.connectStream(
          info.stream_handle,
          out_buffer[info.parent_instruction][info.operand_number]);
    }
  }

 private:
  poplar::StreamCallbackHandle CreateCallback(const StreamCopyInfo& info) {
    // Create a custom callback which we use to copy the inputs as
    // these callbacks are called in a random order we have to work
    // out which tensor we are writing into and we have to check how
    // many inputs we have already initialized so we know to call the
    // user provided operation once they have all been set up.
    return [this, info](void* buffer) {
      StreamCopyInfo::FunctionTy functor = info.callback_to_register;

      std::vector<void*>& in_buffer = in_buffers[info.parent_instruction];
      std::vector<std::uint32_t>& in_size = in_sizes[info.parent_instruction];
      std::uint32_t& number_of_inputs_initialized =
          numbers_of_inputs_initialized[info.parent_instruction];

      // Allocate space for the input tensor and then memcopy into it.
      // The 'buffer' pointer is only garunteed to be alive for the
      // duration of this callback.
      std::uint32_t totalSize = info.size_of_element * info.number_of_elements;
      memory_buffer.push_back(std::unique_ptr<char[]>(new char[totalSize]));
      in_buffer[info.operand_number] =
          static_cast<void*>(memory_buffer.back().get());

      // Copy into the newly allocated memory.
      std::memcpy(in_buffer[info.operand_number], buffer, totalSize);
      number_of_inputs_initialized++;

      // Store the size of each input.
      in_size[info.operand_number] = info.number_of_elements;

      // These callbacks are called in a random order by poplar so we
      // need to only call the user provided callback once, after all
      // of the data has been initialized.
      if (number_of_inputs_initialized == in_buffer.size()) {
        functor(in_buffer, in_size, out_buffer[info.parent_instruction]);
      }
    };
  }

  // Create our own free list which we use to allocate all the memory used
  // by all the tensors.
  std::list<std::unique_ptr<char[]>> memory_buffer;

  // Allocate the parameters for each of the functors, sorted by the user
  // instruction which they are created for.
  std::unordered_map<const HloInstruction*, std::vector<void*>> in_buffers;
  std::unordered_map<const HloInstruction*, std::vector<std::uint32_t>>
      in_sizes;
  std::unordered_map<const HloInstruction*, std::vector<void*>> out_buffer;

  // Track how many inputs have been initialized so far.
  std::unordered_map<const HloInstruction*, uint32_t>
      numbers_of_inputs_initialized;
};

std::string GetExecutableCachePath() {
  // Lazily find the path the first time it is requested.
  static const auto path = []() -> std::string {
    const std::string flag = PoplarXlaFlags::Get().executable_cache_path;
    if (!flag.empty()) {
      return flag;
    }

    const char* env_var = std::getenv("POPDIST_EXECUTABLE_CACHE_PATH");
    if (env_var) {
      return env_var;
    }

    return "";
  }();

  return path;
}

std::unique_ptr<SeedGenerator> CreateDefaultSeedGenerator() {
  static std::random_device rd;
  return absl::make_unique<DistinctReplicaSeedGenerator>(rd());
}

}  // namespace

PoplarExecutor::ArgHandle::ArgHandle(int64 parameter_index,
                                     int64 flat_tensor_index)
    : parameter_index(parameter_index), flat_tensor_index(flat_tensor_index) {}

PoplarExecutor::ArgHandle::ArgHandle(int64 parameter_index,
                                     int64 flat_tensor_index,
                                     const std::string& name)
    : parameter_index(parameter_index),
      flat_tensor_index(flat_tensor_index),
      name(name) {}

bool PoplarExecutor::ArgHandle::operator==(const ArgHandle& rhs) const {
  return (parameter_index == rhs.parameter_index) &&
         (flat_tensor_index == rhs.flat_tensor_index);
}

bool PoplarExecutor::ArgHandle::operator!=(const ArgHandle& rhs) const {
  return !(*this == rhs);
}

bool PoplarExecutor::ArgHandle::operator<(const ArgHandle& rhs) const {
  if (parameter_index < rhs.parameter_index) {
    return true;
  }

  if (parameter_index > rhs.parameter_index) {
    return false;
  }

  if (flat_tensor_index < rhs.flat_tensor_index) {
    return true;
  }

  if (flat_tensor_index > rhs.flat_tensor_index) {
    return false;
  }

  return false;
}

PoplarExecutor::TensorControl::TensorControl(size_t size_) {
  size = size_;
  ref_count = 1;
  on_device = false;
  input_handle.reset();
  output_handle.reset();
  output_convertor = nullptr;
  converted_data.clear();
  data = static_cast<char*>(tensorflow::port::AlignedMalloc(size_, 64));
}

std::size_t PoplarExecutor::TensorControl::GetRemoteBufferSize() const {
  if (host_rearrangement) {
    return ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(
        element_type, {host_rearrangement->replicationFactor,
                       host_rearrangement->totalElementsPerReplica}));
  }
  return size;
}

PoplarExecutor::TensorControl::~TensorControl() {
  tensorflow::port::AlignedFree(data);
}

PoplarExecutor::OutfeedContext::OutfeedContext(const PoplarFeedConfig& config,
                                               const Shape& shape,
                                               int64 replication_factor)
    : config(config),
      replication_factor(replication_factor),
      shapes(GetOutfeedShapes(FlattenedXlaShape(shape), replication_factor)),
      tf_data_types(config.tf_data_types().size()),
      tf_shapes(shapes.size()),
      callback_to_io_thread_queues(shapes.size()) {
  CHECK_EQ(shapes.size(), tf_data_types.size());
  for (uint64 i = 0; i < shapes.size(); i++) {
    tf_data_types[i] =
        static_cast<tensorflow::DataType>(config.tf_data_types()[i]);
    tensorflow::XLAShapeToTensorShape(shapes[i], &tf_shapes[i]);

    // Set up the queue per tensor per replica.
    CHECK_GT(replication_factor, 0);
    const int64 num_bytes_per_replica =
        ShapeUtil::ByteSizeOf(shapes[i]) / replication_factor;
    for (int64 replica_id = 0; replica_id < replication_factor; replica_id++) {
      void* buffer =
          tensorflow::port::AlignedMalloc(sizeof(OutfeedQueueType), 64);
      callback_to_io_thread_queues[i].emplace_back(
          new (buffer) OutfeedQueueType(num_bytes_per_replica),
          [](OutfeedQueueType* ptr) {
            ptr->~OutfeedQueueType();
            tensorflow::port::AlignedFree(ptr);
          });
    }
  }
}

bool PoplarExecutor::OutfeedContext::Matches(
    const TranslatedFeedInfo& other, int64 other_replication_factor) const {
  const auto s = GetOutfeedShapes(FlattenedXlaShape(other.canonical_info.shape),
                                  other_replication_factor);
  if (s != shapes) {
    return false;
  }
  return google::protobuf::util::MessageDifferencer::Equivalent(
      config, other.canonical_info.config);
}

PoplarExecutor::PoplarExecutor()
    : ordinal_(0),
      current_engine_(nullptr),
      poplar_device_hash_(0),
      configured_(false),
      seed_generator_(CreateDefaultSeedGenerator()),
      rendezvous_(tensorflow::NewLocalRendezvous()) {
  TENSORFLOW_TRACEPOINT();
}

PoplarExecutor::~PoplarExecutor() { TENSORFLOW_TRACEPOINT(); }

Status PoplarExecutor::GetAndResetExecutorStatus() {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  const Status status = current_status_;
  current_status_ = Status::OK();
  return status;
}

void* PoplarExecutor::Allocate(uint64 size) {
  TENSORFLOW_TRACEPOINT();
  TensorControl* allocated = new TensorControl(size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    allocations_.push_back(allocated);
  }
  return allocated;
}

void* PoplarExecutor::GetSubBuffer(se::DeviceMemoryBase* parent,
                                   uint64 offset_bytes, uint64 size_bytes) {
  TENSORFLOW_TRACEPOINT();
  TensorControl* tc = reinterpret_cast<TensorControl*>(parent->opaque());
  return tc->data + offset_bytes;
}

void PoplarExecutor::Deallocate(se::DeviceMemoryBase* mem) {
  TENSORFLOW_TRACEPOINT();
  TensorControl* tc = reinterpret_cast<TensorControl*>(mem->opaque());
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (tc->ref_count > 0) {
      tc->ref_count--;
    }
  }
}

/*static*/ Status PoplarExecutor::IncrementBufferReferenceCount(
    const se::DeviceMemoryBase& buffer, const Shape& shape) {
  return ShapeUtil::ForEachSubshapeWithStatus(shape, [buffer](
                                                         const Shape& subshape,
                                                         const ShapeIndex&
                                                             index) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase subbuffer,
                        PoplarExecutor::GetBufferByShapeIndex(buffer, index));
    TensorControl* tc = reinterpret_cast<TensorControl*>(subbuffer.opaque());
    if (tc->ref_count < 1) {
      return FailedPrecondition(
          "Trying to increment a reference counter for a deallocated buffer.");
    }
    tc->ref_count++;
    return Status::OK();
  });
}

/*static*/ Status PoplarExecutor::DecrementBufferReferenceCount(
    const se::DeviceMemoryBase& buffer, const Shape& shape) {
  return ShapeUtil::ForEachSubshapeWithStatus(shape, [buffer](
                                                         const Shape& subshape,
                                                         const ShapeIndex&
                                                             index) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase subbuffer,
                        PoplarExecutor::GetBufferByShapeIndex(buffer, index));
    TensorControl* tc = reinterpret_cast<TensorControl*>(subbuffer.opaque());
    if (tc->ref_count < 1) {
      return FailedPrecondition(
          "Trying to decrement a reference counter for a deallocated buffer.");
    }
    tc->ref_count--;
    return Status::OK();
  });
}

Status PoplarExecutor::ConnectSendCallbacksToRendezvous(
    const SendRecvInfos& send_infos) {
  TENSORFLOW_TRACEPOINT();
  if (send_infos.empty()) {
    return Status::OK();
  }

  const int64 num_replicas = current_replication_factor_;

  for (const SendRecvInfo& send : send_infos) {
    VLOG(1) << "Connecting Poplar IPU->host stream to rendezvous key '"
            << send.rendezvous_key << "' with shape " << send.shape;

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

    for (int64 replica_id = 0; replica_id < num_replicas; ++replica_id) {
      if (replica_id == 0) {
        current_engine_->connectStreamToCallback(
            send.stream_handle, replica_id,
            [rendezvous, key, type, shape](void* src) {
              auto tensor = tensorflow::Tensor(type, shape);
              auto* dst = tensorflow::DMAHelper::buffer(&tensor);
              std::memcpy(dst->data(), src, dst->size());
              rendezvous->Send(key, tensorflow::Rendezvous::Args{}, tensor,
                               /*is_dead=*/false);
            });
      } else {
        // Discard the output from the remaining replicas.
        current_engine_->connectStreamToCallback(send.stream_handle, replica_id,
                                                 [](void*) {});
      }
    }
  }

  return Status::OK();
}

Status PoplarExecutor::ConnectRecvCallbacksToRendezvous(
    const SendRecvInfos& recv_infos) {
  TENSORFLOW_TRACEPOINT();
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

Status PoplarExecutor::ConnectHostEmbeddingLookup(
    const HostEmbeddingInfo& lookup_info,
    HostEmbeddingInterface_* embedding_interface) {
  TENSORFLOW_TRACEPOINT();
  if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding)) {
    return Status::OK();
  }

  // Extract the shapes and types.
  tensorflow::TensorShape indices_shape;
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(
      lookup_info.indices_shape, &indices_shape));

  if (EnableExperimentalRemoteBufferEmbedding()) {
    TF_ASSIGN_OR_RETURN(int token_count, embedding_interface->GetTokenCount());
    TF_ASSIGN_OR_RETURN(int encoding_width,
                        embedding_interface->GetEncodingWidth());

    // Copy the tokens into the approriate replica remote buffer.
    if (lookup_info.strategy == HostEmbeddingSplittingStrategy::Token) {
      for (int i = 0; i < token_count; i++) {
        TF_ASSIGN_OR_RETURN(void* token, embedding_interface->GetRow(i));

        current_engine_->copyToRemoteBuffer(token, lookup_info.embedding_id,
                                            i / current_replication_factor_,
                                            i % current_replication_factor_);
      }

      return Status::OK();
    }

    // Copy the token encoding slices into the approriate replica remote buffer.
    if (lookup_info.strategy == HostEmbeddingSplittingStrategy::Encoding) {
      TF_ASSIGN_OR_RETURN(int element_size,
                          embedding_interface->GetElementSize());
      int replica_encoding_width = encoding_width / current_replication_factor_;
      int replica_encoding_width_padding =
          (encoding_width + current_replication_factor_ - 1) /
          current_replication_factor_;

      // We need a temporary buffer to allow for padding.
      auto tmp_buffer = absl::make_unique<unsigned char[]>(
          replica_encoding_width_padding * element_size);

      for (int i = 0; i < token_count; i++) {
        TF_ASSIGN_OR_RETURN(void* token, embedding_interface->GetRow(i));

        for (int r = 0; r < current_replication_factor_; ++r) {
          char* src = static_cast<char*>(token) +
                      r * replica_encoding_width * element_size;

          std::memcpy(tmp_buffer.get(), src,
                      replica_encoding_width * element_size);

          current_engine_->copyToRemoteBuffer(tmp_buffer.get(),
                                              lookup_info.embedding_id, i,
                                              i % current_replication_factor_);
        }
      }

      return Status::OK();
    }

    return xla::FailedPrecondition("Unknown host embedding splitting strategy");
  }

  for (int replica = 0;
       replica < std::max<int64>(1, current_replication_factor_); ++replica) {
    // Connect the indices callback.
    current_engine_->connectStreamToCallback(
        lookup_info.stream_handle + lookup_info.embedding_id + "_indices",
        replica, [replica, indices_shape, embedding_interface](void* ptr) {
          embedding_interface->EnqueueLookupIndices(
              replica, static_cast<int*>(ptr), indices_shape.num_elements());
        });

    // Connect the grads callback.
    current_engine_->connectStreamToCallback(
        lookup_info.stream_handle + lookup_info.embedding_id + "_activations",
        replica, [replica, embedding_interface](void* ptr) {
          embedding_interface->DequeueLookupActivations(replica, ptr);
        });
  }

  return Status::OK();
}

Status PoplarExecutor::ConnectHostEmbeddingUpdateToRendezvous(
    const HostEmbeddingInfo& update_info,
    HostEmbeddingInterface_* embedding_interface) {
  TENSORFLOW_TRACEPOINT();
  if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding)) {
    return Status::OK();
  }

  if (EnableExperimentalRemoteBufferEmbedding()) {
    return Status::OK();
  }

  // Extract the shapes and types.
  tensorflow::TensorShape indices_shape;
  TF_RETURN_IF_ERROR(tensorflow::XLAShapeToTensorShape(
      update_info.indices_shape, &indices_shape));

  for (int replica = 0;
       replica < std::max<int64>(1, current_replication_factor_); ++replica) {
    // Connect the indices callback.
    current_engine_->connectStreamToCallback(
        update_info.stream_handle + update_info.embedding_id + "_indices",
        replica, [replica, indices_shape, embedding_interface](void* ptr) {
          embedding_interface->EnqueueUpdateIndices(
              replica, static_cast<int*>(ptr), indices_shape.num_elements());
        });

    // Connect the grads callback.
    current_engine_->connectStreamToCallback(
        update_info.stream_handle + update_info.embedding_id + "_grads",
        replica, [replica, embedding_interface](void* ptr) {
          embedding_interface->EnqueueUpdateGrads(replica, ptr);
        });
  }

  return Status::OK();
}

Status PoplarExecutor::ConnectHostEmbeddingNotify(
    const HostEmbeddingInfo& notify_info,
    HostEmbeddingInterface_* embedding_interface) {
  TENSORFLOW_TRACEPOINT();
  if (UseSyntheticDataFor(SyntheticDataCategory::HostEmbedding)) {
    return Status::OK();
  }

  if (EnableExperimentalRemoteBufferEmbedding()) {
    return Status::OK();
  }

  for (int replica = 0;
       replica < std::max<int64>(1, current_replication_factor_); ++replica) {
    // Connect the notify callback.
    current_engine_->connectStreamToCallback(
        notify_info.stream_handle + notify_info.embedding_id + "_notify",
        replica, [replica, embedding_interface](void* ptr) {
          embedding_interface->Notify(replica);
        });
  }

  return Status::OK();
}

Status PoplarExecutor::DisconnectHostEmbeddingLookup(
    const HostEmbeddingInfo& lookup_info,
    HostEmbeddingInterface_* embedding_interface) {
  TENSORFLOW_TRACEPOINT();
  if (EnableExperimentalRemoteBufferEmbedding()) {
    TF_ASSIGN_OR_RETURN(int token_count, embedding_interface->GetTokenCount());
    TF_ASSIGN_OR_RETURN(int encoding_width,
                        embedding_interface->GetEncodingWidth());

    if (lookup_info.strategy == HostEmbeddingSplittingStrategy::Token) {
      for (int i = 0; i < token_count; i++) {
        TF_ASSIGN_OR_RETURN(void* token, embedding_interface->GetRow(i));

        current_engine_->copyFromRemoteBuffer(lookup_info.embedding_id, token,
                                              i / current_replication_factor_,
                                              i % current_replication_factor_);
      }

      return Status::OK();
    }

    if (lookup_info.strategy == HostEmbeddingSplittingStrategy::Encoding) {
      TF_ASSIGN_OR_RETURN(int element_size,
                          embedding_interface->GetElementSize());
      int replica_encoding_width = encoding_width / current_replication_factor_;
      int replica_encoding_width_padding =
          (encoding_width + current_replication_factor_ - 1) /
          current_replication_factor_;

      auto tmp_buffer = absl::make_unique<unsigned char[]>(
          replica_encoding_width_padding * element_size);

      for (int i = 0; i < token_count; i++) {
        TF_ASSIGN_OR_RETURN(void* token, embedding_interface->GetRow(i));

        for (int r = 0; r < current_replication_factor_; ++r) {
          current_engine_->copyFromRemoteBuffer(
              lookup_info.embedding_id, tmp_buffer.get(), i,
              i % current_replication_factor_);

          char* dst = static_cast<char*>(token) +
                      r * replica_encoding_width * element_size;

          std::memcpy(dst, tmp_buffer.get(),
                      replica_encoding_width * element_size);
        }
      }

      return Status::OK();
    }

    return xla::FailedPrecondition("Unknown host embedding splitting strategy");
  }

  return Status::OK();
}

Status PoplarExecutor::DisconnectHostEmbeddingUpdate(
    const HostEmbeddingInfo& update_info,
    HostEmbeddingInterface_* embedding_interface) {
  return Status::OK();
}

Status PoplarExecutor::DisconnectHostEmbeddingNotify(
    const HostEmbeddingInfo& notify_info,
    HostEmbeddingInterface_* embedding_interface) {
  return Status::OK();
}

namespace {
class InfeedPrefetchCallback : public poplar::StreamCallback {
 public:
  InfeedPrefetchCallback(InfeedQueue* queue, uint64 num_bytes)
      : queue_(queue), num_bytes_(num_bytes), look_ahead_(0) {}

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    tensorflow::TensorBuffer* buffer = nullptr;
    // Try to get a value from the queue.
    if (queue_->TryPop(buffer, look_ahead_)) {
      std::memcpy(dest, buffer->data(), num_bytes_);
      look_ahead_++;
      return poplar::StreamCallback::Result::Success;
    } else {
      return poplar::StreamCallback::Result::NotAvailable;
    }
  }

  void fetch(void* dest) noexcept override {
    tensorflow::TensorBuffer* buffer = nullptr;
    if (!queue_->BlockPop(buffer, look_ahead_)) {
      LOG(FATAL) << "Infeed dataset iterator out of range. Are you trying to "
                 << "dequeue more elements than are in the dataset?";
    }

    std::memcpy(dest, buffer->data(), num_bytes_);
    look_ahead_++;
  }

  void complete() noexcept override {
    queue_->AdvanceReadPosition();
    CHECK_GE(look_ahead_--, 1);
  }

 private:
  InfeedQueue* queue_;
  const uint64 num_bytes_;
  alignas(64) std::atomic<std::size_t> look_ahead_;
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
    const TranslatedInfeedInfos& infeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Infeed)) {
    return;
  }

  std::unique_lock<std::mutex> l(infeeds_mutex_);
  for (const auto& infeed_info : infeed_infos) {
    auto itr = infeed_iterators_.find(infeed_info.stream_prefix);
    if (itr == infeed_iterators_.end()) {
      LOG(FATAL) << "Trying to access an infeed dataset iterator which has not "
                    "been created."
                 << " Did you initialize the infeed_queue '"
                 << infeed_info.stream_prefix << "'?";
    }
    auto* infeed_dataset_iterator = itr->second.get();
    auto& shapes = infeed_dataset_iterator->GetShapes();
    auto& queues = infeed_dataset_iterator->GetInfeedQueues();
    CHECK_EQ(queues.size(), current_replication_factor_);

    for (auto replica_id = 0; replica_id < current_replication_factor_;
         ++replica_id) {
      auto& replica_queues = queues[replica_id];
      CHECK_EQ(replica_queues.size(), shapes.size());
      for (size_t j = 0; j < shapes.size(); ++j) {
        const auto bytes = ShapeUtil::ByteSizeOf(shapes[j]);
        std::unique_ptr<poplar::StreamCallback> infeed_callback;
        if (PoplarXlaFlags::Get().null_data_feed) {
          infeed_callback = absl::make_unique<NullPrefetchCallback>(
              GetInfeedAllocator(), bytes);
        } else {
          infeed_callback = absl::make_unique<InfeedPrefetchCallback>(
              replica_queues[j], bytes);
        }
        current_engine_->connectStreamToCallback(
            GetInfeedCopyHandle(infeed_info.canonical_info.config.feed_id(), j),
            replica_id, std::move(infeed_callback));
      }
    }
  }
}

Status PoplarExecutor::SetupInfeedReplication(
    const TranslatedInfeedInfos& infeed_infos) {
  std::unique_lock<std::mutex> l(infeeds_mutex_);
  for (auto& infeed_info : infeed_infos) {
    const int64 replication_factor = current_replication_factor_;
    const std::string& feed_id = infeed_info.stream_prefix;

    auto iter = infeed_iterators_.find(feed_id);
    if (iter == infeed_iterators_.end()) {
      return FailedPrecondition(
          "Trying to access an infeed dataset iterator which has not been"
          " created. Did you initialize the infeed_queue '%s' ?",
          feed_id);
    }

    auto* infeed_iterator = iter->second.get();
    if (!infeed_iterator->HasReplicationFactor()) {
      infeed_iterator->SetReplicationFactor(replication_factor);
    } else if (infeed_iterator->ReplicationFactor() != replication_factor) {
      return FailedPrecondition(
          "Iterator for feed %s has already been setup with a replication "
          "factor of %d when it should have %d."
          "Infeeds can not be shared across graphs with different "
          "replication factors. Please create a new InfeedQueue.",
          feed_id, infeed_iterator->ReplicationFactor(), replication_factor);
    }
  }

  return Status::OK();
}

void PoplarExecutor::ConnectOutfeedToStreamCallback(
    const TranslatedOutfeedInfos& outfeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
    return;
  }

  std::unique_lock<std::mutex> l(outfeeds_mutex_);
  for (const auto& outfeed_info : outfeed_infos) {
    const auto& outfeed_id = outfeed_info.stream_prefix;
    auto itr = outfeed_contexts_.find(outfeed_id);
    if (itr == outfeed_contexts_.end()) {
      LOG(FATAL) << "Outfeed with id='" << outfeed_id
                 << "' is not registered, but is required by the engine.";
    }

    auto* outfeed_context = itr->second.get();
    auto tensor_count = outfeed_context->shapes.size();
    for (unsigned j = 0; j < tensor_count; ++j) {
      size_t length = ShapeUtil::ByteSizeOf(outfeed_context->shapes[j]);
      const auto bytes_per_replica = length / current_replication_factor_;
      for (auto replica_id = 0; replica_id < current_replication_factor_;
           ++replica_id) {
        auto& queue =
            outfeed_context->callback_to_io_thread_queues[j][replica_id];
        current_engine_->connectStreamToCallback(
            GetOutfeedCopyHandle(outfeed_info.canonical_info.config.feed_id(),
                                 j),
            replica_id, [&queue, bytes_per_replica](void* src) {
              // The outfeed callback gets the buffer at the back of the
              // queue, writes to it, and then moves the write position of the
              // queue.
              void* dest = queue->BlockBack();
              std::memcpy(dest, src, bytes_per_replica);
              queue->FinishedBack();
            });
      }
    }
  }
}

IOFunction PoplarExecutor::CreateInfeedIOThreadFunction(
    const TranslatedFeedInfo& infeed_info) {
  TENSORFLOW_TRACEPOINT();
  std::unique_lock<std::mutex> l(infeeds_mutex_);
  // Find the iterator.
  auto itr = infeed_iterators_.find(infeed_info.stream_prefix);
  if (itr == infeed_iterators_.end()) {
    LOG(FATAL)
        << "Trying to access an infeed context which has not been created."
        << " Did you initialize the infeed_queue '" << infeed_info.stream_prefix
        << "'?";
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
      if (VLOG_IS_ON(2)) {
        if (infeed_queues[0][0]->IsFull()) {
          VLOG(2) << "Infeed queue is full.";
        }

        if (infeed_queues[0][0]->IsEmpty()) {
          VLOG(2) << "Infeed queue is empty.";
        }
      }

      if (infeed_queues[0][0]->IsFull()) {
        _mm_pause();
        continue;
      }

      // Enqueue tensors to each replica.
      for (size_t replica_id = 0; replica_id != current_replication_factor_;
           ++replica_id) {
        std::vector<tensorflow::Tensor> outputs;
        bool end_of_sequence = false;
        Status s = infeed_dataset_iterator->GetNext(&outputs, &end_of_sequence);

        // Handle the upstream iterator failing to produce an element
        if (!s.ok()) {
          // LOG(FATAL) aborts the thread and never returns.
          LOG(FATAL) << "An infeed dataset iterator has failed with status: "
                     << s.ToString();
        }

        // Handle the upstream iterator running out of elements
        if (end_of_sequence) {
          VLOG(1) << "The dataset iterator has reached the end of the dataset.";
          infeed_dataset_iterator->SignalAllQueuesToEnd();
          // This is not considered an error. However, we will report an
          // error if the consumer tries to pop past the end of the queue.
          return Status::OK();
        }

        for (size_t j = 0; j != outputs.size(); ++j) {
          auto& queue = infeed_queues[replica_id][j];
          auto* tb = tensorflow::DMAHelper::buffer(&outputs[j]);
          // Increase the refcount for the buffer whilst it is in the infeed
          // queue.
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
inline void AllocateTensors(
    std::deque<std::vector<tensorflow::Tensor>>& queue,
    const std::vector<tensorflow::DataType>& types,
    const std::vector<tensorflow::TensorShape>& shapes) {
  queue.emplace_front(types.size());
  auto& tensors = queue.front();
  for (size_t i = 0; i != types.size(); ++i) {
    tensors[i] = tensorflow::Tensor(types[i], shapes[i]);
  }
}

inline void AllocateTensorsWithDefaults(
    std::deque<std::vector<tensorflow::Tensor>>& queue,
    const std::vector<Shape>& xla_shapes,
    const std::vector<tensorflow::DataType>& types,
    const std::vector<tensorflow::TensorShape>& shapes) {
  std::vector<Literal> default_values;
  for (auto& shape : xla_shapes) {
    default_values.push_back(Literal::CreateFromShape(shape));
  }

  queue.emplace_front(types.size());
  auto& tensors = queue.front();
  for (size_t i = 0; i != types.size(); ++i) {
    tensors[i] = tensorflow::Tensor(types[i], shapes[i]);
    auto* tb = tensorflow::DMAHelper::buffer(&tensors[i]);
    std::memcpy(tb->data(), default_values[i].untyped_data(),
                tensors[i].TotalBytes());
  }
}
}  // namespace

IOFunction PoplarExecutor::CreateOutfeedIOThreadFunction(
    const TranslatedFeedInfo& outfeed_info) {
  TENSORFLOW_TRACEPOINT();
  std::unique_lock<std::mutex> l(outfeeds_mutex_);
  auto itr = outfeed_contexts_.find(outfeed_info.stream_prefix);
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
        _mm_pause();
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
                          outfeed_context->tf_shapes);
        }

        // We need to copy along 2 axis. There are multiple queues from
        // the IPU, one  per tuple and per replica. There is a
        // single queue out of the executor, consisting of a vector of
        // Tensors, one per tuple entry. If there are multiple replicas
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

            std::vector<tensorflow::Tensor>& tensors_to_write_to =
                outfeed_context->io_thread_output_queues.front();

            auto& tensor = tensors_to_write_to[tuple_idx];

            // When there are mutiple replicas, insert the data into a slice
            // out of dinension 0.  Otherwise just use the whole tensor.
            auto output_tensor =
                (replicas == 1 ? tensor : tensor.SubSlice(replica_id));
            auto* tb = tensorflow::DMAHelper::buffer(&output_tensor);

            std::memcpy(tb->data(), src, output_tensor.AllocatedBytes());
            src += output_tensor.AllocatedBytes();
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

void PoplarExecutor::LaunchInfeedThreads(
    const TranslatedInfeedInfos& infeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Start all the infeeds.
  for (const TranslatedFeedInfo& info : infeed_infos) {
    IOFunction fn = CreateInfeedIOThreadFunction(info);
    io_threads_.emplace_back(
        absl::make_unique<IOThread>(info.stream_prefix, std::move(fn)));
  }
}

void PoplarExecutor::LaunchOutfeedThreads(
    const TranslatedOutfeedInfos& outfeed_infos) {
  TENSORFLOW_TRACEPOINT();
  // Start all the outfeeds.
  for (const TranslatedFeedInfo& info : outfeed_infos) {
    IOFunction fn = CreateOutfeedIOThreadFunction(info);
    io_threads_.emplace_back(
        absl::make_unique<IOThread>(info.stream_prefix, std::move(fn)));
  }
}

void PoplarExecutor::StopIOThreads() {
  TENSORFLOW_TRACEPOINT();
  // Blocks the thread until all the threads have stopped and joined back.
  io_threads_.clear();
}

void PoplarExecutor::DeferredDeallocation() {
  TENSORFLOW_TRACEPOINT();
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());

  const auto new_end = std::partition(
      allocations_.begin(), allocations_.end(),
      [](TensorControl* tc) { return tc->ref_count > 0 || tc->on_device; });

  std::for_each(new_end, allocations_.end(), [](TensorControl* tc) {
    VLOG(2) << "Deallocated " << tc;
    delete tc;
  });

  allocations_.erase(new_end, allocations_.end());
}

bool PoplarExecutor::Memcpy(se::Stream* stream, void* host_dst,
                            const se::DeviceMemoryBase& pop_src, uint64 size) {
  AsPoplarStream(stream)->EnqueueTask([this, host_dst, pop_src, size]() {
    Status ok = SynchronousMemcpy(host_dst, pop_src, size);
  });
  AsPoplarStream(stream)->BlockUntilDone();
  return true;
}

bool PoplarExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const void* host_src, uint64 size) {
  se::DeviceMemoryBase dst = *pop_dst;
  AsPoplarStream(stream)->EnqueueTask([this, dst, host_src, size]() mutable {
    Status ok = SynchronousMemcpy(&dst, host_src, size);
  });
  AsPoplarStream(stream)->BlockUntilDone();
  return true;
}

Status PoplarExecutor::SynchronousMemcpy(se::DeviceMemoryBase* pop_dst,
                                         const void* host_src, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  std::memcpy(tc->data, host_src, size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    tc->on_device = false;
    tc->input_handle.reset();
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
    if (tc->on_device == true && tc->output_handle) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost());
    }
  }
  std::memcpy(host_dst, tc->data, size);
  return Status::OK();
}

Status PoplarExecutor::SynchronousMemcpyDeviceToDevice(
    se::DeviceMemoryBase* dst, const se::DeviceMemoryBase& src, uint64 size) {
  TensorControl* dst_tc = reinterpret_cast<TensorControl*>(dst->opaque());
  const TensorControl* src_tc =
      reinterpret_cast<const TensorControl*>(src.opaque());
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    if (src_tc->on_device == true && src_tc->output_handle) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost());
    }
  }
  std::memcpy(dst_tc->data, src_tc->data, size);
  {
    std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
    dst_tc->on_device = false;
    dst_tc->input_handle.reset();
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
  TENSORFLOW_TRACEPOINT();
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::HostCallback(se::Stream* stream,
                                  std::function<Status()> callback) {
  TENSORFLOW_TRACEPOINT();
  AsPoplarStream(stream)->EnqueueTask([callback]() {
    Status status = callback();
    if (!status.ok()) {
      LOG(WARNING) << "Host callback failed: " << status;
    }
  });

  return true;
}

bool PoplarExecutor::CreateStreamDependency(se::Stream* dependent,
                                            se::Stream* other) {
  TENSORFLOW_TRACEPOINT();
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
  return GetAndResetExecutorStatus();
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
  std::lock_guard<std::recursive_mutex> g(mutex_);
  return device_.has_value();
}

bool PoplarExecutor::IPUConfig::DeviceAttached() const {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  return device_attached_;
}

bool PoplarExecutor::IPUConfig::TargetConfigured() const {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  return target_.has_value();
}

void PoplarExecutor::IPUConfig::ClearDevice() {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  device_attached_ = false;
  device_.reset();
}

void PoplarExecutor::IPUConfig::Clear() {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  ClearDevice();
  target_.reset();
}

std::recursive_mutex& PoplarExecutor::IPUConfig::Mutex() { return mutex_; }

const poplar::Target& PoplarExecutor::IPUConfig::Target() {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  if (!target_ && PoplarXlaFlags::Get().use_ipu_model) {
    // If the device has not been configured via configure_ipu_system, but we
    // have requested an IPU model, then we create a CPU device.
    device_ = poplar::Device::createCPUDevice();
    target_ = device_->getTarget();
  }
  return TargetOrDie();
}

const poplar::Target& PoplarExecutor::IPUConfig::TargetOrDie() const {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  CHECK(target_);
  return *target_;
}

const poplar::Device& PoplarExecutor::IPUConfig::Device() const {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  CHECK(device_);
  return *device_;
}

void PoplarExecutor::IPUConfig::SetDevice(poplar::Device&& device) {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  device_ = std::move(device);
}

void PoplarExecutor::IPUConfig::SetDeviceAttached() {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  device_attached_ = true;
}

void PoplarExecutor::IPUConfig::SetDeviceAndTarget(poplar::Device&& device) {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  device_ = std::move(device);
  target_ = device_->getTarget();
}

void PoplarExecutor::IPUConfig::SetTarget(const poplar::Target& target) {
  std::lock_guard<std::recursive_mutex> g(mutex_);
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

int64 PoplarExecutor::GetNumIpusInLocalProcess(
    const poplar::Target& target) const {
  const int64 num_target_ipus = target.getNumIPUs();
  if (HasMultiReplicaDistributionOptions()) {
    // With multi-replica distribution, we run only on our subset of the IPUs.
    const int64 process_count = GetMultiReplicaProcessCount();
    CHECK_GT(process_count, 0);
    CHECK_EQ(num_target_ipus % process_count, 0);
    return num_target_ipus / process_count;
  } else {
    return num_target_ipus;
  }
}

const IpuOptions& PoplarExecutor::GetIpuOptions() const {
  return current_config_;
}

const bool PoplarExecutor::IpuOptionsConfigured() const { return configured_; }

bool PoplarExecutor::PoplarDeviceIsAttached() const {
  return ipu_.DeviceAttached();
}

StatusOr<std::size_t> PoplarExecutor::AttachToPoplarDevice(
    absl::Span<const poplar::Device> device_list, int32 ordinal,
    bool wait_for_device) {
  TENSORFLOW_TRACEPOINT();
  if (device_list.empty()) {
    return InvalidArgumentStrCat(
        "No device matches the requested configuration for ordinal ", ordinal,
        ".");
  }

  const uint64 on_demand_device_poll_time =
      std::max<uint64>(PoplarXlaFlags::Get().on_demand_device_poll_time, 100);
  const uint64 on_demand_device_poll_time_us =
      on_demand_device_poll_time * 1000ULL;
  const uint64 on_demand_device_timeout =
      PoplarXlaFlags::Get().on_demand_device_timeout;

  tensorflow::Env* env = tensorflow::Env::Default();
  auto start_time = std::chrono::steady_clock::now();

  bool logged_message = false;
  while (true) {
    std::size_t attached;
    for (attached = 0; attached < device_list.size(); ++attached) {
      // Try to attach to that device.
      if (device_list[attached].attach()) {
        break;
      }
    }

    if (attached < device_list.size()) {
      return attached;
    }

    if (wait_for_device) {
      auto now_time = std::chrono::steady_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
          now_time - start_time);

      if (elapsed_time.count() > on_demand_device_timeout) {
        return InternalErrorStrCat(
            "Timed out trying to find an available device for ordinal ",
            ordinal, ".");
      } else {
        if (!logged_message) {
          LOG(INFO) << "Currently there is no available device for ordinal ",
              ordinal, ". Waiting for one to become available.";
          logged_message = true;
        }
        env->SleepForMicroseconds(on_demand_device_poll_time_us);
      }
    } else {
      return InternalErrorStrCat(
          "Could not find an available device for ordinal ", ordinal, ".");
    }
  }
}

Status PoplarExecutor::AttachToPoplarDevice() {
  TENSORFLOW_TRACEPOINT();
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  if (ipu_.DeviceAttached()) {
    return Status::OK();
  }

  const bool wait_for_device =
      ConnectionType() == IpuDeviceConnectionType::ON_DEMAND;
  const bool use_ipu_model = PoplarXlaFlags::Get().use_ipu_model;

  try {
    if (!ipu_.TargetConfigured()) {
      if (!use_ipu_model) {
        return InvalidArgument("Device not configured and IPU model disabled.");
      }
      GetOrCreatePoplarTarget();
    }
    if (ipu_.DeviceConfigured()) {
      // Device was selected when the target was created: attach or fail.
      if (use_ipu_model) {
        if (!ipu_.Device().attach()) {
          return InternalErrorStrCat(
              "Unable to acquire Poplar device IPUModel for ordinal ", ordinal_,
              ".");
        }
      } else {
        auto& device = ipu_.Device();
        TF_ASSIGN_OR_RETURN(
            std::size_t device_index,
            AttachToPoplarDevice(absl::Span<const poplar::Device>(&device, 1),
                                 ordinal_, wait_for_device));
      }
    } else {
      // Poplar device would already be set if we were using the model.
      CHECK(HasIpuHardware());
      const poplar::Target& target = GetOrCreatePoplarTarget();
      const int64 num_local_ipus = GetNumIpusInLocalProcess(target);

      // Hardware devices
      auto device_list =
          GetDeviceManager().getDevices(target.getTargetType(), num_local_ipus);
      TF_ASSIGN_OR_RETURN(
          std::size_t attached,
          AttachToPoplarDevice(device_list, ordinal_, wait_for_device));
      ipu_.SetDevice(std::move(device_list.at(attached)));
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
    ipu_.SetDeviceAttached();
  } catch (poplar::poplar_error e) {
    return xla::InternalError("Unable to open Poplar device for ordinal %d: %s",
                              ordinal_, e.what());
  }

  return Status::OK();
}

void PoplarExecutor::DetachFromPoplarDevice() {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  if (PoplarDeviceIsAttached()) {
    VLOG(1) << "Detaching from " << GetDeviceTargetName() << " ordinal "
            << ordinal_;

    ipu_.Device().detach();
    ipu_.ClearDevice();
  }
}

Status PoplarExecutor::CreatePoplarTarget() {
  TENSORFLOW_TRACEPOINT();
  bool has_user_config = (current_config_.device_config_size() > 0);

  if (!PoplarXlaFlags::Get().use_ipu_model) {
    if (ConnectionType() != IpuDeviceConnectionType::NEVER &&
        ConnectionType() != IpuDeviceConnectionType::PRE_COMPILE &&
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
      auto& device = device_list.at(*device_index);
      if (HasMultiReplicaDistributionOptions()) {
        ipu_.SetTarget(CreateMultiReplicaDistributionTarget(
            device.getTarget(), GetMultiReplicaProcessCount()));
        ipu_.SetDevice(std::move(device));
      } else {
        ipu_.SetDeviceAndTarget(std::move(device));
      }
    } else {
      CHECK(num_devices);

      // If there is an IPU version configured then use that.
      if (!current_config_.ipu_version().empty()) {
        int64 num_target_devices = *num_devices;
        if (HasMultiReplicaDistributionOptions()) {
          const int64 process_count = GetMultiReplicaProcessCount();
          CHECK_GT(process_count, 0);
          num_target_devices *= process_count;
        }
        ipu_.SetTarget(poplar::Target::createIPUTarget(
            num_target_devices, current_config_.ipu_version()));
      } else {
        // Deduce the IPU target given the configuration.
        switch (ConnectionType()) {
          case IpuDeviceConnectionType::ALWAYS:
          case IpuDeviceConnectionType::ON_DEMAND: {
            CHECK(HasIpuHardware());
            // Get target from the available devices.
            auto device_list = GetDeviceManager().getDevices(
                poplar::TargetType::IPU, *num_devices);
            if (device_list.empty()) {
              return FailedPrecondition(
                  "Unsupported number of IPUs requested - could not find an"
                  " IPU device with %d IPUs for the TensorFlow virtual device"
                  " /device:IPU:%d. Use `gc-info -l` to view the available"
                  " device configurations.",
                  *num_devices, ordinal_);
            }
            if (HasMultiReplicaDistributionOptions()) {
              ipu_.SetTarget(CreateMultiReplicaDistributionTarget(
                  device_list.front().getTarget(),
                  GetMultiReplicaProcessCount()));
            } else {
              ipu_.SetTarget(device_list.front().getTarget());
            }
            break;
          }
          case IpuDeviceConnectionType::NEVER: {
            return FailedPrecondition(
                "Expected the `ipu_version` to be set when the "
                "`device_connection_type` is set to "
                "`IpuDeviceConnectionType.NEVER`");
          }
          case IpuDeviceConnectionType::PRE_COMPILE: {
            return FailedPrecondition(
                "Expected the `ipu_version` to be set when the "
                "`device_connection_type` is set to "
                "`IpuDeviceConnectionType.PRE_COMPILE`");
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

    if (HasMultiReplicaDistributionOptions()) {
      return Unimplemented(
          "Multi-replica distribution is not supported with the IPU model");
    }

    std::string user_model_version =
        current_config_.ipu_model_config().ipu_model_version();
    if (user_model_version.empty()) {
      LOG(WARNING) << "A version was not supplied when using the IPU Model."
                   << " Defaulting to 'ipu2'.";
      user_model_version = "ipu2";
    }

    poplar::IPUModel model(user_model_version.c_str());
    model.numIPUs = num_ipus;

    model.compileIPUCode =
        current_config_.ipu_model_config().compile_ipu_code();
    if (current_config_.ipu_model_config().tiles_per_ipu() > 0) {
      model.tilesPerIPU = current_config_.ipu_model_config().tiles_per_ipu();
    }
    ipu_.SetDeviceAndTarget(model.createDevice());
  }
  return Status::OK();
}

Status PoplarExecutor::ConfigurePoplarDevice(const IpuOptions& cfg) {
  TENSORFLOW_TRACEPOINT();
  bool has_user_config = (current_config_.device_config_size() > 0);
  if (!DeviceConfigurationsEqual(cfg, current_config_) && has_user_config) {
    XLA_VLOG_LINES(1, "Current config: " + current_config_.DebugString() +
                          "\nNew config: " + cfg.DebugString());
    return FailedPrecondition(
        "IPU system configuration has already been set in this process, "
        "but it should have been reset automatically by the call to "
        "'tensorflow.python.ipu.config.configure_ipu_system'.");
  }
  if (ipu_.DeviceAttached()) {
    if (DeviceConfigurationsEqual(current_config_, IpuOptions())) {
      // If there is no config associated to the open device then it is a CPU
      // device: dettach from it and initialize a Poplar device instead.
      DetachFromPoplarDevice();
    } else {
      VLOG(1) << "Poplar device: type " << GetDeviceTargetName() << " ordinal "
              << ordinal_ << " is already configured: staying attached to it.";
    }
  }
  current_config_ = cfg;
  configured_ = true;

  if (!ipu_.DeviceAttached()) {
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
      option_flags_.set("debug.retainDebugInformation", "true");
      option_flags_.set("debug.computeInstrumentationLevel", "device");
      break;
    case IpuExecutionProfileType::IPU_PROFILE:
      option_flags_.set("debug.instrument", "true");
      option_flags_.set("debug.retainDebugInformation", "true");
      option_flags_.set("debug.computeInstrumentationLevel", "ipu");
      break;
    case IpuExecutionProfileType::TILE_PROFILE:
      option_flags_.set("debug.instrument", "true");
      option_flags_.set("debug.retainDebugInformation", "true");
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

  for (const auto& opt : current_config_.slice_options()) {
    slice_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.profiling().graph_options()) {
    graph_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.profiling().execution_options()) {
    execution_options_.set(opt.option(), opt.value());
  }

  for (const auto& opt : current_config_.gcl_options()) {
    gcl_options_.set(opt.option(), opt.value());
  }

  const auto max_compilation_threads =
      PoplarXlaFlags::Get().max_compilation_threads;
  if (max_compilation_threads > 0) {
    option_flags_.set("opt.maxCompilationThreads",
                      std::to_string(max_compilation_threads));
  }

  if (CompilerReportingEnabled()) {
    option_flags_.set("debug.retainDebugInformation", "true");
    option_flags_.set("debug.allowOutOfMemory", "true");
  }

  if (!PoplarXlaFlags::Get().save_vertex_graph.empty() ||
      !PoplarXlaFlags::Get().save_interval_report.empty()) {
    option_flags_.set("debug.retainDebugInformation", "true");
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

  for (auto opt : gcl_options_) {
    VLOG(1) << "GCL option: " << opt.first << " = " << opt.second;
  }

  // Generate Target hash
  std::vector<int64> target_hash;
  target_hash.push_back(ipu_.Target().getNumTiles());
  target_hash.push_back(ipu_.Target().getDataPathWidth());
  target_hash.push_back(ipu_.Target().getBytesPerTile());
  target_hash.push_back(ipu_.Target().getNumWorkerContexts());
  target_hash.push_back(ipu_.Target().getTilesPerIPU());
  target_hash.push_back(ipu_.Target().getNumIPUs());
  target_hash.push_back(static_cast<int64>(ipu_.Target().getTargetType()));
  target_hash.push_back(
      static_cast<int64>(ipu_.Target().getIpuLinkConfiguration()));
  target_hash.push_back(ipu_.Target().getIpuLinkDomainSize());
  target_hash.push_back(static_cast<int64>(ipu_.Target().getIpuLinkTopology()));
  target_hash.push_back(ipu_.Target().getGatewayMode());
  target_hash.push_back(ipu_.Target().getGatewayMultiReadServiceTable());

  if (ipu_.Target().getTargetType() == poplar::TargetType::IPU) {
    target_hash.push_back(
        std::hash<string>()(ipu_.Target().getTargetArchString()));
  }

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

  // Whether multi-replica distribution is enabled or not. This impacts
  // the syncReplicasIndependently Poplar engine option.
  target_hash.push_back(HasMultiReplicaDistributionOptions());

  // Get hashes for GCL compilation parameters.
  absl::c_copy(GetGclHashes(), std::back_inserter(target_hash));

  poplar_device_hash_ = CombinedHash(target_hash);

  return Status::OK();
}

bool PoplarExecutor::HaveExecutableCache() const {
  return !GetExecutableCachePath().empty();
}

Status PoplarExecutor::CreateExecutableCacheDirIfMissing() const {
  return CreateDirIfMissing(GetExecutableCachePath());
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

ModuleFilenames::ModuleFilenames(uint64 hash,
                                 const std::string& serialization_folder)
    : basename_(tensorflow::strings::Printf("%0llx", hash)),
      serialization_folder_(serialization_folder) {}

std::string ModuleFilenames::CachedExecutableFilename() const {
  return tensorflow::io::JoinPath(GetExecutableCachePath(),
                                  basename_ + ".poplar_exec");
}

std::string ModuleFilenames::CompilationLockFilename() const {
  return tensorflow::io::JoinPath(GetExecutableCachePath(),
                                  basename_ + ".compile_lock");
}

ModuleFilenames PoplarExecutor::GetModuleFilenames(uint64 hash) const {
  return ModuleFilenames(hash, SerializationFolder());
}

bool PoplarExecutor::HaveCachedExecutable(
    const ModuleFilenames& filenames) const {
  return tensorflow::Env::Default()
      ->FileExists(filenames.CachedExecutableFilename())
      .ok();
}

bool PoplarExecutor::SupportsRemoteBuffers() const {
  CHECK(HasPoplarTarget());
  if (!PoplarDeviceIsAttached()) {
    return current_config_.enable_remote_buffers_without_device();
  }

  return ipu_.Device().supportsRemoteBuffers();
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
    const std::string& module_name, const std::string& tensor_map,
    const std::string& instruction_info, int64 duration) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::COMPILE_END);

  auto* compile_end = evt.mutable_compile_end();
  compile_end->set_module_name(std::move(module_name));
  compile_end->set_duration(duration);
  compile_end->set_tensor_map(std::move(tensor_map));
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

void PoplarExecutor::AddExecuteEventRecord(const std::string& module_name) {
  auto evt = NewTraceEvent();
  evt.set_type(tensorflow::IpuTraceEvent::EXECUTE);
  evt.mutable_execute()->set_module_name(std::move(module_name));

  reports_.push_back(evt);
}

Status PoplarExecutor::GetCompilerEvents(
    std::list<tensorflow::IpuTraceEvent>& out) {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  out.splice(out.end(), std::move(reports_));
  reports_.clear();
  return Status::OK();
}

std::string PoplarExecutor::GetModuleReportDirectory(const std::string& name) {
  // Generate a subdirectory for 'name's reports.
  // Use a map to remember what directories were generated for what clusters.

  // Guard against concurrent gets/sets when this Executor is being used to
  // compile multiple clusters simultaneously
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());

  auto it = cluster_report_directories_.find(name);
  if (it == cluster_report_directories_.end()) {
    std::string cluster_directory_name = GenerateDirectoryName("tf_report");
    if (HasMultiReplicaDistributionOptions()) {
      cluster_directory_name.append(
          absl::StrCat("__instance_", GetMultiReplicaProcessIndex()));
    }

    cluster_report_directories_[name] = cluster_directory_name;
    VLOG(1) << "Saving reports for " << name << " to "
            << cluster_report_directories_[name];
  }
  return cluster_report_directories_[name];
}

void PoplarExecutor::FlattenedDeviceMemoryList(
    InputPairList& list, const xla::Shape& shape, void* base,
    const InputOutputAliasingMap::InputInfo& input_info,
    const RemoteParameterInfo* remote_parameter_info) {
  TensorControl* tc = static_cast<TensorControl*>(base);
  if (shape.IsTuple()) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t = 0; t < xla::ShapeUtil::TupleElementCount(shape);
         t++) {
      void* ptr = ptrs[t];
      FlattenedDeviceMemoryList(list,
                                xla::ShapeUtil::GetTupleElementShape(shape, t),
                                ptr, input_info, remote_parameter_info);
    }
  } else {
    list.push_back(InputDef(tc, GetInputConversionFunction(shape),
                            input_info.IsStreaming(), remote_parameter_info));
  }
}

/*static*/ PoplarExecutor::ArgsHandleMap PoplarExecutor::CreateArgsHandleMap(
    const Args& args, se::DeviceMemoryAllocator* allocator,
    const PoplarExecutable& executable, int ordinal) {
  PoplarExecutor::ArgsHandleMap args_map;

  const auto* comp = executable.module().entry_computation();
  std::vector<xla::Shape> shapes(comp->num_parameters());
  for (const auto& inst : comp->parameter_instructions()) {
    shapes[inst->parameter_number()] = inst->shape();
  }

  const auto& inputs_info =
      executable.GetInputOutputAliasingMap().GetEntryInputInfos();
  CHECK_EQ(inputs_info.size(), args.size());
  CHECK_EQ(shapes.size(), args.size());

  // We require all the resource arguments which are modified to be
  // not-aliasing with each other.
  absl::flat_hash_set<const TensorControl*> modified_resources;

  for (unsigned int a = 0; a < inputs_info.size(); a++) {
    const auto& input_info = inputs_info[a];
    InputPairList bufs;
    auto remote_parameter_info =
        FindRemoteParameterInfo(a, executable.GetRemoteParameterInfos());
    const Shape& shape = shapes[a];
    FlattenedDeviceMemoryList(bufs, shape, const_cast<void*>(args[a].opaque()),
                              input_info, remote_parameter_info);
    for (unsigned i = 0; i < bufs.size(); i++) {
      InputDef& input = bufs[i];
      auto input_handle = input_info.Handles().at(i);
      input.tc->element_type = shape.element_type();

      if (remote_parameter_info && remote_parameter_info->host_rearrangement) {
        auto& param_host_rearrangement =
            *remote_parameter_info->host_rearrangement;
        gcl::CollectiveBalancedHostRearrangement host_rearrangement;
        host_rearrangement.replicationFactor =
            param_host_rearrangement.replication_factor;
        host_rearrangement.totalElementsPerReplica =
            param_host_rearrangement.total_elements_per_replica;
        host_rearrangement.gatheredToRefSlices.reserve(
            param_host_rearrangement.gathered_to_ref_slice.size());
        for (auto& slice : param_host_rearrangement.gathered_to_ref_slice) {
          host_rearrangement.gatheredToRefSlices.emplace_back(slice.first,
                                                              slice.second);
        }
        host_rearrangement.elementMap = param_host_rearrangement.element_map;
        input.tc->host_rearrangement = std::move(host_rearrangement);
      }

      if (input_info.IsResource() && !input_info.IsResourceNotModified()) {
        if (modified_resources.contains(input.tc)) {
          // We found an alias - we add a copy.
          VLOG(1) << "Found an alias for input handle " << input_handle
                  << ", duplicating the buffer.";
          se::DeviceMemoryBase allocated =
              allocator->Allocate(ordinal, input.tc->size, false)
                  .ConsumeValueOrDie()
                  .Release();
          TensorControl* tc =
              reinterpret_cast<TensorControl*>(allocated.opaque());
          std::memcpy(tc->data, input.tc->data, input.tc->size);
          input.tc = tc;
        }
        modified_resources.insert(input.tc);
      }

      input.tc->element_type = shape.element_type();
      args_map.emplace(ArgHandle{a, i, input_handle}, input);
    }
  }

  return args_map;
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

void PoplarExecutor::UpdateOutputsHandleMap(const PoplarExecutable& executable,
                                            const xla::Shape& shape,
                                            se::DeviceMemoryBase retbuf) {
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
      outputs_map_[*bufs[i].tc->output_handle] = bufs[i];
    }
  }
}
/*static*/ std::unique_ptr<PoplarExecutor::OutputAllocation>
PoplarExecutor::GetOutputAllocator(const PoplarExecutable& executable,
                                   const ArgsHandleMap& args_map,
                                   se::DeviceMemoryAllocator* allocator,
                                   int ordinal,
                                   IpuDeviceConnectionType connection_type) {
  if (executable.Engine()) {
    if (connection_type == IpuDeviceConnectionType::PRE_COMPILE) {
      return absl::make_unique<PoplarExecutor::PrecompileOutputAllocation>(
          allocator, executable.GetInputOutputAliasingMap(), args_map, ordinal);
    } else {
      return absl::make_unique<PoplarExecutor::BufferOutputAllocation>(
          allocator, executable.GetInputOutputAliasingMap(), args_map, ordinal);
    }
  }

  if (executable.IsConstantGraph() || executable.IsScalarElementwiseGraph()) {
    return absl::make_unique<PoplarExecutor::ConstantOutputAllocation>(
        allocator, executable.GetInputOutputAliasingMap(), args_map, ordinal);
  }

  if (executable.IsRemapGraph()) {
    return absl::make_unique<PoplarExecutor::RemapOutputAllocation>(
        allocator, executable.GetInputOutputAliasingMap(), args_map, ordinal,
        executable.RemapMap());
  }
  LOG(FATAL) << "Cannot get an output allocator.";

  return std::unique_ptr<PoplarExecutor::OutputAllocation>{};
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutor::ConstantOutputAllocation::AllocateBuffer(const Shape& shape,
                                                         int64, int64) const {
  const int64 size = ShapeUtil::ByteSizeOf(shape);
  TF_ASSIGN_OR_RETURN(auto allocated_owned,
                      allocator_->Allocate(ordinal_, size, false));
  se::DeviceMemoryBase allocated = allocated_owned.Release();
  return allocated;
}

Status PoplarExecutor::ConstantOutputAllocation::PopulateBuffer(
    se::DeviceMemoryBase& buffer, const Shape& shape, int64 output_index,
    int64 flat_tensor_index) const {
  CHECK(constants_);
  const auto& constant = constants_->at(output_index).at(flat_tensor_index);
  const int64 size = ShapeUtil::ByteSizeOf(shape);
  TensorControl* tc = reinterpret_cast<TensorControl*>(buffer.opaque());
  tc->size = size;
  tc->element_type = shape.element_type();
  tc->on_device = false;
  tc->output_handle = absl::nullopt;
  tc->output_convertor = nullptr;

  void* buf = static_cast<void*>(tc->data);
  std::memcpy(buf, constant.untyped_data(), constant.size_bytes());
  return Status::OK();
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutor::PrecompileOutputAllocation::AllocateBuffer(const Shape& shape,
                                                           int64, int64) const {
  const int64 size = ShapeUtil::ByteSizeOf(shape);
  TF_ASSIGN_OR_RETURN(auto allocated_owned,
                      allocator_->Allocate(ordinal_, size, false));
  se::DeviceMemoryBase allocated = allocated_owned.Release();
  return allocated;
}

Status PoplarExecutor::PrecompileOutputAllocation::PopulateBuffer(
    se::DeviceMemoryBase& buffer, const Shape& shape, int64, int64) const {
  const int64 size = ShapeUtil::ByteSizeOf(shape);
  TensorControl* tc = reinterpret_cast<TensorControl*>(buffer.opaque());
  tc->size = size;
  tc->element_type = shape.element_type();
  tc->on_device = false;
  tc->output_handle = absl::nullopt;
  tc->output_convertor = nullptr;

  void* buf = static_cast<void*>(tc->data);
  // Create a literal with the right datatype.
  Literal default_values = Literal::CreateFromShape(shape);
  std::memcpy(buf, default_values.untyped_data(), default_values.size_bytes());
  return Status::OK();
}

bool PoplarExecutor::RemapOutputAllocation::AddRemapCopy(
    int64 output_index) const {
  bool make_a_copy = false;

  const auto& input_infos = io_map_.GetEntryInputInfos();
  const auto& output_infos = io_map_.GetEntryOutputInfos();
  if (input_infos.size() > 0 && output_infos.size() > 0) {
    const uint64 input_index = output_infos.at(output_index).GetInputIndex();
    const bool is_input_resource = input_infos.at(input_index).IsResource();
    const bool is_output_resource = output_infos.at(output_index).IsResource();
    make_a_copy = is_input_resource != is_output_resource;
  }

  return make_a_copy;
}

StatusOr<PoplarExecutor::TensorControl*>
PoplarExecutor::RemapOutputAllocation::GetRemapedTensorControl(
    int64 output_index, int64 flat_tensor_index) const {
  const int64 remap_idx = remap_map_.at(output_index);

  auto it = args_map_.find(ArgHandle{remap_idx, flat_tensor_index});
  if (it == args_map_.end()) {
    return FailedPrecondition("Could not remap an output to input tensor.");
  }
  return it->second.tc;
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutor::RemapOutputAllocation::AllocateBuffer(
    const Shape&, int64 output_index, int64 flat_tensor_index) const {
  TF_ASSIGN_OR_RETURN(TensorControl * original,
                      GetRemapedTensorControl(output_index, flat_tensor_index));

  // Add a reference to prevent this from being deallocated.
  original->ref_count++;
  if (AddRemapCopy(output_index)) {
    TF_ASSIGN_OR_RETURN(auto allocated_owned,
                        allocator_->Allocate(ordinal_, original->size, false));
    se::DeviceMemoryBase allocated = allocated_owned.Release();
    return allocated;
  } else {
    // Return a reference.
    return se::DeviceMemoryBase(original, original->size);
  }
}

Status PoplarExecutor::RemapOutputAllocation::PopulateBuffer(
    se::DeviceMemoryBase& buffer, const Shape&, int64 output_index,
    int64 flat_tensor_index) const {
  TF_ASSIGN_OR_RETURN(TensorControl * original,
                      GetRemapedTensorControl(output_index, flat_tensor_index));

  if (AddRemapCopy(output_index)) {
    TensorControl* tc = reinterpret_cast<TensorControl*>(buffer.opaque());
    CHECK(!original->on_device);
    std::memcpy(tc->data, original->data, original->size);
    // Remove the extra reference as the buffer has been cloned.
    original->ref_count--;
  }
  return Status::OK();
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutor::BufferOutputAllocation::AllocateBuffer(
    const Shape& shape, int64 output_index, int64 flat_tensor_index) const {
  const auto& output_info = io_map_.GetEntryOutputInfos().at(output_index);
  if (output_info.IsResourceModified()) {
    // The output is an in-place update of one of the inputs.
    auto it = args_map_.find(
        ArgHandle{output_info.GetInputIndex(), flat_tensor_index});
    if (it == args_map_.end()) {
      return FailedPrecondition(
          "Could not find matching input resource tensor.");
    }
    TensorControl* tc = it->second.tc;
    tc->ref_count++;
    return se::DeviceMemoryBase(tc, tc->size);
  } else {
    // The output is not one of the inputs.
    const int64 size = ShapeUtil::ByteSizeOf(shape);
    TF_ASSIGN_OR_RETURN(auto allocated_owned,
                        allocator_->Allocate(ordinal_, size, false));
    se::DeviceMemoryBase allocated = allocated_owned.Release();
    return allocated;
  }
}

Status PoplarExecutor::BufferOutputAllocation::PopulateBuffer(
    se::DeviceMemoryBase& buffer, const Shape& shape, int64 output_index,
    int64 flat_tensor_index) const {
  const auto& output_info = io_map_.GetEntryOutputInfos().at(output_index);
  TensorControl* tc = reinterpret_cast<TensorControl*>(buffer.opaque());
  tc->size = ShapeUtil::ByteSizeOf(shape);
  tc->element_type = shape.element_type();
  tc->on_device = output_info.IsStreaming() ? false : true;
  tc->output_handle = ArgHandle{output_index, flat_tensor_index,
                                output_info.Handles().at(flat_tensor_index)};
  tc->output_convertor = GetOutputConversionFunction(shape);
  return Status::OK();
}

/*static*/ StatusOr<se::DeviceMemoryBase> PoplarExecutor::AllocateOutputBuffer(
    const PoplarExecutable& executable, se::DeviceMemoryAllocator* allocator,
    const ArgsHandleMap& args_map, int ordinal,
    IpuDeviceConnectionType connection_type) {
  TENSORFLOW_TRACEPOINT();
  const Shape& shape = executable.result_shape();
  VLOG(2) << "Allocating output buffer " << shape << " for "
          << executable.module().name();

  auto output_allocator = GetOutputAllocator(executable, args_map, allocator,
                                             ordinal, connection_type);

  if (!shape.IsTuple()) {
    return output_allocator->AllocateBuffer(shape, /*output_index=*/0,
                                            /*current_flat_tuple_index=*/0);
  }

  const int64 tuple_size = ShapeUtil::TupleElementCount(shape);
  const int64 top_size = xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  TF_ASSIGN_OR_RETURN(auto top_buffer_owned,
                      allocator->Allocate(ordinal, top_size, false));
  se::DeviceMemoryBase top_buffer = top_buffer_owned.Release();
  TensorControl* top_tc = reinterpret_cast<TensorControl*>(top_buffer.opaque());
  void** top_buf = reinterpret_cast<void**>(top_tc->data);

  for (int64 output_index = 0; output_index != tuple_size; ++output_index) {
    // Walk the shape for the current output index.
    const Shape& output_index_shape = shape.tuple_shapes(output_index);
    ShapeTree<se::DeviceMemoryBase> buffer_tree(output_index_shape);

    // Note that this walk is in pre-order (DFT).
    int64 flat_tuple_index = 0;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        output_index_shape,
        [&](const Shape& subshape, const ShapeIndex& index) -> Status {
          VLOG(2) << "Allocating output buffer " << subshape << " at "
                  << output_index << " index " << index;
          // Create a buffer for this node.
          se::DeviceMemoryBase* node = buffer_tree.mutable_element(index);
          if (subshape.IsTuple()) {
            // If the shape is a tuple, create a buffer to point at all the
            // subshapes.
            const int64 size =
                xla::ShapeUtil::ByteSizeOf(subshape, sizeof(void*));
            TF_ASSIGN_OR_RETURN(auto owned_allocated,
                                allocator->Allocate(ordinal, size, false));
            se::DeviceMemoryBase allocated = owned_allocated.Release();
            TensorControl* tc =
                reinterpret_cast<TensorControl*>(allocated.opaque());
            *node = se::DeviceMemoryBase(tc, size);

          } else {
            const int64 current_flat_tuple_index = flat_tuple_index++;
            TF_ASSIGN_OR_RETURN(
                *node, output_allocator->AllocateBuffer(
                           subshape, output_index, current_flat_tuple_index));
          }

          // Propagate the allocated buffer into the buffer above.
          if (!index.empty()) {
            ShapeIndex parent_index = index;
            const int64 parent_output_index = parent_index.back();
            parent_index.pop_back();

            se::DeviceMemoryBase* parent_buffer =
                buffer_tree.mutable_element(parent_index);
            TensorControl* tc =
                reinterpret_cast<TensorControl*>(parent_buffer->opaque());
            void** parent_buf = reinterpret_cast<void**>(tc->data);
            parent_buf[parent_output_index] = node->opaque();
          }
          return Status::OK();
        }));
    top_buf[output_index] = buffer_tree.mutable_element({})->opaque();
  }
  return top_buffer;
}

Status PoplarExecutor::PopulateOutputBuffer(
    se::DeviceMemoryBase& buffer, const PoplarExecutable& executable,
    se::DeviceMemoryAllocator* allocator,
    const PoplarExecutor::OutputAllocation& output_allocator,
    const Shape& shape) {
  VLOG(2) << "Populating output buffer " << shape << " for "
          << executable.module().name();
  if (!shape.IsTuple()) {
    return output_allocator.PopulateBuffer(buffer, shape, /*output_index=*/0,
                                           /*current_flat_tuple_index=*/0);
  }

  for (int64 output_index = 0;
       output_index != ShapeUtil::TupleElementCount(shape); ++output_index) {
    // Walk the shape for the current output index.
    const Shape& output_index_shape = shape.tuple_shapes(output_index);
    int64 flat_tuple_index = 0;
    // Only need to populate the leaf nodes.
    for (const auto& indexed_shape :
         ShapeUtil::GetLeafShapes(output_index_shape)) {
      const Shape& leaf_shape = indexed_shape.shape;
      const ShapeIndex& index = indexed_shape.index;

      VLOG(2) << "Populating output buffer " << leaf_shape << " at "
              << output_index << " index " << index;

      ShapeIndex full_index = index;
      full_index.push_front(output_index);

      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase leaf_buffer,
          PoplarExecutor::GetBufferByShapeIndex(buffer, full_index));

      TF_RETURN_IF_ERROR(output_allocator.PopulateBuffer(
          leaf_buffer, leaf_shape, output_index, flat_tuple_index++));
    }
  }

  return Status::OK();
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
    std::memcpy(buf, converted.data(), converted.size());
  }
}

StatusOr<bool> PoplarExecutor::CheckMoveDeviceToHostRequired(
    const bool engine_changed) {
  // Pull previous execution outputs back from device if:
  // a) one is on the device _and_
  // b)   the engine is changing _or_
  // c)   output buffer isn't an input to the current engine _or_
  // d)   output buffer isn't currently in the right place for the new input
  for (const auto& tc : allocations_) {
    if (tc->on_device == true && tc->output_handle) {
      if (engine_changed || args_map_.count(*tc->input_handle) == 0 ||
          tc != args_map_.at(*tc->input_handle).tc) {
        return true;
      }
    }
  }
  return false;
}

// Check if there is tensor/arg of current executable on device.
StatusOr<bool> PoplarExecutor::CheckAnyArgOnDevice(const Args& args) {
  for (auto& device_buffer : args) {
    const TensorControl* tc =
        reinterpret_cast<const TensorControl*>(device_buffer.opaque());

    if (tc->on_device && tc->output_handle) {
      return true;
    }
  }
  return false;
}

StatusOr<bool> PoplarExecutor::CheckRemapGraphNeedsOnDeviceBuffers(
    const OutputAllocation& output_allocator, const Shape& shape) {
  const RemapOutputAllocation* remap_allocator =
      static_cast<const RemapOutputAllocation*>(&output_allocator);

  auto remap_requires_device_buffer =
      [remap_allocator](int64 output_index,
                        int64 flat_tuple_index) -> StatusOr<bool> {
    if (remap_allocator->AddRemapCopy(output_index)) {
      TF_ASSIGN_OR_RETURN(auto tc, remap_allocator->GetRemapedTensorControl(
                                       output_index, flat_tuple_index));
      return tc->on_device;
    }
    return false;
  };

  if (!shape.IsTuple()) {
    return remap_requires_device_buffer(/*output_index=*/0,
                                        /*current_flat_tuple_index=*/0);
  }

  for (int64 output_index = 0;
       output_index != ShapeUtil::TupleElementCount(shape); ++output_index) {
    // Walk the shape for the current output index.
    const Shape& output_index_shape = shape.tuple_shapes(output_index);

    for (int64 flat_tuple_index = 0;
         flat_tuple_index != ShapeUtil::GetLeafCount(output_index_shape);
         ++flat_tuple_index) {
      TF_ASSIGN_OR_RETURN(
          bool requires_device_buffer,
          remap_requires_device_buffer(output_index, flat_tuple_index));
      if (requires_device_buffer) {
        return true;
      }
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
          *arg.second.tc->input_handle != arg.first) {
        do_host_to_device = true;
      }
    }
  }
  return do_host_to_device;
}

void PoplarExecutor::ConnectReplicatedDeviceToHost(
    const std::string& stream_name, TensorControl* tc) {
  TENSORFLOW_TRACEPOINT();
  void* dest = static_cast<void*>(tc->data);
  const std::size_t size = HostSizeToDeviceSize(tc->size, tc->element_type);

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
  TENSORFLOW_TRACEPOINT();
  if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
    // Make sure all the allocations are marked as on host.
    for (auto* tc : allocations_) {
      tc->on_device = false;
    }

    return Status::OK();
  }

  Json::Value root;
  root["tensors"] = Json::Value(Json::arrayValue);
  uint64 total_size = 0;
  uint64 total_count = 0;
  try {
    for (const auto& tc : allocations_) {
      // Set up streams
      if (tc->on_device == true && tc->output_handle) {
        auto buffer_size = tc->size;
        if (tc->in_memory_remote_parameter_info) {
          // We currently only get one copy of the buffer.
          // Note that only resource variables are on device, hence they must
          // have the input handle set too.
          CHECK(tc->input_handle);

          const std::string buffer_name =
              tc->in_memory_remote_parameter_info->buffer_name;
          const int64 buffer_offset =
              tc->in_memory_remote_parameter_info->buffer_offset;
          buffer_size = tc->GetRemoteBufferSize();

          if (tc->in_memory_remote_parameter_info->is_replica_partitioned) {
            if (HostSizeToDeviceSize(buffer_size, tc->element_type) !=
                buffer_size) {
              return InvalidArgumentStrCat(
                  "Unsupported replica partitioned type ",
                  PrimitiveType_Name(tc->element_type), " for ", buffer_name);
            }

            const std::size_t bytes_per_replica =
                PartitionedByteCountPerReplica(buffer_size, tc->element_type,
                                               current_replication_factor_);

            std::vector<char> buffer;
            const bool rearrange = tc->host_rearrangement.has_value();
            if (rearrange) {
              buffer.resize(buffer_size);
            } else {
              buffer.resize(bytes_per_replica);
            }

            // This is a remote parameter - copy it to the remote buffer for
            // each replica.
            for (int replica_id = 0; replica_id < current_replication_factor_;
                 ++replica_id) {
              const std::size_t offset = replica_id * bytes_per_replica;
              CHECK_LE(offset, buffer_size);
              const std::size_t replica_length =
                  std::min(bytes_per_replica, buffer_size - offset);

              if (rearrange) {
                // Collect all shards into full buffer to rearrange collective
                // balanced reorder later.
                current_engine_->copyFromRemoteBuffer(
                    buffer_name, buffer.data() + offset, buffer_offset,
                    replica_id);
              } else {
                current_engine_->copyFromRemoteBuffer(
                    buffer_name, buffer.data(), buffer_offset, replica_id);

                std::memcpy(tc->data + offset, buffer.data(), replica_length);
              }
            }
            if (rearrange) {
              auto bytes_per_element =
                  ShapeUtil::ByteSizeOfPrimitiveType(tc->element_type);
              VLOG(3) << "Undo rearrangement for " << tc->output_handle->name
                      << ", size: " << tc->size << "/" << buffer_size;
              tc->host_rearrangement->undoRearrangeForCollective(
                  buffer.data(), tc->data, bytes_per_element);
            }
          } else {
            const unsigned replica_id = 0;
            current_engine_->copyFromRemoteBuffer(buffer_name, tc->data,
                                                  buffer_offset, replica_id);
          }
        } else {
          ConnectReplicatedDeviceToHost(tc->output_handle->name, tc);
        }

        Json::Value tensor;
        tensor["name"] = Json::Value(tc->output_handle->name);
        tensor["parameter_index"] =
            Json::Value::Int64(tc->output_handle->parameter_index);
        tensor["flat_tensor_index"] =
            Json::Value::Int64(tc->output_handle->flat_tensor_index);
        tensor["size"] = Json::Value::UInt64(buffer_size);
        root["tensors"].append(tensor);
        total_size += buffer_size;
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
    for (auto* tc : allocations_) {
      if (tc->on_device == true && tc->output_handle) {
        PostProcessBuffer(tc);
      }

      TF_RETURN_IF_ERROR(ResetTensorControlState(tc));
    }
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Device to host]", e);
  }
  return Status::OK();
}

Status PoplarExecutor::ResetTensorControlState(TensorControl* tc) {
  tc->in_memory_remote_parameter_info = absl::nullopt;
  tc->on_device = false;
  tc->output_handle.reset();
  tc->input_handle.reset();
  return Status::OK();
}

Status PoplarExecutor::ResetOnDeviceBuffers() {
  for (auto* tc : allocations_) {
    TF_RETURN_IF_ERROR(ResetTensorControlState(tc));
  }
  return Status::OK();
}

Status PoplarExecutor::MoveHostToDevice() {
  TENSORFLOW_TRACEPOINT();

  if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
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
      auto buffer_size = tc->size;
      if (!arg.second.streamed) {
        buf = PreProcessBuffer(arg.second);

        if (arg.second.remote_parameter_info) {
          tc->in_memory_remote_parameter_info.emplace(
              *arg.second.remote_parameter_info);

          buffer_size = tc->GetRemoteBufferSize();
          const std::string buffer_name =
              tc->in_memory_remote_parameter_info->buffer_name;
          const int64 buffer_offset =
              tc->in_memory_remote_parameter_info->buffer_offset;

          if (tc->in_memory_remote_parameter_info->is_replica_partitioned) {
            if (HostSizeToDeviceSize(buffer_size, tc->element_type) !=
                buffer_size) {
              return InvalidArgumentStrCat(
                  "Unsupported replica partitioned type ",
                  PrimitiveType_Name(tc->element_type), " for ", buffer_name);
            }

            const std::size_t bytes_per_replica =
                PartitionedByteCountPerReplica(buffer_size, tc->element_type,
                                               current_replication_factor_);

            std::vector<char> buffer;
            const bool rearrange = tc->host_rearrangement.has_value();
            if (rearrange) {
              CHECK_LE(tc->size, buffer_size);
              buffer.resize(buffer_size);

              VLOG(3) << "Rearranging data for collective " << buffer_name
                      << ", size: " << tc->size << "/" << buffer_size;
              auto bytes_per_element =
                  ShapeUtil::ByteSizeOfPrimitiveType(tc->element_type);
              if (tc->size < buffer_size) {
                std::vector<char> temp(buffer_size);
                memcpy(temp.data(), tc->data, tc->size);
                memset(temp.data() + tc->size, 0, buffer_size - tc->size);
                tc->host_rearrangement->rearrangeForCollective(
                    temp.data(), buffer.data(), bytes_per_element);
              } else {
                tc->host_rearrangement->rearrangeForCollective(
                    tc->data, buffer.data(), bytes_per_element);
              }
            } else {
              buffer.resize(bytes_per_replica);
            }

            // This is a remote parameter - copy it to the remote buffer for
            // each replica.
            for (int replica_id = 0; replica_id < current_replication_factor_;
                 ++replica_id) {
              const std::size_t offset = replica_id * bytes_per_replica;
              CHECK_LE(offset, buffer_size);

              if (rearrange) {
                current_engine_->copyToRemoteBuffer(buffer.data() + offset,
                                                    buffer_name, buffer_offset,
                                                    replica_id);
              } else {
                const std::size_t replica_length =
                    std::min(bytes_per_replica, buffer_size - offset);
                // Copy the replica-local region into the tmp buffer (with
                // padding).
                std::memcpy(buffer.data(), static_cast<char*>(buf) + offset,
                            replica_length);

                // Zero the padding
                std::memset(buffer.data() + replica_length, 0,
                            bytes_per_replica - replica_length);

                // Copy the padded buffer to the remote buffer.
                current_engine_->copyToRemoteBuffer(buffer.data(), buffer_name,
                                                    buffer_offset, replica_id);
              }
            }
          } else {
            CHECK(!tc->host_rearrangement.has_value());
            for (int replica_id = 0; replica_id < current_replication_factor_;
                 ++replica_id) {
              current_engine_->copyToRemoteBuffer(buf, buffer_name,
                                                  buffer_offset, replica_id);
            }
          }
        } else {
          tc->in_memory_remote_parameter_info = absl::nullopt;
          current_engine_->connectStream(arg.first.name, buf);
        }

        tc->on_device = true;
        tc->input_handle = arg.first;

        Json::Value tensor;
        tensor["name"] = Json::Value(arg.first.name);
        tensor["parameter_index"] =
            Json::Value::Int64(arg.first.parameter_index);
        tensor["flat_tensor_index"] =
            Json::Value::Int64(arg.first.flat_tensor_index);
        tensor["name"] = Json::Value(arg.first.name);
        tensor["size"] = Json::Value::UInt64(buffer_size);
        root["tensors"].append(tensor);
        total_size += buffer_size;

        stream_list.push_back(std::make_pair(arg.first.name, 0));
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
    return PoplarExceptionToTensorflowStatus("[Host to device]", e);
  }

  return Status::OK();
}

/*static*/ StatusOr<se::DeviceMemoryBase> PoplarExecutor::GetBufferByShapeIndex(
    const se::DeviceMemoryBase& top, const ShapeIndex& index) {
  se::DeviceMemoryBase buffer = top;
  for (int64 i : index) {
    const TensorControl* tc =
        reinterpret_cast<const TensorControl*>(buffer.opaque());
    void** bufs = reinterpret_cast<void**>(tc->data);
    const int64 size = reinterpret_cast<const TensorControl*>(bufs[i])->size;
    buffer = se::DeviceMemoryBase(bufs[i], size);
  }
  return buffer;
}

void PoplarExecutor::ConnectStreamedVariablesHostToDevice() {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
    return;
  }

  for (auto arg : args_map_) {
    if (arg.second.streamed) {
      void* buf = PreProcessBuffer(arg.second);
      current_engine_->connectStream(arg.first.name, buf);
    }
  }
}

void PoplarExecutor::ConnectStreamedVariablesDeviceToHost() {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Parameters)) {
    return;
  }

  for (auto output : outputs_map_) {
    if (output.second.streamed) {
      TensorControl* tc = output.second.tc;
      ConnectReplicatedDeviceToHost(output.first.name, tc);
    }
  }
}

void PoplarExecutor::PostProcessStreamedVariablesDeviceToHost() {
  TENSORFLOW_TRACEPOINT();
  for (auto output : outputs_map_) {
    if (output.second.streamed) {
      PostProcessBuffer(output.second.tc);
    }
  }
}

void PoplarExecutor::Reset() {
  std::lock_guard<std::recursive_mutex> lock(ipu_.Mutex());

  AboutToFreeEngine(current_engine_);
  StopIOThreads();
  DetachFromPoplarDevice();
  GetAndResetExecutorStatus();

  // Note that we don't reset the IO feeds as that would require the
  // infeeds to be reinitialised. Similarly for host embeddings.
  ResetOptionFlags();
  ResetConfiguration();
  ResetReports();
  ResetHandles();
}

void PoplarExecutor::ResetOptionFlags() {
  option_flags_.clear();
  conv_options_.clear();
  matmul_options_.clear();
  pooling_options_.clear();
  graph_options_.clear();
  execution_options_.clear();
  gcl_options_.clear();
}

void PoplarExecutor::ResetConfiguration() {
  ipu_.Clear();
  current_config_.Clear();

  configured_ = false;
  poplar_device_hash_ = 0;
  current_replication_factor_ = 1;
}

void PoplarExecutor::ResetReports() {
  reports_.clear();
  cluster_report_directories_.clear();
}

void PoplarExecutor::ResetHandles() {
  args_map_.clear();
  outputs_map_.clear();
}

void PoplarExecutor::AboutToFreeEngine(poplar::Engine* engine) {
  TENSORFLOW_TRACEPOINT();
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  if (current_engine_) {
    if (engine == current_engine_) {
      auto status = MoveDeviceToHost();
      if (!status.ok()) {
        LOG(FATAL) << status.ToString();
      }
      DeferredDeallocation();
      current_engine_ = nullptr;
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
  TENSORFLOW_TRACEPOINT();
  auto& feed_id = config.feed_id();
  std::unique_lock<std::mutex> l(infeeds_mutex_);
  if (infeed_iterators_.contains(feed_id)) {
    LOG(FATAL) << "Infeed with id='" << feed_id
               << "' already exists. Consider changing the `feed_name` in "
                  "IPUInfeedQueue. The Poplar backend requires all infeeds in "
                  "the same TensorFlow device to have unique names.";
  } else {
    infeed_iterators_[feed_id] = absl::make_unique<InfeedIterator>(
        flr, params, dataset, GetInfeedAllocator(), shapes, feed_id);
  }
}

Status PoplarExecutor::DeleteInfeedIterator(const std::string& feed_id) {
  TENSORFLOW_TRACEPOINT();
  std::lock_guard<std::recursive_mutex> l(ipu_.Mutex());
  std::unique_lock<std::mutex> il(infeeds_mutex_);
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
  OutfeedContext* outfeed_context = nullptr;
  std::unique_lock<std::mutex> outfeed_lock(outfeeds_mutex_);
  auto itr = outfeed_contexts_.find(feed_id);
  if (itr == outfeed_contexts_.end()) {
    LOG(INFO)
        << "Trying to dequeue elements from the outfeed queue with id="
        << feed_id
        << " which has not executed yet. Make sure to execute the "
           "program with the outfeed before trying to dequeue an outfeed.";
    return {};
  }
  outfeed_context = itr->second.get();
  // Lock whilst we dequeue all the tensors.
  std::lock_guard<std::recursive_mutex> guard(outfeed_context->mutex);
  outfeed_lock.unlock();

  if (ConnectionType() == IpuDeviceConnectionType::PRE_COMPILE) {
    // For pre-compilation mode ensure that each dequeue call has a tensor of
    // zeros to allow applications to continue.
    AllocateTensorsWithDefaults(
        outfeed_context->io_thread_output_queues, outfeed_context->shapes,
        outfeed_context->tf_data_types, outfeed_context->tf_shapes);
  }

  if (mode == xla::poplarplugin::PoplarFeedConfig::GetAll) {
    std::vector<std::vector<tensorflow::Tensor>> output(
        outfeed_context->io_thread_output_queues.size());
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = outfeed_context->io_thread_output_queues.back();
      outfeed_context->io_thread_output_queues.pop_back();
    }
    return output;
  } else {
    if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
      LOG(WARNING)
          << "Trying to dequeue elements from the outfeed queue with id="
          << feed_id
          << " which has outfeed_mode `GetLast`. This is not supported when "
             "using synthetic data.";
      return {};
    }
    std::vector<std::vector<tensorflow::Tensor>> output(1);
    output[0] = outfeed_context->io_thread_output_queues.front();
    outfeed_context->io_thread_output_queues.clear();
    return output;
  }
}

int64 PoplarExecutor::GetReplicationFactorForOutfeed(
    const std::string& feed_id) const {
  OutfeedContext* outfeed_context = nullptr;
  std::lock_guard<std::mutex> outfeed_lock(outfeeds_mutex_);

  auto iter = outfeed_contexts_.find(feed_id);
  if (iter == outfeed_contexts_.end()) {
    LOG(WARNING)
        << "Trying to get replication factor for the outfeed queue with id="
        << feed_id
        << " which has not executed yet. Make sure to execute the "
           "program with the outfeed before trying to dequeue an outfeed.";
    return 1;
  }

  outfeed_context = iter->second.get();
  return outfeed_context->replication_factor;
}

Status PoplarExecutor::RegisterOutfeeds(
    const TranslatedOutfeedInfos& outfeed_infos) {
  std::unique_lock<std::mutex> l(outfeeds_mutex_);
  for (auto& outfeed_info : outfeed_infos) {
    auto outfeed_id = outfeed_info.stream_prefix;
    const auto existing_feed = outfeed_contexts_.find(outfeed_id);
    if (existing_feed != outfeed_contexts_.end()) {
      if (!existing_feed->second->Matches(outfeed_info,
                                          current_replication_factor_)) {
        return xla::FailedPrecondition(
            "Outfeed with id='%s' already exists but with a different tensor "
            "shape or replication factor. Consider changing the `feed_name` "
            "in IPUOutfeedQueue. The Poplar backend requires all outfeeds in "
            "the same TensorFlow device to have unique names.",
            outfeed_id.c_str());
      }
    } else {
      if (UseSyntheticDataFor(SyntheticDataCategory::Outfeed) &&
          outfeed_info.canonical_info.config.mode() ==
              xla::poplarplugin::PoplarFeedConfig::GetLast) {
        LOG(WARNING) << "Outfeed with id=" << outfeed_id
                     << " has mode `GetLast` which is not supported when "
                        "using synthetic data. An exception will be thrown if "
                        "the dequeue TensorFlow operation is executed. "
                        "Consider changing the `outfeed_mode` in "
                        "IPUOutfeedQueue.";
      }
      outfeed_contexts_[outfeed_id] = absl::make_unique<OutfeedContext>(
          outfeed_info.canonical_info.config, outfeed_info.canonical_info.shape,
          current_replication_factor_);
    }
  }
  return Status::OK();
}

bool PoplarExecutor::HasOutfeed(const std::string& feed_id) const {
  std::unique_lock<std::mutex> outfeeds_lock(outfeeds_mutex_);

  return outfeed_contexts_.contains(feed_id);
}

Status PoplarExecutor::DeleteOutfeed(const std::string& feed_id) {
  std::lock_guard<std::recursive_mutex> l(ipu_.Mutex());

  if (io_threads_.size()) {
    return xla::FailedPrecondition(
        "Cannot delete outfeed with id='%s' while in use", feed_id.c_str());
  }

  std::unique_lock<std::mutex> ol(outfeeds_mutex_);
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

    if (host_embeddings_.contains(embedding_id)) {
      return xla::FailedPrecondition(
          "Cannot register host embedding with id='%s' it already exists!",
          embedding_id.c_str());
    }

    host_embeddings_[embedding_id] = std::move(embedding);
  }

  return Status::OK();
}

Status PoplarExecutor::DeregisterHostEmbedding(
    const std::string& embedding_id) {
  {
    std::unique_lock<std::mutex> lk(host_embeddings_mutex_);

    host_embeddings_.erase(embedding_id);
  }

  return Status::OK();
}

tensorflow::Rendezvous* PoplarExecutor::GetRendezvous() {
  return rendezvous_.get();
}

void PoplarExecutor::ConnectSeedCallback() {
  TENSORFLOW_TRACEPOINT();
  // Don't connect any streams if using synthetic data
  if (UseSyntheticDataFor(SyntheticDataCategory::Seed)) {
    return;
  }

  auto& generator = seed_generator_;
  for (int replica_id = 0; replica_id < current_replication_factor_;
       ++replica_id) {
    auto callback = [&generator, replica_id](void* ptr) mutable {
      reinterpret_cast<uint64_t*>(ptr)[0] = generator->Get(replica_id);
    };

    current_engine_->connectStreamToCallback(GetRandomNumberSeedStream(),
                                             replica_id, callback);
  }
}

void PoplarExecutor::ResetSeed(int seed, bool identical_replicas) {
  if (identical_replicas) {
    seed_generator_ = absl::make_unique<IdenticalReplicaSeedGenerator>(seed);
  } else {
    seed_generator_ = absl::make_unique<DistinctReplicaSeedGenerator>(seed);
  }
}

std::string PoplarExecutor::GetCycleCounterStream() {
  return "__cycle_count_stream";
}

void PoplarExecutor::ConnectCycleCounterCallback() {
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

StatusOr<std::vector<std::vector<Literal>>>
PoplarExecutor::LiteralEvaluateForScalarElementwiseGraph(
    PoplarExecutable& executable, const Args& args) {
  std::vector<Literal> arg_literals;
  const auto* comp = executable.module().entry_computation();

  for (const auto& inst : comp->parameter_instructions()) {
    Literal literal(inst->shape(), true);
    const TensorControl* src_tc = reinterpret_cast<const TensorControl*>(
        args[inst->parameter_number()].opaque());
    std::memcpy(literal.untyped_data(), src_tc->data, src_tc->size);
    arg_literals.push_back(std::move(literal));
  }

  HloEvaluator hlo_evaluator(1);
  TF_ASSIGN_OR_RETURN(Literal literal_evaluate,
                      hlo_evaluator.Evaluate(*comp, arg_literals));

  const auto& output_shape = executable.result_shape();
  std::vector<std::vector<Literal>> constant_outputs;
  TF_RETURN_IF_ERROR(TransformEvaluatorOutput(output_shape, literal_evaluate,
                                              constant_outputs));

  return constant_outputs;
}

void PoplarExecutor::ExecuteEngine(se::DeviceMemoryBase* result_buffer,
                                   se::StreamExecutor* executor,
                                   PoplarExecutable& executable,
                                   const ArgsHandleMap& args_map,
                                   se::DeviceMemoryAllocator* allocator,
                                   const Args& args) {
  TENSORFLOW_TRACEPOINT();
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  if (!current_status_.ok()) {
    LOG(FATAL) << current_status_.ToString();
  }
  current_status_ = ExecuteEngineImpl(result_buffer, executor, executable,
                                      args_map, allocator, args);
  if (!current_status_.ok()) {
    StopIOThreads();
    TF_CHECK_OK(ResetOnDeviceBuffers());
    current_engine_ = nullptr;
  }
}

Status PoplarExecutor::ExecuteEngineImpl(se::DeviceMemoryBase* result_buffer,
                                         se::StreamExecutor* executor,
                                         PoplarExecutable& executable,
                                         const ArgsHandleMap& args_map,
                                         se::DeviceMemoryAllocator* allocator,
                                         const Args& args) {
  std::lock_guard<std::recursive_mutex> g(ipu_.Mutex());
  args_map_ = args_map;

  const auto& input_output_aliasing_map =
      executable.GetInputOutputAliasingMap();
  const Shape& output_shape = executable.result_shape();
  poplar::Engine* engine = executable.Engine();

  const bool engine_changed = current_engine_ != engine;

  auto output_allocator = GetOutputAllocator(executable, args_map_, allocator,
                                             ordinal_, ConnectionType());

  if (!executable.GetHostEmbeddingLookupInfos().empty()) {
    std::unique_lock<std::mutex> lk(host_embeddings_mutex_);

    // Go through and double check we have all the required host embeddings.
    for (auto& host_embedding_lookup_info :
         executable.GetHostEmbeddingLookupInfos()) {
      if (!host_embeddings_.contains(host_embedding_lookup_info.embedding_id)) {
        return xla::FailedPrecondition(
            "Host embedding interface with id='%s' not registered. Did you run "
            "the associated host_embedding op in the session?",
            host_embedding_lookup_info.embedding_id);
      }
    }
  }

  if (!engine) {
    // An empty engine means that we should not execute the module on the
    // device.
    if (executable.IsScalarElementwiseGraph()) {
      // The graph is small - execute it on the CPU.
      // If some arguments are on device, move them to host.
      TF_ASSIGN_OR_RETURN(bool any_arg_on_device, CheckAnyArgOnDevice(args));
      if (any_arg_on_device) {
        TF_RETURN_IF_ERROR(MoveDeviceToHost());
      }

      TF_ASSIGN_OR_RETURN(
          auto constant_outputs,
          LiteralEvaluateForScalarElementwiseGraph(executable, args));
      executable.SetLiteralValue(std::move(constant_outputs));
    }

    if (executable.IsConstantGraph() || executable.IsScalarElementwiseGraph()) {
      // Set the constants to populate the output buffers with.
      static_cast<ConstantOutputAllocation*>(output_allocator.get())
          ->SetConstants(&executable.LiteralValue());
    } else if (executable.IsRemapGraph()) {
      // If any of the buffers which are remaped and required a copy are
      // currently on the device, then the copy needs to be done on the host.
      TF_ASSIGN_OR_RETURN(
          bool needs_on_device_buffer,
          CheckRemapGraphNeedsOnDeviceBuffers(*output_allocator, output_shape));
      if (needs_on_device_buffer) {
        TF_RETURN_IF_ERROR(MoveDeviceToHost());
      }
    }
    // Populate all the output buffers - no execution of the engine is required.
    TF_RETURN_IF_ERROR(PopulateOutputBuffer(*result_buffer, executable,
                                            allocator, *output_allocator,
                                            output_shape));
  } else if (ConnectionType() == IpuDeviceConnectionType::PRE_COMPILE) {
    TF_RETURN_IF_ERROR(PopulateOutputBuffer(*result_buffer, executable,
                                            allocator, *output_allocator,
                                            output_shape));
    // Create outfeed queues with the correct replication factor.
    SetCurrentReplicationFactor(executable.GetReplicationFactor());
    TF_RETURN_IF_ERROR(RegisterOutfeeds(executable.GetOutfeedInfos()));
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
        SetCurrentReplicationFactor(executable.GetReplicationFactor());

        ConnectSeedCallback();
        if (executable.LoggingCycleCount()) {
          ConnectCycleCounterCallback();
        }

        if (current_config_.profiling().enable_ipu_trace_events() &&
            current_config_.profiling().enable_io_trace()) {
          AddLoadEngineEventRecord(executable.module().name());
        }

        executable.OnEngineLoaded();
      } catch (const std::exception& e) {
        return PoplarExceptionToTensorflowStatus("[Load engine]", e);
      }
    }

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
      TF_RETURN_IF_ERROR(PopulateOutputBuffer(*result_buffer, executable,
                                              allocator, *output_allocator,
                                              output_shape));

      UpdateOutputsHandleMap(executable, output_shape, *result_buffer);
    }

    VLOG(1) << "Executing on Poplar stream ordinal " << ordinal_ << " of type "
            << GetDeviceTargetName();

    // Create any outfeed queues which do not already exist
    TF_RETURN_IF_ERROR(RegisterOutfeeds(executable.GetOutfeedInfos()));

    try {
      // Connect the streams to and from the device
      ConnectStreamedVariablesHostToDevice();
      ConnectStreamedVariablesDeviceToHost();

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
        TF_RETURN_IF_ERROR(SetupInfeedReplication(infeed_infos));
        ConnectInfeedsToStreamCallback(infeed_infos);
      }

      for (auto& host_embedding_lookup_info :
           executable.GetHostEmbeddingLookupInfos()) {
        TF_RETURN_IF_ERROR(ConnectHostEmbeddingLookup(
            host_embedding_lookup_info,
            host_embeddings_.at(host_embedding_lookup_info.embedding_id)
                .get()));
      }

      for (auto& host_embedding_update_info :
           executable.GetHostEmbeddingUpdateInfos()) {
        TF_RETURN_IF_ERROR(ConnectHostEmbeddingUpdateToRendezvous(
            host_embedding_update_info,
            host_embeddings_.at(host_embedding_update_info.embedding_id)
                .get()));
      }

      for (auto& host_embedding_notify_info :
           executable.GetHostEmbeddingNotifyInfos()) {
        TF_RETURN_IF_ERROR(ConnectHostEmbeddingNotify(
            host_embedding_notify_info,
            host_embeddings_.at(host_embedding_notify_info.embedding_id)
                .get()));
      }

      const auto& outfeed_infos = executable.GetOutfeedInfos();
      if (!outfeed_infos.empty()) {
        ConnectOutfeedToStreamCallback(outfeed_infos);
      }

      // Handle user ops.
      UserOpsExecutionState user_ops_state(executable.GetStreamMetaInfos());
      const StreamInfos& stream_infos = executable.GetStreamInfos();
      for (auto& pair : stream_infos) {
        const std::list<StreamCopyInfo>& list = pair.second;

        // For all of the stream copies, both inputs and outputs.
        for (const StreamCopyInfo& info : list) {
          user_ops_state.ConnectStream(info, *current_engine_);
        }
      }

      // Launch the IO threads when we are not using synthetic data.
      if (!UseSyntheticDataFor(SyntheticDataCategory::Infeed)) {
        LaunchInfeedThreads(infeed_infos);
      }

      if (!UseSyntheticDataFor(SyntheticDataCategory::Outfeed)) {
        LaunchOutfeedThreads(outfeed_infos);
      }

      // Before executing the main program, prepare the random seeds for each
      // replica.
      seed_generator_->PrepareSeedsForReplicas(current_replication_factor_);

      // Run the main engine
      current_engine_->enableExecutionProfiling();
      current_engine_->run(PoplarProgramType::MAIN_SEQUENCE);

      StopIOThreads();

      for (auto& host_embedding_lookup_info :
           executable.GetHostEmbeddingLookupInfos()) {
        TF_RETURN_IF_ERROR(DisconnectHostEmbeddingLookup(
            host_embedding_lookup_info,
            host_embeddings_.at(host_embedding_lookup_info.embedding_id)
                .get()));
      }

      for (auto& host_embedding_update_info :
           executable.GetHostEmbeddingUpdateInfos()) {
        TF_RETURN_IF_ERROR(DisconnectHostEmbeddingUpdate(
            host_embedding_update_info,
            host_embeddings_.at(host_embedding_update_info.embedding_id)
                .get()));
      }

      for (auto& host_embedding_notify_info :
           executable.GetHostEmbeddingNotifyInfos()) {
        TF_RETURN_IF_ERROR(DisconnectHostEmbeddingNotify(
            host_embedding_notify_info,
            host_embeddings_.at(host_embedding_notify_info.embedding_id)
                .get()));
      }

      // We need to call post process to make sure all the data is in the
      // right format on the host
      PostProcessStreamedVariablesDeviceToHost();
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Execute engine]", e);
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
        AddExecuteEventRecord(executable.module().name());
      }
    } catch (const std::exception& e) {
      return PoplarExceptionToTensorflowStatus("[Execute engine]", e);
    }
  }

  // Deallocate all the marked buffers.
  DeferredDeallocation();

  return Status::OK();
}

void PoplarExecutor::SetCurrentReplicationFactor(
    int64 executable_replication_factor) {
  if (HasMultiReplicaDistributionOptions()) {
    const int64 process_index = GetMultiReplicaProcessIndex();
    const int64 process_count = GetMultiReplicaProcessCount();
    CHECK_GT(process_count, 0);
    CHECK_EQ(executable_replication_factor % process_count, 0);

    current_replication_factor_ = executable_replication_factor / process_count;

    LOG(INFO) << "Multi-replica distribution: process index " << process_index
              << ", process count " << process_count
              << ", global replication factor " << executable_replication_factor
              << ", local replication factor " << current_replication_factor_;
  } else {
    current_replication_factor_ = executable_replication_factor;
  }
}

}  // namespace poplarplugin
}  // namespace xla
