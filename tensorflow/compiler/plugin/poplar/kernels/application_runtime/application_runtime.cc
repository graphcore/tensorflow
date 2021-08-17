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

#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"
#include "include/json/json.h"

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Executable.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poplar/replication_factor.hpp>
#include <poputil/TileMapping.hpp>

#include "ipu/poplar_executable_data.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/io_thread.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_executable_binary_file.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tracepoint.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

template <typename T>
using StatusOr = xla::StatusOr<T>;

using PrimitiveType = xla::PrimitiveType;
using ShapeProto = xla::ShapeProto;
using FeedConfig = xla::poplarplugin::FeedConfig;
using Input = xla::poplarplugin::Input;
using Output = xla::poplarplugin::Output;
using Signature = xla::poplarplugin::Signature;
using PoplarExecutableProto = xla::poplarplugin::PoplarExecutableProto;

using PoplarExecutor = xla::poplarplugin::PoplarExecutor;
using PoplarExecutableBinaryFile =
    xla::poplarplugin::PoplarExecutableBinaryFile;

namespace tensorflow {
namespace {

StatusOr<poplar::TargetType> ParsePoplarTargetType(
    const std::string& target_type_string) {
  if (target_type_string == "IPU") {
    return poplar::TargetType::IPU;
  } else if (target_type_string == "IPU_MODEL") {
    return poplar::TargetType::IPU_MODEL;
  } else if (target_type_string == "CPU") {
    return poplar::TargetType::CPU;
  }
  return errors::InvalidArgument("Invalid target type ", target_type_string);
}

StatusOr<poplar::Device> GetIpuDevice(const poplar::TargetType target_type,
                                      const std::string target_arch_string,
                                      const int64 num_IPUs,
                                      const bool gateway_mode,
                                      const bool supports_remote_buffers) {
  VLOG(2) << "Getting a device.";
  poplar::DeviceManager manager = poplar::DeviceManager::createDeviceManager();
  auto devices =
      manager.getDevices(target_type, num_IPUs,
                         {{"gatewayMode", gateway_mode ? "true" : "false"}});
  TF_ASSIGN_OR_RETURN(std::size_t device_idx,
                      PoplarExecutor::AttachToPoplarDevice(devices, 0, true));

  if (supports_remote_buffers && !devices[device_idx].supportsRemoteBuffers()) {
    return errors::InvalidArgument(
        "The compiled TensorFlow executable requires remote buffer support, "
        "but it is not available on this device");
  }

  const std::string device_target_arch_string =
      devices[device_idx].getTarget().getTargetArchString();
  if (device_target_arch_string != target_arch_string) {
    return errors::InvalidArgument(
        "The target architecture for the compiled executable (",
        target_arch_string, ") does not match device's target architecture (",
        device_target_arch_string, ").");
  }

  return std::move(devices[device_idx]);
}

struct IOItem {
  std::string name;
  std::string handle;
  int64 argument;
  int64 tuple_index;

  DataType datatype;
  TensorShape shape;
};

void VerifyExecutable(PoplarExecutableProto& executable_proto) {
  CHECK_EQ(executable_proto.replication_factor(), 1)
      << "Embedded runtime does not support executables with a replication "
         "factor greater than one.";
  CHECK_EQ(executable_proto.infeeds().size(), 1)
      << "Embedded runtime only supports executables with a single infeed";
  CHECK_EQ(executable_proto.outfeeds().size(), 1)
      << "Embedded runtime only supports executables with a single outfeed";

  CHECK_EQ(executable_proto.sends().size(), 0)
      << "Embedded runtime does not support executables with sends";
  CHECK_EQ(executable_proto.recvs().size(), 0)
      << "Embedded runtime does not support executables with recvs";

  CHECK_EQ(executable_proto.lookups().size(), 0)
      << "Embedded runtime does not support executables with host embeddings";
  CHECK_EQ(executable_proto.updates().size(), 0)
      << "Embedded runtime does not support executables with host embeddings";
  CHECK_EQ(executable_proto.notifications().size(), 0)
      << "Embedded runtime does not support executables with host embeddings";

  CHECK_EQ(executable_proto.remote_parameters().size(), 0)
      << "Embedded runtime does not support executables with remote parameters";
}

class IOConfig {
 public:
  using IOGroup = absl::flat_hash_map<std::string, IOItem>;
  IOConfig() = default;

  void ParsePoplarExecutableProto(PoplarExecutableProto& executable_proto) {
    ParseSignature(executable_proto.embedded_runtime_config().signature());
  }

  const IOGroup& GetInputs() const { return inputs_; }

  const IOGroup& GetOutputs() const { return outputs_; }

  const IOGroup& GetInfeeds() const { return infeeds_; }

  const IOGroup& GetOutfeeds() const { return outfeeds_; }

 private:
  DataType PrimitiveTypeToDataType(PrimitiveType primitive_type) {
    switch (primitive_type) {
      case PrimitiveType::F16:
        return DataType::DT_HALF;
      case PrimitiveType::F32:
        return DataType::DT_FLOAT;

      case PrimitiveType::S8:
        return DataType::DT_INT8;
      case PrimitiveType::S16:
        return DataType::DT_INT16;
      case PrimitiveType::S32:
        return DataType::DT_INT32;

      case PrimitiveType::U8:
        return DataType::DT_UINT8;
      case PrimitiveType::U16:
        return DataType::DT_UINT16;
      case PrimitiveType::U32:
        return DataType::DT_UINT32;
    }
    return DataType::DT_INVALID;
  }

  std::pair<DataType, TensorShape> ParseShape(ShapeProto shape_proto) {
    PrimitiveType primitive_type = shape_proto.element_type();
    DataType datatype = PrimitiveTypeToDataType(primitive_type);
    TensorShape shape;
    if (shape_proto.dimensions_size() == 0) {
      int64 dims[] = {1};
      TensorShapeUtils::MakeShape(dims, 0, &shape);
    } else {
      std::vector<int64> dimensions;
      for (auto& dim : shape_proto.dimensions()) {
        dimensions.push_back(dim);
      }
      TensorShapeUtils::MakeShape(dimensions.data(), dimensions.size(), &shape);
    }
    return std::make_pair(datatype, shape);
  }

  void ParseInput(IOGroup& io_group, Input input) {
    const auto& handle = input.handle();
    DataType datatype;
    TensorShape shape;
    std::tie(datatype, shape) = ParseShape(input.shape());
    io_group[handle] = IOItem{input.name(),        handle,   input.argument(),
                              input.tuple_index(), datatype, shape};
  }

  void ParseOutput(IOGroup& io_group, Output output) {
    const auto& handle = output.handle();
    DataType datatype;
    TensorShape shape;
    std::tie(datatype, shape) = ParseShape(output.shape());
    io_group[handle] =
        IOItem{output.name(), handle, 0, output.tuple_index(), datatype, shape};
  }

  void ParseSignature(Signature signature) {
    for (auto& input : signature.inputs()) {
      ParseInput(inputs_, input);
    }

    for (auto output : signature.outputs()) {
      ParseOutput(outputs_, output);
    }

    for (auto& input : signature.streamed_inputs()) {
      ParseInput(infeeds_, input);
    }

    for (auto output : signature.streamed_outputs()) {
      ParseOutput(outfeeds_, output);
    }
  }

  IOGroup inputs_;
  IOGroup outputs_;
  IOGroup infeeds_;
  IOGroup outfeeds_;
};

class CommunicationManager {
 public:
  explicit CommunicationManager(PoplarExecutableProto& proto) {
    io_config_.ParsePoplarExecutableProto(proto);
  }

  void PushInputData(const std::string& name, tensorflow::Tensor tensor) {
    {
      std::unique_lock<std::mutex> lk(input_mutex_);
      input_queues_[name].push(std::move(tensor));
      input_cv_.notify_one();
    }
  }

  tensorflow::Tensor PopInputData(const std::string& name) {
    tensorflow::Tensor result;
    {
      std::unique_lock<std::mutex> lk(input_mutex_);
      input_cv_.wait(lk, [&, this]() {
        return Exiting() || !input_queues_[name].empty();
      });

      if (!Exiting()) {
        result = std::move(input_queues_[name].front());
        input_queues_[name].pop();
      }
    }

    return result;
  }

  void PushOutputCallback(const std::string& name,
                          std::function<void(void*)> callback) {
    {
      std::unique_lock<std::mutex> lk(output_mutex_);
      output_queues_[name].push(std::move(callback));
      output_cv_.notify_one();
    }
  }

  std::function<void(void*)> PopOutputCallback(const std::string& name) {
    std::function<void(void*)> result;
    {
      std::unique_lock<std::mutex> lk(output_mutex_);
      output_cv_.wait(lk, [&, this]() {
        return Exiting() || !output_queues_[name].empty();
      });

      if (!Exiting()) {
        result = std::move(output_queues_[name].front());
        output_queues_[name].pop();
      }
    }

    return result;
  }

  void InitiateExit() {
    engine_exit_notification_.Notify();
    input_cv_.notify_all();
    output_cv_.notify_all();
  }

  bool Exiting() const { return engine_exit_notification_.HasBeenNotified(); }

  IOConfig& GetIOConfig() { return io_config_; }

 private:
  absl::Notification engine_exit_notification_;

  std::mutex input_mutex_;
  std::mutex output_mutex_;

  std::condition_variable input_cv_;
  std::condition_variable output_cv_;

  absl::flat_hash_map<std::string, std::queue<tensorflow::Tensor>> input_queues_
      GUARDED_BY(input_mutex_);
  absl::flat_hash_map<std::string, std::queue<std::function<void(void*)>>>
      output_queues_ GUARDED_BY(output_mutex_);

  IOConfig io_config_;
};

class PrefetchCallback : public poplar::StreamCallback {
 public:
  PrefetchCallback(CommunicationManager* comm_mgr, const std::string& name)
      : comm_mgr_(comm_mgr), name_(name) {}

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    return poplar::StreamCallback::Result::NotAvailable;
  }

  void fetch(void* dest) noexcept override {
    tensorflow::Tensor t = comm_mgr_->PopInputData(name_);
    if (!comm_mgr_->Exiting()) {
      auto buffer = tensorflow::DMAHelper::buffer(&t);
      std::memcpy(dest, buffer->data(), buffer->size());
    }
  }

  void complete() noexcept override {}

 private:
  CommunicationManager* comm_mgr_;
  const std::string name_;
};

class EngineResource {
 public:
  EngineResource(const std::string& engine_name,
                 poplar::Executable&& executable, poplar::Device&& device,
                 PoplarExecutableProto& proto)
      : engine_name_(engine_name),
        device_(std::move(device)),
        engine_(std::move(executable)),
        communication_manager_(proto) {}

  poplar::Engine& GetEngine() { return engine_; }

  CommunicationManager& GetCommunicationManager() {
    return communication_manager_;
  }

  IOConfig& GetIOConfig() { return GetCommunicationManager().GetIOConfig(); }

  Status StartEngine(OpInputList& input_list) {
    VLOG(2) << "Starting engine execution for " << engine_name_;

    if (execute_thread_) {
      return errors::Internal("Engine thread already exists for engine ",
                              engine_name_);
    }

    // Get the status from the connection of streams.
    Status connect_streams_status;

    execute_thread_.reset(tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), engine_name_ + "_execute_thread",
        [&connect_streams_status, &input_list, this] {
          engine_.load(device_);
          connect_streams_status = ConnectStreams(input_list);
          connect_streams_notification_.Notify();
          if (!connect_streams_status.ok()) {
            return;
          }

          VLOG(2) << "Engine loop starting.";
          engine_.run(0);
          do {
            engine_.run(1);
          } while (!communication_manager_.Exiting());
        }));

    // Wait for streams to be connected.
    connect_streams_notification_.WaitForNotification();

    return connect_streams_status;
  }

  // Populates the values for buffers which are not streamed every run
  // operation.
  Status PopulateAndConnectInputBuffers(OpInputList& input_list) {
    VLOG(2) << "Populating the input buffers.";
    // TODO(T41137): The assumption here is that the inputs are passed in the
    // correct order.
    auto& io_config = GetIOConfig();

    for (auto& input_pair : io_config.GetInputs()) {
      auto input = input_pair.first;
      auto& io_item = input_pair.second;
      tensorflow::Tensor t = input_list[io_item.argument];
      if (io_item.shape != t.shape()) {
        return errors::FailedPrecondition(
            "Input tensor shape ", t.shape().DebugString(), " for input ",
            io_item.handle, " does not match shape in signature (",
            io_item.shape.DebugString(), ")");
      }
      auto tensor_buffer = tensorflow::DMAHelper::buffer(&t);
      input_buffers_[io_item.argument] =
          std::vector<unsigned char>(tensor_buffer->size());
      auto& buffer = input_buffers_[io_item.argument];
      std::memcpy(buffer.data(), tensor_buffer->data(), tensor_buffer->size());
      engine_.connectStream(input, buffer.data());
    }
    return Status::OK();
  }

  Status ConnectStreams(OpInputList& input_list) {
    VLOG(2) << "Connecting streams";

    auto& io_config = GetIOConfig();

    // Connect seed stream.
    engine_.connectStreamToCallback("__seed_stream", [](void* ptr) {
      static std::random_device rd;
      xla::poplarplugin::IdenticalReplicaSeedGenerator generator(rd());
      uint64 seed = generator.Get(0);
      reinterpret_cast<uint64*>(ptr)[0] = seed;
    });

    // Handle the inputs which are only copied at the beginning.
    TF_RETURN_IF_ERROR(PopulateAndConnectInputBuffers(input_list));

    // Connect streamed inputs.
    for (auto& infeed_pair : io_config.GetInfeeds()) {
      std::string feed = infeed_pair.first;
      engine_.connectStreamToCallback(
          feed, /*replica_id=*/0,
          absl::make_unique<PrefetchCallback>(&communication_manager_, feed));
    }

    // Connect streamed outputs.
    for (auto& outfeed_pair : io_config.GetOutfeeds()) {
      std::string feed = outfeed_pair.first;
      engine_.connectStreamToCallback(feed, [feed, this](void* ptr) {
        auto callback = communication_manager_.PopOutputCallback(feed);
        if (!communication_manager_.Exiting()) {
          callback(ptr);
        }
      });
    }

    return Status::OK();
  }

  void StopEngine() {
    VLOG(2) << "Stopping engine execution for " << engine_name_;

    if (!execute_thread_) {
      LOG(FATAL) << "Trying to stop the engine " << engine_name_
                 << " which is not running.";
    }
    communication_manager_.InitiateExit();
  }

 private:
  std::string engine_name_;
  poplar::Device device_;
  poplar::Engine engine_;
  CommunicationManager communication_manager_;
  std::map<int64, std::vector<unsigned char>> input_buffers_;

  // Make sure all Poplar/engine operations are performed in the same thread.
  // This ensures that the streams are connected and input values copied before
  // the start op finishes.
  absl::Notification connect_streams_notification_;

  // Thread which keeps running the engine until the communication manager is
  // asked to exit.
  // Note that the destructor blocks until the thread completes.
  std::unique_ptr<tensorflow::Thread> execute_thread_;
};

// A singleton class which owns the engines and associated resources for the
// execution.
class EngineManager {
 public:
  static EngineManager& Instance() {
    static EngineManager mgr;
    return mgr;
  }

  virtual ~EngineManager() {
    std::unique_lock<std::recursive_mutex> lk(engine_resource_map_mutex_);
    try {
      for (auto const& it : engine_resource_map_) {
        it.second->StopEngine();
      }
    } catch (std::exception e) {
    }
  }

  bool EngineExists(const std::string& engine_name) {
    std::unique_lock<std::recursive_mutex> lk(engine_resource_map_mutex_);
    return engine_resource_map_.contains(engine_name);
  }

  // Function which creates an engine if it already doesn't exist. If it
  // doesn't exist, then it creates it.
  Status CreateEngine(const std::string& engine_name,
                      const std::string& executable_filename,
                      OpInputList& input_list) {
    std::unique_lock<std::recursive_mutex> lk(engine_resource_map_mutex_);
    if (EngineExists(engine_name)) {
      return Status::OK();
    }

    VLOG(2) << "Creating an engine.";
    PoplarExecutableProto proto;
    TF_ASSIGN_OR_RETURN(
        poplar::Executable executable,
        PoplarExecutableBinaryFile::Read(executable_filename, &proto));

    // Check the the executable is compatible.
    VerifyExecutable(proto);

    // Check the versions.
    if (proto.tf_major_version() != TF_MAJOR_VERSION ||
        proto.tf_minor_version() != TF_MINOR_VERSION) {
      return errors::InvalidArgument(
          "TensorFlow version mismatch. Runtime version is ", TF_MAJOR_VERSION,
          ".", TF_MINOR_VERSION, ", executable version is ",
          proto.tf_major_version(), ".", proto.tf_minor_version(), ".");
    }

    if (proto.tf_git_version() != tf_git_version()) {
      return errors::InvalidArgument(
          "TensorFlow build version mismatch. Runtime version is ",
          tf_git_version(), ", executable version is ", proto.tf_git_version(),
          ".");
    }

    auto& erc = proto.embedded_runtime_config();

    TF_ASSIGN_OR_RETURN(poplar::TargetType target_type,
                        ParsePoplarTargetType(erc.target_type()));

    // Create a device.
    TF_ASSIGN_OR_RETURN(
        auto device,
        GetIpuDevice(target_type, erc.target_arch(), erc.num_ipus(),
                     erc.gateway_mode(), erc.supports_remote_buffers()));

    auto engine_resource = absl::make_unique<EngineResource>(
        engine_name, std::move(executable), std::move(device), proto);

    TF_RETURN_IF_ERROR(engine_resource->StartEngine(input_list));

    // Only insert the engine into the map if it's successfully started.
    engine_resource_map_.insert_or_assign(engine_name,
                                          std::move(engine_resource));
    return Status::OK();
  }

  StatusOr<EngineResource*> GetEngine(const std::string& engine_name) {
    std::unique_lock<std::recursive_mutex> lk(engine_resource_map_mutex_);
    if (!EngineExists(engine_name)) {
      return errors::FailedPrecondition("Engine ", engine_name,
                                        " does not exist.");
    }
    return engine_resource_map_.at(engine_name).get();
  }

 private:
  // This is a singleton class.
  EngineManager() = default;

  std::recursive_mutex engine_resource_map_mutex_;
  absl::flat_hash_map<std::string, std::unique_ptr<EngineResource>>
      engine_resource_map_ GUARDED_BY(engine_resource_map_mutex_);
};

}  // namespace

class ApplicationRuntime : public OpKernel {
 public:
  explicit ApplicationRuntime(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("engine_name", &engine_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList input_list;
    ctx->input_list("inputs", &input_list);
    auto& engine_mgr = EngineManager::Instance();

    OP_REQUIRES_OK(
        ctx, engine_mgr.CreateEngine(engine_name_, filename_, input_list));
  }

 private:
  std::string filename_;
  std::string engine_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(ApplicationRuntime);
};

REGISTER_KERNEL_BUILDER(Name("ApplicationRuntime").Device(DEVICE_CPU),
                        ApplicationRuntime);

class ApplicationCall : public AsyncOpKernel {
 public:
  explicit ApplicationCall(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("engine_name", &engine_name_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // Get the engine resource.
    auto& engine_mgr = EngineManager::Instance();
    auto status_or = engine_mgr.GetEngine(engine_name_);
    OP_REQUIRES_OK_ASYNC(ctx, status_or.status(), done);
    EngineResource* engine_resource = status_or.ValueOrDie();

    auto& comm_mgr = engine_resource->GetCommunicationManager();
    auto& io_config = engine_resource->GetIOConfig();

    OpInputList infeed_list;
    ctx->input_list("infeeds", &infeed_list);

    OpOutputList outfeed_list;
    ctx->output_list("outfeeds", &outfeed_list);

    int pushed_cb_count = 0;
    int processed_cb_count = 0;
    std::condition_variable cb_processed;
    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);

    auto check_shape = [](const IOItem& feed_params,
                          const tensorflow::Tensor& t) -> Status {
      if (feed_params.shape != t.shape()) {
        return errors::FailedPrecondition(
            "Input tensor shape ", t.shape().DebugString(), " for input ",
            feed_params.handle, " does not match shape in signature (",
            feed_params.shape.DebugString(), ")");
      }
      return Status::OK();
    };

    for (auto& infeed_pair : io_config.GetInfeeds()) {
      std::string feed = infeed_pair.first;
      const IOItem& feed_params = infeed_pair.second;

      tensorflow::Tensor t = infeed_list[feed_params.tuple_index];
      OP_REQUIRES_OK_ASYNC(ctx, check_shape(feed_params, t), done);

      comm_mgr.PushInputData(feed,
                             std::move(infeed_list[feed_params.tuple_index]));
    }

    for (auto& outfeed_pair : io_config.GetOutfeeds()) {
      std::string feed = outfeed_pair.first;
      const IOItem& feed_params = outfeed_pair.second;

      int64 i = feed_params.tuple_index;

      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(
          ctx, outfeed_list.allocate(i, feed_params.shape, &output_tensor),
          done);

      pushed_cb_count++;
      comm_mgr.PushOutputCallback(
          feed, [this, output_tensor, feed, &processed_cb_count,
                 &cb_processed](void* data) {
            auto buffer = tensorflow::DMAHelper::buffer(output_tensor);
            std::memcpy(buffer->data(), data, buffer->size());
            processed_cb_count++;
            cb_processed.notify_one();
          });
    }

    cb_processed.wait(lock,
                      [&]() { return pushed_cb_count == processed_cb_count; });
    done();
  }

 private:
  std::string engine_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(ApplicationCall);
};

REGISTER_KERNEL_BUILDER(Name("ApplicationCall").Device(DEVICE_CPU),
                        ApplicationCall);
}  // namespace tensorflow
