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

#include <pthread.h>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

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

const char APPLICATION_RUNTIME_RESOURCE_CONTAINER[] =
    "ApplicationRuntimeResourceContainer";

bool ParsePoplarTargetType(const std::string target_type_string,
                           poplar::TargetType& target_type) {
  if (target_type_string == "IPU") {
    target_type = poplar::TargetType::IPU;
    return true;
  } else if (target_type_string == "IPU_MODEL") {
    target_type = poplar::TargetType::IPU_MODEL;
    return true;
  } else if (target_type_string == "CPU") {
    target_type = poplar::TargetType::CPU;
    return true;
  }
  return false;
}

StatusOr<poplar::Device> GetIpuDevice(const poplar::TargetType target_type,
                                      const std::string target_arch_string,
                                      const int64 num_IPUs,
                                      const bool gateway_mode,
                                      const bool supports_remote_buffers) {
  poplar::DeviceManager manager = poplar::DeviceManager::createDeviceManager();
  auto devices =
      manager.getDevices(target_type, num_IPUs,
                         {{"gatewayMode", gateway_mode ? "true" : "false"}});
  TF_ASSIGN_OR_RETURN(std::size_t device_idx,
                      PoplarExecutor::AttachToPoplarDevice(devices, 0, true));

  if (supports_remote_buffers && !devices[device_idx].supportsRemoteBuffers()) {
    return errors::InvalidArgument(
        "The compiled TensorFlow executable requires remote buffer support, "
        "but it is not available on "
        "this device");
  }

  const auto& device_target_arch_string =
      devices[device_idx].getTarget().getTargetArchString();
  if (device_target_arch_string != target_arch_string) {
    return errors::InvalidArgument(absl::StrFormat(
        "The target architecture for the compiled executable (%s) does not "
        "match device's target architure (%s)",
        target_arch_string, device_target_arch_string));
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
  CHECK(executable_proto.infeeds().size() == 1)
      << "Embedded runtime only supports executables with a single infeed";
  CHECK(executable_proto.outfeeds().size() == 1)
      << "Embedded runtime only supports executables with a single outfeed";

  CHECK(executable_proto.sends().size() == 0)
      << "Embedded runtime does not support executables with sends";
  CHECK(executable_proto.recvs().size() == 0)
      << "Embedded runtime does not support executables with recvs";

  CHECK(executable_proto.lookups().size() == 0)
      << "Embedded runtime does not support executables with host embeddings";
  CHECK(executable_proto.updates().size() == 0)
      << "Embedded runtime does not support executables with host embeddings";
  CHECK(executable_proto.notifications().size() == 0)
      << "Embedded runtime does not support executables with host embeddings";

  CHECK(executable_proto.remote_parameters().size() == 0)
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

class ApplicationRuntimeComm {
 public:
  ApplicationRuntimeComm() = default;

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
};

class ApplicationRuntimeResources : public ResourceBase {
 public:
  ApplicationRuntimeResources() = default;

  std::string DebugString() const override {
    return "ApplicationRuntimeResources";
  }

  ApplicationRuntimeComm& Comm() { return comm_; }

  IOConfig& IOCfg() { return io_config_; }

 private:
  ApplicationRuntimeComm comm_;
  IOConfig io_config_;
};

}  // namespace

class ApplicationRuntime : public OpKernel {
 public:
  explicit ApplicationRuntime(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("engine_name", &engine_name_));

    if (!GetEngines().contains(engine_name_)) {
      PoplarExecutableProto proto;
      poplar::Executable executable =
          PoplarExecutableBinaryFile::Read(filename_, &proto).ValueOrDie();

      auto& ertc = proto.embedded_runtime_config();
      const std::string target_type_string = ertc.target_type();
      poplar::TargetType target_type;
      OP_REQUIRES(ctx, ParsePoplarTargetType(target_type_string, target_type),
                  errors::InvalidArgument(absl::StrFormat(
                      "Invalid target type %s", target_type_string)));

      auto status_or_device =
          GetIpuDevice(target_type, ertc.target_arch(), ertc.num_ipus(),
                       ertc.gateway_mode(), ertc.supports_remote_buffers());
      OP_REQUIRES_OK(ctx, status_or_device.status());
      auto& device = status_or_device.ValueOrDie();

      auto& io_config = resources_.IOCfg();
      io_config.ParsePoplarExecutableProto(proto);
      VerifyExecutable(proto);

      auto engine = absl::make_unique<poplar::Engine>(std::move(executable));
      engine->load(device);

      GetEngine(engine_name_) = std::move(engine);
    }
  }

  virtual ~ApplicationRuntime() {
    if (GetEngineThreads().contains(engine_name_)) {
      auto& thread = GetEngineThreads()[engine_name_];
      if (thread.joinable()) {
        auto& comm = resources_.Comm();
        comm.InitiateExit();
        thread.join();
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    std::unique_lock<std::mutex> lk(compute_mutex_);

    const std::string& e = engine_name_;

    cacheInputBuffers(context);

    auto res_mgr = context->resource_manager();
    CHECK(res_mgr);

    ApplicationRuntimeResources* resources;
    TF_CHECK_OK(res_mgr->LookupOrCreate<ApplicationRuntimeResources>(
        std::string(APPLICATION_RUNTIME_RESOURCE_CONTAINER), engine_name_,
        &resources, [this](ApplicationRuntimeResources** resources) -> Status {
          *resources = &resources_;
          return Status::OK();
        }));

    if (!GetEngineThreads().contains(engine_name_)) {
      GetEngineThreads()[engine_name_] =
          std::thread([context, e, res_mgr, this]() {
            ConnectStreams();
            ApplicationRuntime::GetEngine(e)->run(0);

            auto& comm = resources_.Comm();
            do {
              ApplicationRuntime::GetEngine(e)->run(1);
            } while (!comm.Exiting());

            TF_CHECK_OK(res_mgr->Delete<ApplicationRuntimeResources>(
                APPLICATION_RUNTIME_RESOURCE_CONTAINER, engine_name_));
          });
    }
  }

 private:
  void cacheInputBuffers(OpKernelContext* context) {
    auto& io_config = resources_.IOCfg();

    OpInputList input_list;
    context->input_list("inputs", &input_list);

    for (auto& input_pair : io_config.GetInputs()) {
      auto input = input_pair.first;
      auto& io_item = input_pair.second;
      tensorflow::Tensor t = input_list[io_item.argument];
      CHECK(io_item.shape == t.shape())
          << "Input tensor shape " << t.shape() << " for input "
          << io_item.handle << " does not match shape in signature ("
          << io_item.shape << ")\n";
      auto tensor_buffer = tensorflow::DMAHelper::buffer(&t);
      std::vector<unsigned char> buffer;
      buffer.resize(tensor_buffer->size());
      std::memcpy(buffer.data(), tensor_buffer->data(), tensor_buffer->size());
      input_buffers_[io_item.argument] = std::move(buffer);
    }
  }

  void ConnectStreams() {
    auto& engine = GetEngines()[engine_name_];

    auto& io_config = resources_.IOCfg();
    auto& comm = resources_.Comm();

    // Connect seed stream.
    engine->connectStreamToCallback("__seed_stream", [](void* ptr) {
      static std::random_device rd;
      xla::poplarplugin::IdenticalReplicaSeedGenerator generator(rd());
      uint64 seed = generator.Get(0);
      reinterpret_cast<uint64*>(ptr)[0] = seed;
    });

    // Connect inputs. This is still hardcoded until this information
    // is added to the executable.
    for (auto& input_pair : io_config.GetInputs()) {
      auto input = input_pair.first;
      auto& io_item = input_pair.second;
      engine->connectStream(input, input_buffers_[io_item.argument].data());
    }

    for (auto& infeed_pair : io_config.GetInfeeds()) {
      std::string feed = infeed_pair.first;
      auto& io_item = infeed_pair.second;
      engine->connectStreamToCallback(
          feed, [feed, io_item, &comm, this](void* ptr) {
            tensorflow::Tensor t = comm.PopInputData(feed);
            if (!comm.Exiting()) {
              auto buffer = tensorflow::DMAHelper::buffer(&t);
              std::memcpy(ptr, buffer->data(), buffer->size());
            }
          });
    }

    for (auto& outfeed_pair : io_config.GetOutfeeds()) {
      std::string feed = outfeed_pair.first;
      engine->connectStreamToCallback(feed, [feed, &comm, this](void* ptr) {
        auto callback = comm.PopOutputCallback(feed);
        if (!comm.Exiting()) {
          callback(ptr);
        }
      });
    }
  }

  static absl::flat_hash_map<std::string, std::unique_ptr<poplar::Engine>>&
  GetEngines() {
    static absl::flat_hash_map<std::string, std::unique_ptr<poplar::Engine>>
        engines_;
    return engines_;
  }

  static std::unique_ptr<poplar::Engine>& GetEngine(const std::string& name) {
    return GetEngines()[name];
  }

  static absl::flat_hash_map<std::string, std::thread>& GetEngineThreads() {
    static absl::flat_hash_map<std::string, std::thread> engine_threads_;
    return engine_threads_;
  }

  std::string filename_;
  std::string engine_name_;
  ApplicationRuntimeResources resources_;
  std::mutex compute_mutex_;
  std::map<int64, std::vector<unsigned char>> input_buffers_;

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

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    ApplicationRuntimeResources* resources;

    auto res_mgr = context->resource_manager();
    CHECK(res_mgr);
    TF_CHECK_OK(res_mgr->Lookup(APPLICATION_RUNTIME_RESOURCE_CONTAINER,
                                engine_name_, &resources));

    auto& comm = resources->Comm();

    OpInputList input_list;
    context->input_list("inputs", &input_list);

    OpInputList infeed_list;
    context->input_list("infeeds", &infeed_list);

    OpOutputList outfeed_list;
    context->output_list("outfeeds", &outfeed_list);

    int pushed_cb_count = 0;
    int processed_cb_count = 0;
    std::condition_variable cb_processed;
    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);

    auto& io_config = resources->IOCfg();

    for (auto& infeed_pair : io_config.GetInfeeds()) {
      std::string feed = infeed_pair.first;
      const IOItem& feed_params = infeed_pair.second;

      tensorflow::Tensor t = infeed_list[feed_params.tuple_index];
      CHECK(feed_params.shape == t.shape())
          << "Input tensor shape " << t.shape() << " for input "
          << feed_params.handle << " does not match shape in signature ("
          << feed_params.shape << ")\n";

      comm.PushInputData(feed, std::move(infeed_list[feed_params.tuple_index]));
    }

    for (auto& outfeed_pair : io_config.GetOutfeeds()) {
      std::string feed = outfeed_pair.first;
      const IOItem& feed_params = outfeed_pair.second;

      int64 i = feed_params.tuple_index;

      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, outfeed_list.allocate(i, feed_params.shape, &output_tensor),
          [i]() {
            LOG(FATAL) << "  Outfeed tensor allocation failed for tensor " << i;
          });

      pushed_cb_count++;
      comm.PushOutputCallback(
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
