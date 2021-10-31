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
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
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

using Tracepoint = xla::poplarplugin::TensorflowPoplarPluginTracepoint;
using TensorVector = std::vector<tensorflow::Tensor>;

namespace tensorflow {
namespace {
using OutputCallback = std::function<void(void*)>;

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

// Base class which is used for processing the results from outfeed callbacks.
class ResultProcessorBase {
 public:
  // Returns whether all outputs have been processed.
  virtual bool ProcessOutput(const std::string& name, void* data) {
    // Check whether this is the last callback.
    return ++processed_outputs_ == num_outputs_;
  }

  // A method called once all the outputs have been processed.
  virtual void Done() {}

  // A method called to indicate failure.
  virtual void Abort(Status status) { Done(); }

  // Returns whether this result is for the user or not.
  virtual bool IsUserResult() const = 0;

 protected:
  explicit ResultProcessorBase(std::size_t num_outputs)
      : num_outputs_(num_outputs) {}

 private:
  const std::size_t num_outputs_;
  std::atomic<std::size_t> processed_outputs_{0};
};

// A wrapper class which copies the outputs and calls the callback done once all
// the results have been processed.
class ResultProcessor : public ResultProcessorBase {
 public:
  ResultProcessor(
      OpKernelContext* ctx, AsyncOpKernel::DoneCallback done,
      const absl::flat_hash_map<std::string, TensorBuffer*>& output_tensors)
      : ResultProcessorBase(output_tensors.size()),
        ctx_(ctx),
        done_(std::move(done)),
        output_tensors_(output_tensors) {}

  bool ProcessOutput(const std::string& name, void* data) override {
    // Copy the data.
    auto* buffer = output_tensors_.at(name);
    std::memcpy(buffer->data(), data, buffer->size());
    return ResultProcessorBase::ProcessOutput(name, data);
  }

  void Done() override { done_(); }

  void Abort(Status status) override {
    ctx_->CtxFailureWithWarning(status);
    Done();
  }

  bool IsUserResult() const override { return true; }

 private:
  OpKernelContext* ctx_;
  AsyncOpKernel::DoneCallback done_;
  const absl::flat_hash_map<std::string, TensorBuffer*> output_tensors_;
};

// Result processor for dummy data which has been pushed through.
class DummyResultProcessor : public ResultProcessorBase {
 public:
  explicit DummyResultProcessor(std::size_t num_outputs)
      : ResultProcessorBase(num_outputs) {}

  bool IsUserResult() const override { return false; }
};

class CommunicationManager {
 public:
  explicit CommunicationManager(PoplarExecutableProto& proto,
                                std::size_t timeout_us)
      : env_(tensorflow::Env::Default()),
        timeout_us_(timeout_us),
        status_(Status::OK()) {
    io_config_.ParsePoplarExecutableProto(proto);
    executable_can_stall_ =
        proto.embedded_runtime_config().executable_can_stall();
  }

  bool TryPeekInputData(const std::string& name, tensorflow::Tensor& result,
                        std::size_t look_ahead = 0) {
    TENSORFLOW_TRACEPOINT();
    {
      std::unique_lock<std::recursive_mutex> lk(io_mutex_);

      if (Exiting() || (input_queues_[name].size() <= look_ahead)) {
        return false;
      }

      result = input_queues_[name][look_ahead];
    }

    return true;
  }

  tensorflow::Tensor PeekInputData(const std::string& name,
                                   std::size_t look_ahead = 0) {
    TENSORFLOW_TRACEPOINT();
    tensorflow::Tensor result;
    {
      std::unique_lock<std::recursive_mutex> lk(io_mutex_);
      Tracepoint::BeginTrace("Wait");
      const auto predicate = [&, this]() {
        return Exiting() || (input_queues_[name].size() > look_ahead);
      };

      // If the application will stall waiting forever, only wait for the
      // timeout.
      if (executable_can_stall_ && user_requests_to_process_) {
        bool ready = input_cv_.wait_for(
            lk, std::chrono::microseconds(timeout_us_), predicate);

        // If we hit the timeout, push a dummy input instead.
        if (!Exiting() && !ready) {
          VLOG(2) << "Pipeline stalled - pushing dummy data.";
          auto status = PushInputDataAndResultProcessor(
              dummy_inputs_, absl::make_unique<DummyResultProcessor>(
                                 io_config_.GetOutfeeds().size()));

          // Report failure back to the user.
          if (!status.ok()) {
            Abort(status);
            InitiateExit();
          }
        }
      } else {
        // If we can't stall forever, just wait for data.
        input_cv_.wait(lk, predicate);
      }
      Tracepoint::EndTrace("Wait");

      if (!Exiting()) {
        result = input_queues_[name][look_ahead];
      }
    }

    return result;
  }

  void AdvanceInputData(const std::string& name) {
    TENSORFLOW_TRACEPOINT();
    std::unique_lock<std::recursive_mutex> lk(io_mutex_);

    Tracepoint::BeginTrace("Wait");
    input_cv_.wait(
        lk, [&, this]() { return Exiting() || !input_queues_[name].empty(); });
    Tracepoint::EndTrace("Wait");

    if (!Exiting()) {
      std::rotate(input_queues_[name].begin(), input_queues_[name].begin() + 1,
                  input_queues_[name].end());
      input_queues_[name].pop_back();
    }
  }

  OutputCallback PopOutputCallback(const std::string& name) {
    TENSORFLOW_TRACEPOINT();
    OutputCallback result;
    {
      std::unique_lock<std::recursive_mutex> lk(io_mutex_);

      Tracepoint::BeginTrace("Wait");
      output_cv_.wait(lk, [&, this]() {
        return Exiting() || !output_queues_[name].empty();
      });
      Tracepoint::EndTrace("Wait");

      if (!Exiting()) {
        result = std::move(output_queues_[name].front());
        output_queues_[name].pop();
      }
    }

    return result;
  }

  void PopResultProcessor() {
    TENSORFLOW_TRACEPOINT();
    std::unique_lock<std::recursive_mutex> lk(io_mutex_);
    result_processors_.pop();
  }

  Status PushInputDataAndResultProcessor(
      absl::flat_hash_map<std::string, tensorflow::Tensor> inputs,
      std::unique_ptr<ResultProcessorBase> result_processor) {
    TENSORFLOW_TRACEPOINT();
    std::unique_lock<std::recursive_mutex> lk(io_mutex_);

    if (!status_.ok()) {
      return status_;
    }

    for (auto& it : inputs) {
      input_queues_[it.first].push_back(std::move(it.second));
    }
    input_cv_.notify_one();

    result_processors_.push(std::move(result_processor));
    ResultProcessorBase* result_processor_ptr = result_processors_.back().get();

    // Keep track of how many user requests there are.
    if (result_processor_ptr->IsUserResult()) {
      ++user_requests_to_process_;
    }

    for (auto& outfeed_pair : io_config_.GetOutfeeds()) {
      const std::string feed = outfeed_pair.first;
      output_queues_[feed].push(
          [feed, result_processor_ptr, this](void* data) -> void {
            if (result_processor_ptr->ProcessOutput(feed, data)) {
              // Keep track of how many user requests there are.
              if (result_processor_ptr->IsUserResult()) {
                --user_requests_to_process_;
              }
              // All outfeeds have data now, call the done function and remove
              // the processor.
              result_processor_ptr->Done();
              PopResultProcessor();
            }
          });
    }
    output_cv_.notify_one();

    return Status::OK();
  }

  void InitiateExit() {
    engine_exiting_notification_.Notify();
    input_cv_.notify_all();
    output_cv_.notify_all();
  }

  bool Exiting() const {
    return engine_exiting_notification_.HasBeenNotified();
  }

  IOConfig& GetIOConfig() { return io_config_; }

  void Abort(Status status) {
    TENSORFLOW_TRACEPOINT();
    VLOG(1) << "Communication manager aborting: " << status;
    std::unique_lock<std::recursive_mutex> lk(io_mutex_);
    status_ = status;

    while (!result_processors_.empty()) {
      auto& result_processor = result_processors_.front();
      result_processor->Abort(status);
      result_processors_.pop();
    }

    input_queues_.clear();
    output_queues_.clear();
  }

  std::recursive_mutex& GetIOMutex() { return io_mutex_; }

  void ResetStatus() { status_ = Status::OK(); }

  void InitializeDummyInputs() {
    for (auto& it : io_config_.GetInfeeds()) {
      const auto& io_item = it.second;
      Tensor t{io_item.datatype, io_item.shape};
      auto buffer = tensorflow::DMAHelper::buffer(&t);
      // Default the values to zero.
      std::memset(buffer->data(), 0, buffer->size());
      dummy_inputs_.insert_or_assign(it.first, t);
    }
  }

 private:
  tensorflow::Env* env_;
  const std::size_t timeout_us_;

  // This notification is set when the engine needs to start exiting.
  absl::Notification engine_exiting_notification_;

  std::recursive_mutex io_mutex_;

  std::condition_variable_any input_cv_;
  std::condition_variable_any output_cv_;

  absl::flat_hash_map<std::string, std::vector<tensorflow::Tensor>>
      input_queues_ GUARDED_BY(io_mutex_);
  absl::flat_hash_map<std::string, std::queue<OutputCallback>> output_queues_
      GUARDED_BY(io_mutex_);
  std::queue<std::unique_ptr<ResultProcessorBase>> result_processors_
      GUARDED_BY(io_mutex_);
  Status status_ GUARDED_BY(io_mutex_);

  IOConfig io_config_;

  // Stores whether the executable is a model which can stall and which means
  // that data might need to be pushed through the if there are no inputs
  // incoming.
  bool executable_can_stall_ = false;

  // Dummy values which are pushed through the model by the dummy thread.
  absl::flat_hash_map<std::string, tensorflow::Tensor> dummy_inputs_;

  // Stores how many user call requests are left to process.
  std::atomic<std::size_t> user_requests_to_process_{0};
};

class PrefetchCallback : public poplar::StreamCallback {
 public:
  PrefetchCallback(CommunicationManager* comm_mgr, const std::string& name)
      : comm_mgr_(comm_mgr), name_(name), look_ahead_(0) {}

  poplar::StreamCallback::Result prefetch(void* dest) noexcept override {
    TENSORFLOW_TRACEPOINT();
    tensorflow::Tensor t;
    // Try to peek at the input data.
    if (comm_mgr_->TryPeekInputData(name_, t, look_ahead_)) {
      // Peek was successful, so memcpy to the poplar buffer.
      auto buffer = tensorflow::DMAHelper::buffer(&t);
      std::memcpy(dest, buffer->data(), buffer->size());
      look_ahead_++;

      // Indicate to poplar that the prefetch was successful.
      return poplar::StreamCallback::Result::Success;
    }

    // Indicate to poplar that the prefetch was not successful.
    return poplar::StreamCallback::Result::NotAvailable;
  }

  void fetch(void* dest) noexcept override {
    TENSORFLOW_TRACEPOINT();
    tensorflow::Tensor t = comm_mgr_->PeekInputData(name_, look_ahead_);
    if (!comm_mgr_->Exiting()) {
      auto buffer = tensorflow::DMAHelper::buffer(&t);
      std::memcpy(dest, buffer->data(), buffer->size());
      look_ahead_++;
    }
  }

  void complete() noexcept override {
    TENSORFLOW_TRACEPOINT();
    if (!comm_mgr_->Exiting()) {
      comm_mgr_->AdvanceInputData(name_);
      look_ahead_--;
    }

    // look_ahead_ should never become negative. This indicates more
    // completions than prefetches/fetches.
    CHECK_GE(look_ahead_, 0);
  }

  void invalidatePrefetched() noexcept override { look_ahead_ = 0; }

 private:
  CommunicationManager* comm_mgr_;
  const std::string name_;
  std::atomic<int64> look_ahead_;
};

class EngineResource {
 public:
  EngineResource(const std::string& engine_name, std::size_t timeout_us,
                 poplar::Executable&& executable, poplar::Device&& device,
                 PoplarExecutableProto& proto)
      : engine_name_(engine_name),
        device_(std::move(device)),
        engine_(std::move(executable)),
        communication_manager_(proto, timeout_us) {}

  poplar::Engine& GetEngine() { return engine_; }

  CommunicationManager& GetCommunicationManager() {
    return communication_manager_;
  }

  IOConfig& GetIOConfig() { return GetCommunicationManager().GetIOConfig(); }

  Status StartEngineAndConnectStreams(bool& reset_engine) {
    TENSORFLOW_TRACEPOINT();
    {
      // Prevent any new requests from being inserted whilst the engine is being
      // setup.
      std::unique_lock<std::recursive_mutex> lk(
          communication_manager_.GetIOMutex());

      {
        Tracepoint trace("engine_.load(device_)");
        try {
          engine_.load(device_);
        } catch (std::exception& e) {
          return xla::poplarplugin::PoplarExceptionToTensorflowStatus(
              "[Load engine]", e, reset_engine);
        }
      }

      TF_RETURN_IF_ERROR(ConnectStreams());
      communication_manager_.InitializeDummyInputs();

      VLOG(2) << "Engine loop starting.";
      {
        Tracepoint trace("engine_.run(0)");
        try {
          engine_.run(0);
        } catch (std::exception& e) {
          return xla::poplarplugin::PoplarExceptionToTensorflowStatus(
              "[Host to device]", e, reset_engine);
        }
      }
      communication_manager_.ResetStatus();
    }
    return Status::OK();
  }

  Status EngineLoop(bool& reset_engine) {
    TENSORFLOW_TRACEPOINT();
    do {
      try {
        engine_.run(1);
      } catch (std::exception& e) {
        return xla::poplarplugin::PoplarExceptionToTensorflowStatus(
            "[Execute engine]", e, reset_engine);
      }
    } while (!communication_manager_.Exiting());
    return Status::OK();
  }

  Status StartEngine(TensorVector&& input_tensors) {
    TENSORFLOW_TRACEPOINT();
    VLOG(2) << "Starting engine execution for " << engine_name_;

    if (execute_thread_) {
      return errors::Internal("Engine thread already exists for engine ",
                              engine_name_);
    }
    input_tensors_ = std::move(input_tensors);

    execute_thread_.reset(tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), engine_name_ + "_execute_thread", [this] {
          Status runtime_status = Status::OK();
          bool reset_engine = false;
          while (!communication_manager_.Exiting()) {
            {
              // Prevent any new requests from being inserted whilst the engine
              // is being setup.
              std::unique_lock<std::recursive_mutex> lk(
                  communication_manager_.GetIOMutex());

              // If the previous iteration of this loop was not ok abort all
              // existing requests.
              if (!runtime_status.ok()) {
                communication_manager_.Abort(runtime_status);
                // If the exception raised only requires an engine reset, then
                // continue, otherwise we can't recover.
                if (!reset_engine) {
                  communication_manager_.InitiateExit();
                  return;
                }
              }

              runtime_status = StartEngineAndConnectStreams(reset_engine);
              if (!runtime_status.ok()) {
                continue;
              }

              reset_engine = false;
              runtime_status = Status::OK();
            }
            runtime_status = EngineLoop(reset_engine);
          }
        }));

    return Status::OK();
  }

  // Populates the values for buffers which are not streamed every run
  // operation.
  Status PopulateAndConnectInputBuffers() {
    TENSORFLOW_TRACEPOINT();
    VLOG(2) << "Populating the input buffers.";
    // TODO(T41137): The assumption here is that the inputs are passed in the
    // correct order.
    auto& io_config = GetIOConfig();

    for (auto& input_pair : io_config.GetInputs()) {
      auto input = input_pair.first;
      auto& io_item = input_pair.second;
      tensorflow::Tensor& t = input_tensors_.at(io_item.argument);
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

  Status ConnectStreams() {
    TENSORFLOW_TRACEPOINT();
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
    TF_RETURN_IF_ERROR(PopulateAndConnectInputBuffers());

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
    TENSORFLOW_TRACEPOINT();
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

  // Thread which keeps running the engine until the communication manager is
  // asked to exit.
  // Note that the destructor blocks until the thread completes.
  std::unique_ptr<tensorflow::Thread> execute_thread_;

  // Storage of the input tensors used by program 0.
  TensorVector input_tensors_;
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
                      std::size_t timeout_us, const OpInputList& input_list) {
    TENSORFLOW_TRACEPOINT();
    std::unique_lock<std::recursive_mutex> lk(engine_resource_map_mutex_);
    if (EngineExists(engine_name)) {
      return Status::OK();
    }

    // Copy the input tensors to an internal storage incase the engine needs
    // restarting.
    TensorVector input_tensors(input_list.size());
    for (int64 i = 0; i != input_list.size(); ++i) {
      tensorflow::Tensor input = input_list[i];
      tensorflow::Tensor copy(input.dtype(), input.shape());
      tensorflow::StringPiece from = input.tensor_data();
      tensorflow::StringPiece to = copy.tensor_data();
      memcpy(const_cast<char*>(to.data()), from.data(), from.size());
      input_tensors[i] = copy;
    }

    VLOG(2) << "Creating an engine.";
    PoplarExecutableProto proto;
    TF_ASSIGN_OR_RETURN(
        poplar::Executable executable,
        PoplarExecutableBinaryFile::Read(executable_filename, &proto));

    // Check the the executable is compatible.
    VerifyExecutable(proto);

    // Check the versions.
    xla::poplarplugin::CheckPoplarPackageHash();
    const std::string poplar_package_hash = std::string(poplar::packageHash());
    if (proto.poplar_package_hash() != poplar_package_hash) {
      const std::string message = absl::StrCat(
          "Poplar package mismatch: The executable was compiled against "
          "Poplar package ",
          proto.poplar_package_hash(),
          ", however the current Poplar package is ", poplar_package_hash,
          ". ");
      if (xla::poplarplugin::PoplarXlaFlags::Get()
              .disable_poplar_version_check) {
        LOG(INFO) << message
                  << "This check has been manually disabled and this might "
                     "lead to unexpected issues.";
      } else {
        return errors::InvalidArgument(
            message, "Please make sure to use the correct Poplar version.");
      }
    }

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
        engine_name, timeout_us, std::move(executable), std::move(device),
        proto);

    TF_RETURN_IF_ERROR(engine_resource->StartEngine(std::move(input_tensors)));

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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_us", &timeout_us_));
  }

  void Compute(OpKernelContext* ctx) override {
    TENSORFLOW_TRACEPOINT();
    OpInputList input_list;
    ctx->input_list("inputs", &input_list);
    auto& engine_mgr = EngineManager::Instance();

    OP_REQUIRES_OK(ctx, engine_mgr.CreateEngine(engine_name_, filename_,
                                                timeout_us_, input_list));

    tensorflow::Tensor* anchor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("anchor", TensorShape({}), &anchor));
  }

 private:
  std::string filename_;
  std::string engine_name_;
  int64 timeout_us_;

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
    TENSORFLOW_TRACEPOINT();
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

    // Gather the inputs.
    absl::flat_hash_map<std::string, tensorflow::Tensor> inputs;
    for (auto& infeed_pair : io_config.GetInfeeds()) {
      std::string feed = infeed_pair.first;
      const IOItem& feed_params = infeed_pair.second;

      tensorflow::Tensor t = infeed_list[feed_params.tuple_index];
      OP_REQUIRES_OK_ASYNC(ctx, check_shape(feed_params, t), done);

      inputs.insert_or_assign(feed,
                              std::move(infeed_list[feed_params.tuple_index]));
    }

    // Allocate the outputs.
    absl::flat_hash_map<std::string, TensorBuffer*> output_tensors;
    for (auto& outfeed_pair : io_config.GetOutfeeds()) {
      const std::string& feed = outfeed_pair.first;
      const IOItem& feed_params = outfeed_pair.second;

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(ctx,
                           outfeed_list.allocate(feed_params.tuple_index,
                                                 feed_params.shape, &output),
                           done);
      output_tensors[feed] = tensorflow::DMAHelper::buffer(output);
    }

    // Set up the result processor.
    std::unique_ptr<ResultProcessor> result_processor =
        absl::make_unique<ResultProcessor>(ctx, done, output_tensors);

    OP_REQUIRES_OK_ASYNC(ctx,
                         comm_mgr.PushInputDataAndResultProcessor(
                             inputs, std::move(result_processor)),
                         done);
  }

 private:
  std::string engine_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(ApplicationCall);
};

REGISTER_KERNEL_BUILDER(Name("ApplicationCall").Device(DEVICE_CPU),
                        ApplicationCall);
}  // namespace tensorflow
