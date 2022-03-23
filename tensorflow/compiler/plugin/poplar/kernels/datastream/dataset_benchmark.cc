/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poplar/replication_factor.hpp>
#include <poputil/TileMapping.hpp>

#include "absl/container/flat_hash_set.h"
#include "include/json/json.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/io_thread.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {
namespace {
void XlaShapesFromAttr(OpKernelConstruction* ctx,
                       std::vector<xla::Shape>& result) {
  std::vector<TensorShape> shapes;
  std::vector<tensorflow::DataType> types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &shapes));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &types));

  for (unsigned i = 0; i < shapes.size(); ++i) {
    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(types[i], &xla_type));
    result.emplace_back(TensorShapeToXLAShape(xla_type, shapes[i]));
  }
}

xla::poplarplugin::IOFunction ConsumerThread(
    bool print_stats, bool do_memcpy, uint64 number_of_epochs,
    uint64 elements_per_epochs, Notification* finished_notifiation,
    Json::Value* stats_json, xla::poplarplugin::InfeedIterator* itr) {
  return [=](std::atomic<bool>& cancelled) {
    Json::Value epochs(Json::arrayValue);

    // We only ever have a single replica.
    const size_t replica_id = 0;
    auto queues = itr->GetInfeedQueues();
    auto shapes = itr->GetShapes();

    // Allocate the buffers we copy into.
    std::vector<void*> buffers(shapes.size());
    std::vector<size_t> buffer_sizes(shapes.size());
    std::size_t total_bytes = 0;
    for (uint64 i = 0; i != shapes.size(); ++i) {
      buffer_sizes[i] = xla::ShapeUtil::ByteSizeOf(shapes[i]);
      total_bytes += buffer_sizes[i];
      buffers[i] = port::AlignedMalloc(buffer_sizes[i], 4096);
    }

    // Run for the amount of time user has asked us to.
    for (uint64 i = 0; i != number_of_epochs; ++i) {
      using seconds = std::chrono::duration<float>;
      auto t0 = std::chrono::high_resolution_clock::now();
      for (uint64 j = 0; j != elements_per_epochs; ++j) {
        auto& replica_queues = queues[replica_id];
        for (uint64 k = 0; k != replica_queues.size(); ++k) {
          tensorflow::TensorBuffer* buf;
          // Continue to try and get a buffer unless we have been cancelled.
          while (!replica_queues[k]->TryPop(buf)) {
            if (cancelled) {
              return tensorflow::errors::Aborted("Consumer thread cancelled");
            }
          }
          if (do_memcpy) {
            std::memcpy(buffers[k], buf->data(), buffer_sizes[k]);
          }
          replica_queues[k]->AdvanceReadPosition();
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      Json::Value epoch_stats;
      const float elements_processed = static_cast<float>(elements_per_epochs);
      const float total_bytes_processed = elements_processed * total_bytes;
      const float time_elapsed = static_cast<float>(seconds(t1 - t0).count());
      const float elements_per_second = elements_processed / time_elapsed;
      const float bandwidth =
          total_bytes_processed / (1000000000.f * time_elapsed);

      epoch_stats["elements_processed"] = elements_processed;
      epoch_stats["total_bytes_processed"] = total_bytes_processed;
      epoch_stats["time_elapsed"] = time_elapsed;
      epoch_stats["elements_per_second"] = elements_per_second;
      epoch_stats["bandwidth"] = bandwidth;
      epochs.append(epoch_stats);

      if (print_stats) {
        LOG(INFO) << "Processed: " << elements_per_second
                  << " elements/second.";
        LOG(INFO) << "Bandwidth: " << bandwidth << " GB/s.";
      }
      LOG(INFO) << "Dataset iterator completed epoch " << i << ".";
    }
    // Store stats into the json.
    (*stats_json)["epochs"] = epochs;
    // Notify we have finished.
    finished_notifiation->Notify();
    return Status::OK();
  };
}

xla::poplarplugin::IOFunction ProducerThread(
    xla::poplarplugin::InfeedIterator* itr) {
  return [=](std::atomic<bool>& cancelled) {
    // We only ever have a single replica.
    const size_t replica_id = 0;
    auto queues = itr->GetInfeedQueues();

    while (!cancelled) {
      if (queues[0][0]->IsFull()) {
        VLOG(1) << "Infeed queue is full.";
        continue;
      }

      if (queues[0][0]->IsEmpty()) {
        VLOG(1) << "Infeed queue is empty.";
      }

      std::vector<tensorflow::Tensor> outputs;
      bool end_of_sequence = false;
      TF_RETURN_IF_ERROR(itr->GetNext(&outputs, &end_of_sequence));

      if (end_of_sequence) {
        return tensorflow::errors::OutOfRange(
            "The dataset iterator has reached the end of the dataset.");
      }

      for (size_t j = 0; j < outputs.size(); ++j) {
        auto& queue = queues[replica_id][j];
        TensorBuffer* tb = tensorflow::DMAHelper::buffer(&outputs[j]);
        tb->Ref();
        queue->BlockPush(tb);
        queue->AdvanceWritePosition();
      }
    }
    return Status::OK();
  };
}

}  // namespace

class DatasetBenchmark : public OpKernel {
 public:
  explicit DatasetBenchmark(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("print_stats", &print_stats_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("do_memcpy", &do_memcpy_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("number_of_epochs", &number_of_epochs_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("elements_per_epochs", &elements_per_epochs_));
    XlaShapesFromAttr(ctx, shapes_);
  }

  ~DatasetBenchmark() override {}

  void Compute(OpKernelContext* ctx) override {
    // Get the flr and create base parameters.
    FunctionLibraryRuntime* flr = ctx->function_library();
    data::IteratorContext::Params params(ctx);

    // Get the dataset
    data::DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    xla::poplarplugin::InfeedAllocator infeed_allocator;
    xla::poplarplugin::InfeedIterator infeed_iterator(
        flr, params, dataset, &infeed_allocator, shapes_, "benchmark");

    infeed_iterator.SetReplicationFactor(1);

    Json::Value stats_json;
    {
      Notification consumer_finished_notifiation;

      // Start the consumer thread.
      xla::poplarplugin::IOThread consumer_thread(
          "consumer",
          ConsumerThread(print_stats_, do_memcpy_, number_of_epochs_,
                         elements_per_epochs_, &consumer_finished_notifiation,
                         &stats_json, &infeed_iterator));

      // Start the producer thread.
      xla::poplarplugin::IOThread producer_thread(
          "producer", ProducerThread(&infeed_iterator));

      // Wait until the consumer thread has finished - note that the thread will
      // be cancelled and destroyed as we exit the scope.
      consumer_finished_notifiation.WaitForNotification();
    }
    // Copy the output out as a string.
    Json::StreamWriterBuilder builder;
    std::stringstream ss;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(stats_json, &ss);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("out", TensorShape({1}), &output_tensor));
    output_tensor->flat<tstring>()(0) = ss.str();
  }

 private:
  bool print_stats_;
  bool do_memcpy_;
  int number_of_epochs_;
  int elements_per_epochs_;
  std::vector<xla::Shape> shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(DatasetBenchmark);
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("DatasetBenchmark").Device(DEVICE_CPU),
                        DatasetBenchmark);
}  // namespace tensorflow
