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

#include <fstream>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
#include "tensorflow/compiler/plugin/poplar/tools/poplar_executable_runner.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"

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

xla::StatusOr<ipu::DataType> PrimitiveTypeToDataType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::F32:
      return ipu::F32;
    case xla::PrimitiveType::F16:
      return ipu::F16;
    case xla::PrimitiveType::S32:
      return ipu::S32;
    default:
      return tensorflow::errors::InvalidArgument("Unsupported PrimitiveType ",
                                                 xla::PrimitiveType_Name(type));
  }
}

}  // namespace

class DatasetExtractor : public OpKernel {
 public:
  explicit DatasetExtractor(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("print_stats", &print_stats_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_elements", &num_elements_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feed_name", &name_));
    XlaShapesFromAttr(ctx, shapes_);
  }

  ~DatasetExtractor() override {}

  void Compute(OpKernelContext* ctx) override {
    // Get the flr and create base parameters.
    FunctionLibraryRuntime* flr = ctx->function_library();
    IteratorContext::Params params(ctx);

    // Get the dataset
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    CancellationManager cancellation_manager;
    xla::poplarplugin::InfeedAllocator infeed_allocator;
    xla::poplarplugin::InfeedIterator infeed_iterator(
        flr, params, dataset, &cancellation_manager, &infeed_allocator,
        /* replication factor */ 1, shapes_, "extractor");

    // We only ever have a single replica.
    auto queues = infeed_iterator.GetInfeedQueues();
    ipu::BinaryWriter writer{filename_};
    std::vector<ipu::FeedWriter> output_streams;
    OP_REQUIRES_OK(ctx, [&]() {
      try {
        for (uint64 i = 0; i != shapes_.size(); ++i) {
          const xla::Shape& xla_shape = shapes_[i];
          if (!xla::LayoutUtil::IsDenseArray(xla_shape)) {
            return xla::InvalidArgument(
                "All shapes in a feed element must be dense arrays");
          }
          TF_ASSIGN_OR_RETURN(
              ipu::DataType data_type,
              PrimitiveTypeToDataType(xla_shape.element_type()));
          std::vector<int64_t> dimensions;
          absl::c_transform(xla_shape.dimensions(),
                            std::back_inserter(dimensions),
                            [](int64 dim) { return dim; });
          ipu::TensorShape shape(dimensions, data_type);
          std::string name;
          if (name_.empty()) {
            name = "infeed";
          } else {
            name = name_;
          }
          name = absl::StrCat(name, ".", i);
          ipu::TensorInfo info{name, "", shape, ipu::TensorType::Infeed};
          output_streams.push_back(
              std::move(writer.CreateFeed(name, info, num_elements_)));
        }

        using seconds = std::chrono::duration<float>;
        auto t0 = std::chrono::steady_clock::now();
        for (int64_t remaining_elements = num_elements_; remaining_elements > 0;
             remaining_elements--) {
          if (queues[0][0]->IsFull()) {
            VLOG(1) << "Infeed queue is full.";
            continue;
          }

          if (queues[0][0]->IsEmpty()) {
            VLOG(1) << "Infeed queue is empty.";
          }

          std::vector<tensorflow::Tensor> outputs;
          bool end_of_sequence = false;
          TF_RETURN_IF_ERROR(
              infeed_iterator.GetNext(&outputs, &end_of_sequence));

          if (end_of_sequence) {
            return tensorflow::errors::OutOfRange(
                "The dataset iterator has reached the end of the dataset.");
          }
          if (print_stats_ && (remaining_elements % 1000 == 0)) {
            auto t1 = std::chrono::steady_clock::now();
            LOG(INFO) << "Exported " << (num_elements_ - remaining_elements)
                      << " out of " << num_elements_ << " in "
                      << seconds(t1 - t0).count() << " seconds";
          }

          for (size_t j = 0; j < outputs.size(); ++j) {
            TensorBuffer* tb = tensorflow::DMAHelper::buffer(&outputs[j]);
            output_streams[j].AppendTensor(tb->data());
          }
        }
        writer.Close();
        return Status::OK();
      } catch (const std::runtime_error& err) {
        return xla::InvalidArgument(err.what());
      }
    }());
  }

 private:
  bool print_stats_;
  std::string filename_;
  int num_elements_;
  std::string name_;
  std::vector<xla::Shape> shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(DatasetExtractor);
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("DatasetExtractor").Device(DEVICE_CPU),
                        DatasetExtractor);
class VariablesExporter : public OpKernel {
 public:
  explicit VariablesExporter(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("print_stats", &print_stats_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_input", &is_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names_));
    // Remove the ":0" suffix added by TF:
    absl::c_for_each(names_, [](std::string& name) {
      if (name.size() > 2 && name.substr(name.size() - 2, 2) == ":0") {
        name = name.substr(0, name.size() - 2);
      }
    });
  }

  ~VariablesExporter() override {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, [&]() {
      ipu::BinaryWriter writer{filename_};
      for (int i = 0; i < ctx->num_inputs(); i++) {
        Tensor input = ctx->input(i);
        xla::PrimitiveType xla_type;
        TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(input.dtype(), &xla_type));
        xla::Shape xla_shape = TensorShapeToXLAShape(xla_type, input.shape());
        TF_ASSIGN_OR_RETURN(ipu::DataType data_type,
                            PrimitiveTypeToDataType(xla_shape.element_type()));

        // Convert from vector<int64> to vector<int64_t> (long long int vs long
        // int)
        std::vector<int64_t> dimensions;
        absl::c_transform(xla_shape.dimensions(),
                          std::back_inserter(dimensions),
                          [](int64 dim) { return dim; });

        ipu::TensorShape shape(dimensions, data_type);
        ipu::TensorInfo info{names_[i], "", shape,
                             is_input_ ? ipu::TensorType::InputData
                                       : ipu::TensorType::Parameter};

        TensorBuffer* tb = tensorflow::DMAHelper::buffer(&input);
        ipu::Tensor out{info, tb->data()};
        writer.WriteTensor(out);
      }
      return Status::OK();
    }());
  }

 private:
  bool print_stats_;
  bool is_input_;
  std::string filename_;
  std::vector<std::string> names_;

  TF_DISALLOW_COPY_AND_ASSIGN(VariablesExporter);
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("VariablesExporter").Device(DEVICE_CPU),
                        VariablesExporter);
}  // namespace tensorflow
