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

#include "ipu/poplar_executable_data.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_iterator.h"
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

xla::StatusOr<ipu::TensorShape> ConvertShapeToIpuTensorInfo(
    const xla::Shape& xla_shape) {
  TF_ASSIGN_OR_RETURN(ipu::DataType data_type,
                      PrimitiveTypeToDataType(xla_shape.element_type()));

  // Convert from vector<int64> to vector<int64_t> (long long int vs long
  // int)
  std::vector<int64_t> dimensions;
  absl::c_transform(xla_shape.dimensions(), std::back_inserter(dimensions),
                    [](int64 dim) { return dim; });

  return ipu::TensorShape{dimensions, data_type};
}

xla::StatusOr<ipu::TensorShape> ConvertTensorToIpuTensorInfo(
    const Tensor& tensor) {
  xla::PrimitiveType xla_type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tensor.dtype(), &xla_type));
  xla::Shape xla_shape = TensorShapeToXLAShape(xla_type, tensor.shape());
  return ConvertShapeToIpuTensorInfo(xla_shape);
}

void CleanUpNames(std::vector<std::string>& names) {
  // Remove the ":0" suffix added by TF:
  absl::c_for_each(names, [](std::string& name) {
    if (name.size() > 2 && name.substr(name.size() - 2, 2) == ":0") {
      name = name.substr(0, name.size() - 2);
    }
  });
}

void FindMissingTensors(const ipu::Metadata& metadata,
                        ipu::TensorType tensor_type,
                        const std::vector<std::string>& tf_names,
                        std::vector<std::string>& missing_from_tf,
                        std::vector<std::string>& missing_from_files) {
  auto data_found = [&metadata, tensor_type](const std::string& name) {
    for (auto tensor : metadata.inputs) {
      if (tensor.Name() == name && tensor.Type() == tensor_type) {
        return true;
      }
    }
    return false;
  };
  auto tensor_found = [tf_names](const std::string& name) {
    return absl::c_find(tf_names, name) != tf_names.end();
  };

  for (auto name : tf_names) {
    if (!data_found(name)) {
      missing_from_files.push_back(name);
    }
  }
  for (auto tensor : metadata.inputs) {
    if (tensor.Type() != tensor_type) {
      continue;
    }
    if (!tensor_found(tensor.Name())) {
      missing_from_tf.push_back(tensor.Name());
    }
  }
}

const ipu::TensorInfo& InputInfoFromMetadata(const ipu::Metadata& metadata,
                                             ipu::TensorType tensor_type,
                                             const std::string& name) {
  for (auto& tensor : metadata.inputs) {
    if (tensor.Type() == tensor_type && tensor.Name() == name) {
      return tensor;
    }
  }
  VLOG(0) << "Unreachable: FindMissingTensors should have already checked all "
             "the tensors are present";
}

std::shared_ptr<ipu::Metadata> LoadMetadata(const std::string& filename) {
  if (ipu::IsJsonFile(filename)) {
    return std::make_shared<ipu::Metadata>(ipu::LoadJsonFromFile(filename));
  }
  ipu::BinaryReader loader;
  loader.LoadFile(filename);
  return loader.ReadMetadata();
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

    xla::poplarplugin::InfeedAllocator infeed_allocator;
    xla::poplarplugin::InfeedIterator infeed_iterator(
        flr, params, dataset, &infeed_allocator,
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("metadata_file", &metadata_));
    // Remove the ":0" suffix added by TF:
    CleanUpNames(names_);
  }

  ~VariablesExporter() override {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, [&]() {
      try {
        const ipu::TensorType tensor_type =
            is_input_ ? ipu::TensorType::InputData : ipu::TensorType::Parameter;
        std::shared_ptr<ipu::Metadata> meta;
        if (!metadata_.empty()) {
          meta = LoadMetadata(metadata_);
          std::vector<std::string> missing_metadata;
          std::vector<std::string> missing_tensors;
          FindMissingTensors(*meta, tensor_type, names_, missing_tensors,
                             missing_metadata);
          std::string error;
          if (!missing_metadata.empty()) {
            error += " The metadata doesn't contain the following tensors: " +
                     absl::StrJoin(missing_metadata, ",");
          }
          if (!missing_tensors.empty()) {
            error +=
                " The following tensors are present in the metadata but not in "
                "the graph: " +
                absl::StrJoin(missing_tensors, ",");
          }
          if (!error.empty()) {
            return tensorflow::errors::InvalidArgument(error);
          }
        }
        ipu::BinaryWriter writer{filename_};
        for (int i = 0; i < ctx->num_inputs(); i++) {
          Tensor input = ctx->input(i);
          TF_ASSIGN_OR_RETURN(ipu::TensorShape shape,
                              ConvertTensorToIpuTensorInfo(input));
          ipu::TensorInfo info{names_[i], "", shape, tensor_type};
          if (meta && !info.TypeAndShapeMatch(InputInfoFromMetadata(
                          *meta, tensor_type, names_[i]))) {
            return tensorflow::errors::InvalidArgument(
                absl::StrCat("Mismatch in type/shape between the metadata and "
                             "the graph tensor for ",
                             names_[i]));
          }

          TensorBuffer* tb = tensorflow::DMAHelper::buffer(&input);
          ipu::Tensor out{info, tb->data()};
          writer.WriteTensor(out);
        }
        return Status::OK();
      } catch (const std::runtime_error& err) {
        return xla::InvalidArgument(err.what());
      }
    }());
  }

 private:
  bool print_stats_;
  bool is_input_;
  std::string filename_;
  std::vector<std::string> names_;
  std::string metadata_;

  TF_DISALLOW_COPY_AND_ASSIGN(VariablesExporter);
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("VariablesExporter").Device(DEVICE_CPU),
                        VariablesExporter);

class VariablesImporter : public OpKernel {
 public:
  explicit VariablesImporter(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("print_stats", &print_stats_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_input", &is_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strict", &strict_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filenames", &filenames_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("names", &names_));
    XlaShapesFromAttr(ctx, shapes_);
    // Remove the ":0" suffix added by TF:
    CleanUpNames(names_);
  }

  ~VariablesImporter() override {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, [&]() {
      try {
        ipu::BinaryReader loader;
        // Parse all the provided binary files
        // Note: this builds a list of all the objects stored
        // in the files but doesn't actually load any data.
        for (auto file : filenames_) {
          loader.LoadFile(file);
        }
        const ipu::TensorType tensor_type =
            is_input_ ? ipu::TensorType::InputData : ipu::TensorType::Parameter;
        if (strict_) {
          // If we're in strict mode: load the metadata from the binaries and
          // check the graph contains exactly the same inputs / parameters.
          std::shared_ptr<ipu::Metadata> metadata = loader.ReadMetadata();
          std::vector<std::string> missing_data;
          std::vector<std::string> missing_tensors;
          FindMissingTensors(*metadata, tensor_type, names_, missing_tensors,
                             missing_data);
          std::string error;
          if (!missing_data.empty()) {
            error +=
                " The binaries provided didn't contain any data for the "
                "following tensors: " +
                absl::StrJoin(missing_data, ",");
          }
          if (!missing_tensors.empty()) {
            error +=
                " The binaries provided contain data for the following tensors "
                "which cannot be found in the graph: " +
                absl::StrJoin(missing_tensors, ",");
          }
          if (!error.empty()) {
            return tensorflow::errors::InvalidArgument(error);
          }
        }

        // For each tensor in the graph
        for (int i = 0; i < shapes_.size(); i++) {
          TF_ASSIGN_OR_RETURN(ipu::TensorShape shape,
                              ConvertShapeToIpuTensorInfo(shapes_[i]));
          ipu::TensorInfo info{names_[i], "", shape, tensor_type};
          // Check if the binaries provided contains data for it.
          if (!loader.ContainsObject(ipu::ObjectType::Tensor, names_[i])) {
            // No data available for this tensor: skip it.
            continue;
          }
          // Get a data stream from the Loader
          std::unique_ptr<ipu::StreamReader> reader =
              loader.GetTensorStream(names_[i]);
          // Load the data in a temporary Tensor
          ipu::Tensor out{*reader};
          // Make sure the tensor in the graph has the same shape and
          // type as the one from the binary files.
          if (!info.TypeAndShapeMatch(out.Info())) {
            return tensorflow::errors::InvalidArgument(
                "For tensor ", names_[i], " the tensorflow info ",
                info.ToString(), " doesn't match the one from the ipu::Tensor ",
                out.Info().ToString());
          }
          // Allocate the output Tensorflow tensor.
          Tensor* output = nullptr;
          TensorShape tensor_shape;
          TF_RETURN_IF_ERROR(XLAShapeToTensorShape(shapes_[i], &tensor_shape));
          TF_RETURN_IF_ERROR(ctx->allocate_output(i, tensor_shape, &output));
          TensorBuffer* tb = tensorflow::DMAHelper::buffer(output);
          if (tb->size() != out.Info().Shape().DataSizeInBytes()) {
            return tensorflow::errors::InvalidArgument(
                "Tensorflow tensor size ", tb->size(),
                " doesn't match the size of the ipu::Tensor size ",
                out.Info().Shape().DataSizeInBytes(), " for tensor ",
                names_[i]);
          }
          // Copy the data from the ipu::Tensor to the Tensorflow tensor.
          memcpy(tb->data(), out.Data(), tb->size());
        }
        return Status::OK();
      } catch (const std::exception& e) {
        return tensorflow::errors::InvalidArgument(e.what());
      }
    }());
  }

 private:
  bool print_stats_;
  bool is_input_;
  bool strict_;
  std::vector<std::string> filenames_;
  std::vector<std::string> names_;
  std::vector<xla::Shape> shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(VariablesImporter);
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("VariablesImporter").Device(DEVICE_CPU),
                        VariablesImporter);
}  // namespace tensorflow
