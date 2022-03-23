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
#include "tensorflow/compiler/plugin/poplar/kernels/dataset/buffer_dataset_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const BufferDatasetOp::kDatasetType;
/* static */ constexpr const char* const BufferDatasetOp::kInputDataset;
/* static */ constexpr const char* const BufferDatasetOp::kBufferSize;
/* static */ constexpr const char* const BufferDatasetOp::kOutputTypes;
/* static */ constexpr const char* const BufferDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kBufferDataset[] = "BufferDataset";

constexpr char kBuffer[] = ".buffer";
constexpr char kBufferSize[] = ".buffer_size";
constexpr char kBufferPosition[] = ".buffer_position";
constexpr char kSizeSuffix[] = ".size";

class BufferDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64 buffer_size, const DatasetBase* input,
          int op_version)
      : DatasetBase(DatasetContext(ctx)),
        buffer_size_(buffer_size),
        input_(input),
        op_version_(op_version) {
    input_->Ref();

    output_shapes_ = input_->output_shapes();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    params.set_args(buffer_size_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64 Cardinality() const override {
    int64 n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n - (n % buffer_size_);
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, buffer_size}, {}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      buffer_size_ = dataset()->buffer_size_;
      buffer_position_ = buffer_size_;
      buffer_.resize(buffer_size_);
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (buffer_position_ == buffer_size_) {
        // We need to load elements.
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        *end_of_sequence = false;
        int buffer_idx = 0;
        for (int i = 0; i < buffer_size_ && !*end_of_sequence; ++i) {
          std::vector<Tensor> buffer_element_tuple;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &buffer_[i], end_of_sequence));
          if (!*end_of_sequence) {
            buffer_idx++;
          } else {
            input_impl_.reset();
          }
        }

        // We could not load enough data hence we are dropping it.
        if (buffer_idx < buffer_size_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        buffer_position_ = 0;
      }
      // Set the output.
      *out_tensors = std::move(buffer_[buffer_position_]);
      // We can move the buffer position.
      ++buffer_position_;
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), buffer_size_);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        // Save all the remaining buffers.
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kBufferSize), buffer_size_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kBufferPosition), buffer_position_));

        for (size_t i = buffer_position_; i < buffer_size_; i++) {
          std::vector<Tensor>& tensors = buffer_[i];
          // Store the size
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
              tensors.size()));
          // Store all the buffers.
          for (size_t j = 0; j < tensors.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                tensors[j]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        {
          int64 temp;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kBufferSize), &temp));
          buffer_size_ = static_cast<size_t>(temp);
        }
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kBufferPosition), &temp));
          buffer_position_ = static_cast<size_t>(temp);
        }
        // Resize the storage.
        buffer_.clear();
        buffer_.resize(buffer_size_);

        for (size_t i = buffer_position_; i < buffer_size_; i++) {
          // Get how many tensors we need to read.
          size_t num_tensors;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
                &temp));
            num_tensors = static_cast<size_t>(temp);
          }
          buffer_[i].resize(num_tensors);
          // Read the tensors.
          for (size_t j = 0; j < num_tensors; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                &buffer_[i][j]));
          }
        }
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

    data::TraceMeMetadata GetTraceMeMetadata() const override {
      data::TraceMeMetadata result;
      result.push_back(std::make_pair(
          "buffer_size",
          strings::Printf("%lld", static_cast<size_t>(buffer_size_))));
      return result;
    }

   private:
    size_t buffer_size_ = -1;
    // Stores the next position in the buffer we should read from. If equal to
    // buffer_size we need to load more data.
    size_t buffer_position_ = -1;
    // Internal storage.
    std::vector<std::vector<Tensor>> buffer_;

    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  };
  const int64 buffer_size_;
  const DatasetBase* const input_;
  const int op_version_;
  std::vector<PartialTensorShape> output_shapes_;
};

BufferDatasetOp::BufferDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), op_version_(1) {}

void BufferDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("Buffer size must be greater than zero."));

  *output = new Dataset(ctx, buffer_size, input, op_version_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("BufferDataset").Device(DEVICE_CPU),
                        BufferDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
