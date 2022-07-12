/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/poplar_feed_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_transfer_manager.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/infeed_allocator.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"

namespace tensorflow {

namespace {
const std::set<DataType> ok_types = {
    DataType::DT_INT32, DataType::DT_FLOAT, DataType::DT_HALF,
    DataType::DT_BOOL,  DataType::DT_INT8,  DataType::DT_UINT8,
};

struct TypeToStringFormatter {
  void operator()(std::string* out, DataType type) const {
    out->append(DataTypeString(type));
  }
};

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

void GetFeedConfig(OpKernelConstruction* ctx,
                   xla::poplarplugin::PoplarFeedConfig& config) {
  std::string feed_id;
  int64 prefetch_depth;
  std::vector<tensorflow::DataType> types;
  bool optimise_latency;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &types));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feed_id", &feed_id));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("prefetch_depth", &prefetch_depth));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("optimise_latency", &optimise_latency));

  config.set_feed_id(feed_id);
  config.set_prefetch_depth(prefetch_depth);
  config.set_optimise_latency(optimise_latency);

  OP_REQUIRES(
      ctx, 0 < prefetch_depth,
      errors::InvalidArgument("Need 0 < prefetch_depth, got ", prefetch_depth));

  OP_REQUIRES(ctx, prefetch_depth < 256,
              errors::InvalidArgument("Need prefetch_depth < 256, got ",
                                      prefetch_depth));

  *(config.mutable_tf_data_types()) = {types.begin(), types.end()};

  // Verify that the input doesn't contain any data types that we
  // do not like.
  for (unsigned i = 0; i < types.size(); ++i) {
    if (ok_types.count(types[i]) == 0) {
      std::string acceptable =
          absl::StrJoin(ok_types, ",", TypeToStringFormatter());
      ctx->CtxFailureWithWarning(errors::FailedPrecondition(
          "Unsupported datatype ", DataTypeString(types[i]), " on index ", i,
          " of feed operation ", ctx->def().name(), " with feed id '", feed_id,
          "'. You should use a tf.DataSet.map() operation to cast the element "
          "into one of (",
          acceptable, ")."));
    }
  }
}

void GetOutfeedMode(OpKernelConstruction* ctx,
                    xla::poplarplugin::PoplarFeedConfig& config) {
  std::string outfeed_mode_str;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("outfeed_mode", &outfeed_mode_str));
  if (outfeed_mode_str == "all") {
    config.set_mode(xla::poplarplugin::PoplarFeedConfig::GetAll);
  } else if (outfeed_mode_str == "get_last") {
    config.set_mode(xla::poplarplugin::PoplarFeedConfig::GetLast);
  } else {
    OP_REQUIRES(
        ctx, false,
        errors::InvalidArgument("Unkown outfeed_mode : ", outfeed_mode_str,
                                ", supported values are 'all' and 'get_last'"));
  }
}

bool AllShapesNonTrivial(const std::vector<xla::Shape>& shapes) {
  return absl::c_all_of(shapes, [](const xla::Shape& shape) -> bool {
    return xla::ShapeUtil::ElementsIn(shape);
  });
}
}  // namespace

class PopDatastreamInfeedDequeueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamInfeedDequeueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    GetFeedConfig(ctx, config_);
    XlaShapesFromAttr(ctx, xla_shapes_);

    OP_REQUIRES(ctx, AllShapesNonTrivial(xla_shapes_),
                errors::InvalidArgument(
                    "Detected an input tensor to an infeed queue with a "
                    "dimension of size 0. Please remove it from the inputs."));
  }

  ~PopDatastreamInfeedDequeueOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto tuple_shape = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
    std::string config_str;
    if (!config_.SerializeToString(&config_str)) {
      ctx->CtxFailureWithWarning(errors::FailedPrecondition(
          "Could not serialize the infeed configuration."));
    }
    xla::XlaOp output_tuple = xla::Infeed(b, tuple_shape, config_str);
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(output_tuple, i));
    }
  }

 private:
  xla::poplarplugin::PoplarFeedConfig config_;
  std::vector<xla::Shape> xla_shapes_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamInfeedDequeueOp);
};

REGISTER_IPU_OP("PopDatastreamInfeedDequeue", PopDatastreamInfeedDequeueOp);

class IPUCreateDatasetIteratorOp : public OpKernel {
 public:
  explicit IPUCreateDatasetIteratorOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    GetFeedConfig(ctx, config_);

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
    XlaShapesFromAttr(ctx, xla_shapes_);
    OP_REQUIRES(ctx, AllShapesNonTrivial(xla_shapes_),
                errors::InvalidArgument(
                    "Detected an input tensor to an infeed queue with a "
                    "dimension of size 0. Please remove it from the inputs."));
  }

  ~IPUCreateDatasetIteratorOp() override {}

  void Compute(OpKernelContext* ctx) override {
    // Get the flr and create base parameters.
    FunctionLibraryRuntime* flr = ctx->function_library();
    IteratorContext::Params params(ctx);

    // Get the dataset
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    // Pass to the correct executor
    poplar_executor->CreateInfeedIterator(config_, xla_shapes_, params, flr,
                                          dataset);
  }

 private:
  int device_ordinal_;
  xla::poplarplugin::PoplarFeedConfig config_;
  std::vector<xla::Shape> xla_shapes_;
  TF_DISALLOW_COPY_AND_ASSIGN(IPUCreateDatasetIteratorOp);
};

REGISTER_KERNEL_BUILDER(Name("IPUCreateDatasetIterator").Device(DEVICE_CPU),
                        IPUCreateDatasetIteratorOp);

class IPUDeleteDatasetIteratorOp : public OpKernel {
 public:
  explicit IPUDeleteDatasetIteratorOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feed_id", &feed_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
  }

  ~IPUDeleteDatasetIteratorOp() override {}

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    OP_REQUIRES_OK(ctx, poplar_executor->DeleteInfeedIterator(feed_id_));
  }

 private:
  int device_ordinal_;
  std::string feed_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(IPUDeleteDatasetIteratorOp);
};

REGISTER_KERNEL_BUILDER(Name("IPUDeleteDatasetIterator").Device(DEVICE_CPU),
                        IPUDeleteDatasetIteratorOp);

class PopDatastreamOutfeedEnqueueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamOutfeedEnqueueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    GetFeedConfig(ctx, config_);
    GetOutfeedMode(ctx, config_);
  }

  ~PopDatastreamOutfeedEnqueueOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const auto num_inputs = ctx->num_inputs();

    std::vector<xla::XlaOp> inputs(num_inputs);
    std::vector<xla::Shape> xla_shapes(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      inputs[i] = ctx->Input(i);
      auto input_shape = ctx->InputShape(i);
      auto dtype = ctx->input_type(i);
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtype, input_shape, &xla_shape));
      xla_shapes[i] = xla_shape;
    }

    OP_REQUIRES(ctx, AllShapesNonTrivial(xla_shapes),
                errors::InvalidArgument(
                    "Detected an input tensor to an outfeed queue with a "
                    "dimension of size 0. Please remove it from the inputs."));

    xla::Shape outfeed_shape;
    xla::XlaOp outfeed_input;
    const bool is_tuple = num_inputs > 1;
    if (is_tuple) {
      outfeed_shape = xla::ShapeUtil::MakeTupleShape(xla_shapes);
      outfeed_input = Tuple(b, inputs);
    } else {
      outfeed_shape = xla_shapes[0];
      outfeed_input = inputs[0];
    }

    std::string config_str;
    if (!config_.SerializeToString(&config_str)) {
      ctx->CtxFailureWithWarning(errors::FailedPrecondition(
          "Could not serialize the outfeed configuration."));
    }
    xla::XlaOp outfeed_token = CreateToken(b);
    xla::XlaOp outfeed = OutfeedWithToken(outfeed_input, outfeed_token,
                                          outfeed_shape, config_str);
  }

 private:
  xla::poplarplugin::PoplarFeedConfig config_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamOutfeedEnqueueOp);
};

REGISTER_IPU_OP("PopDatastreamOutfeedEnqueue", PopDatastreamOutfeedEnqueueOp);

class PopDatastreamOutfeedDequeueOp : public OpKernel {
 public:
  explicit PopDatastreamOutfeedDequeueOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    GetFeedConfig(ctx, config_);
    GetOutfeedMode(ctx, config_);
    XlaShapesFromAttr(ctx, xla_shapes_);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &tensor_shapes_));

    num_outputs_ = ctx->num_outputs();
    OP_REQUIRES(ctx,
                static_cast<size_t>(ctx->num_outputs()) == xla_shapes_.size(),
                errors::InvalidArgument(
                    "Outfeed num_outputs() != Attribute num outputs: ",
                    ctx->num_outputs(), " != ", xla_shapes_.size()));

    OP_REQUIRES(ctx, AllShapesNonTrivial(xla_shapes_),
                errors::InvalidArgument(
                    "Detected an input tensor to an outfeed queue with a "
                    "dimension of size 0. Please remove it from the inputs."));
  }

  ~PopDatastreamOutfeedDequeueOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    const auto replication_factor =
        poplar_executor->GetReplicationFactorForOutfeed(config_.feed_id());

    // Get all the tensors which were stored in the outfeed.
    // Note that this call will block until we can acquire a lock on the
    // outfeed.
    auto outfeed_tensors = poplar_executor->GetTensorsFromOutfeed(
        config_.feed_id(), config_.mode());
    if (config_.mode() == xla::poplarplugin::PoplarFeedConfig::GetAll) {
      // Allocate all the output buffers with the extra dimension for the number
      // of executions.
      std::vector<Tensor*> output_tensors;
      for (size_t i = 0; i < num_outputs_; ++i) {
        // Insert an extra dimension to the shape to represent the number of
        // iterations.
        TensorShape tensor_shape = tensor_shapes_[i];
        if (replication_factor > 1) {
          tensor_shape.InsertDim(0, replication_factor);
        }
        tensor_shape.InsertDim(0, outfeed_tensors.size());
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output(i, tensor_shape, &output_tensor));
        output_tensors.push_back(output_tensor);
      }
      // Copy the data into the output tensors.
      // Go through all the iterations, and copy the tensors into the right
      // output slices.
      for (size_t iteration = 0; iteration < outfeed_tensors.size();
           ++iteration) {
        auto& tensors_for_iteration = outfeed_tensors[iteration];
        CHECK_EQ(tensors_for_iteration.size(), num_outputs_);
        for (size_t j = 0; j < num_outputs_; ++j) {
          OP_REQUIRES_OK(
              ctx, batch_util::CopyElementToSlice(
                       tensors_for_iteration[j], output_tensors[j], iteration));
        }
      }
    } else {
      // Just set the output data if we are getting the last element.
      CHECK_EQ(config_.mode(), xla::poplarplugin::PoplarFeedConfig::GetLast);
      if (outfeed_tensors.size() == 1) {
        CHECK_EQ(outfeed_tensors[0].size(), num_outputs_);
        for (size_t j = 0; j < num_outputs_; ++j) {
          ctx->set_output(j, outfeed_tensors[0][j]);
        }
      } else if (xla::poplarplugin::UseSyntheticDataFor(
                     xla::poplarplugin::SyntheticDataCategory::Outfeed)) {
        OP_REQUIRES(ctx, false,
                    errors::FailedPrecondition(
                        "Trying to get the last value from an outfeed queue "
                        "when using synthethic data. This is not supported."));
      } else {
        // Dequeue was called before there was any data.
        OP_REQUIRES(ctx, false,
                    errors::FailedPrecondition(
                        "Trying to get the last value from an outfeed queue "
                        "before it was executed. Make sure the outfeed dequeue "
                        "TensorFlow operation is executed after the enqueue to "
                        "the outfeed has completed."));
      }
    }
  }

 private:
  int device_ordinal_;
  std::vector<xla::Shape> xla_shapes_;
  std::vector<TensorShape> tensor_shapes_;
  size_t num_outputs_;
  xla::poplarplugin::PoplarFeedConfig config_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamOutfeedDequeueOp);
};

REGISTER_KERNEL_BUILDER(Name("PopDatastreamOutfeedDequeue").Device(DEVICE_CPU),
                        PopDatastreamOutfeedDequeueOp);

class IPUDeleteOutfeedOp : public OpKernel {
 public:
  explicit IPUDeleteOutfeedOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feed_id", &feed_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
  }

  ~IPUDeleteOutfeedOp() override {}

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());

    if (poplar_executor->HasOutfeed(feed_id_)) {
      OP_REQUIRES_OK(ctx, poplar_executor->DeleteOutfeed(feed_id_));
    } else {
      VLOG(2) << "Can't delete Outfeed '" << feed_id_
              << "' as it doesn't exist.";
    }
  }

 private:
  int device_ordinal_;
  std::string feed_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(IPUDeleteOutfeedOp);
};

REGISTER_KERNEL_BUILDER(Name("IPUDeleteOutfeed").Device(DEVICE_CPU),
                        IPUDeleteOutfeedOp);

}  // namespace tensorflow
