/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"

#include <popdist/backend.hpp>
#include <popdist/collectives.hpp>
#include <popdist/context.hpp>

namespace tensorflow {
template <typename T>
class PopDistAllReduceOp : public OpKernel {
 public:
  explicit PopDistAllReduceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto& input = ctx->input(0);
    Tensor* output;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    OP_REQUIRES_OK(ctx, tensorflow::functor::DoCopy(ctx->eigen_cpu_device(),
                                                    input, output));

    auto* flattened_buffer = output->flat<T>().data();

    popdist::collectives::sequential::allReduceSum(
        flattened_buffer, input.NumElements(),
        poplar::equivalent_device_type<T>().value);

    const auto num_instances = popdist::getNumInstances();

    if (reduce_op_ == "MEAN") {
      for (auto i = 0; i < input.NumElements(); ++i) {
        *(flattened_buffer + i) /= num_instances;
      }
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopDistAllReduceOp);

  std::string reduce_op_;
};

#define REGISTER_CPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("PopdistAllReduce").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PopDistAllReduceOp<T>);

TF_CALL_INTEGRAL_TYPES(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
#undef REGISTER_CPU
}  // namespace tensorflow
