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
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"

#include <popdist/backend.hpp>
#include <popdist/collectives.hpp>
#include <popdist/context.hpp>

namespace poplar {
template <>
struct equivalent_device_type<Eigen::half> {
  const Type& value = HALF;
};
}  // namespace poplar

namespace tensorflow {
template <typename T>
class PopDistBroadcastOp : public AsyncOpKernel {
 public:
  explicit PopDistBroadcastOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto& input = ctx->input(0);
    Tensor* output;

    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input.shape(), &output),
                         done);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        tensorflow::functor::DoCopy(ctx->eigen_cpu_device(), input, output),
        done);

    Env::Default()->SchedClosure([output, ctx, done, this] {
      OP_REQUIRES_OK_ASYNC(
          ctx,
          xla::poplarplugin::RunPoplarFunction<popdist::popdist_error>(
              [&output, &ctx, &done, this] {
                popdist::collectives::broadcast(
                    output->flat<T>().data(), output->NumElements(),
                    poplar::equivalent_device_type<T>().value,
                    0 /** root_rank */, this->tensor_name_);

                done();
              }),
          done);
    });
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopDistBroadcastOp);
  std::string tensor_name_;
};  // namespace tensorflow

#define REGISTER_CPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("PopdistBroadcast").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PopDistBroadcastOp<T>);

TF_CALL_INTEGRAL_TYPES(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
#undef REGISTER_CPU
}  // namespace tensorflow
