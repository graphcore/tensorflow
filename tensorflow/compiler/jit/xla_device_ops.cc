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

#include "tensorflow/compiler/jit/xla_device_ops.h"

#include <memory>

#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_tensor.h"

namespace tensorflow {

XlaDeviceDummyOp::XlaDeviceDummyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaDeviceDummyOp::Compute(OpKernelContext* ctx) {
  LOG(FATAL) << "Attempted to execute Op " << name() << " type "
             << type_string() << " on an XLA device. This should never happen.";
}

XlaAssignVariableOp::XlaAssignVariableOp(OpKernelConstruction* c)
    : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
}

void XlaAssignVariableOp::Compute(OpKernelContext* context) {
  OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
              errors::InvalidArgument(
                  "Variable and value dtypes don't match; respectively, ",
                  DataTypeString(dtype_), " and ",
                  DataTypeString(context->input(1).dtype())));
  core::RefCountPtr<Var> variable;
  const Tensor& value = context->input(1);
  // Note: every resource-variable-manipulating op assumes copy-on-write
  // semantics, and creates a copy of the variable's Tensor if its refcount is
  // bigger than 1 when we try to modify it. This means we never need to copy
  // the original tensor for AssignVariableOp; even if there are other live
  // users of it we know none can modify it so this is always safe (even in
  // esoteric cases where the same tensor is used to initialize multiple
  // variables or the tensor is a constant this is safe, as future writes will
  // trigger copies).
  OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                              context, HandleFromInput(context, 0), &variable,
                              [this, &value](Var** ptr) {
                                *ptr = new Var(dtype_);
                                *(*ptr)->tensor() = value;
                                (*ptr)->is_initialized = true;
                                return Status::OK();
                              }));
  mutex_lock ml(*variable->mu());
  OP_REQUIRES(
      context,
      !variable->is_initialized || variable->tensor()->dtype() == dtype_,
      errors::InvalidArgument(
          "Trying to assign variable with wrong dtype. Expected ",
          DataTypeString(variable->tensor()->dtype()), " got ",
          DataTypeString(dtype_)));

  const bool is_ipu = DeviceType(static_cast<Device*>(context->device())
                                     ->attributes()
                                     .device_type()) == "IPU";
  if (is_ipu) {
    // With IPU the above comment does not apply as the IPU executor tries to
    // keep variables on the device and update variables - we therefore need to
    // copy the value via the host (XLA to XLA is not supported).
    DeviceContext* device_context = context->op_device_context();
    Device* device = static_cast<Device*>(context->device());
    Allocator* allocator = context->device()->GetAllocator({});

    // Create a host value and copy to it.
    Tensor host_value(value.dtype(), value.shape());
    OP_REQUIRES_OK(context,
                   device_context->CopyDeviceTensorToCPUSync(
                       &value, /*tensor_name=*/"", device, &host_value));
    // Copy from the host value to a new on device tensor.
    Tensor value_copy(allocator, value.dtype(), value.shape());
    OP_REQUIRES_OK(context, device_context->CopyCPUTensorToDeviceSync(
                                &host_value, device, &value_copy));

    *variable->tensor() = value_copy;
  } else {
    *variable->tensor() = value;
  }
  variable->is_initialized = true;
}

}  // namespace tensorflow
